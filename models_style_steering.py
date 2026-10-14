import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer, Sequential, Linear, ReLU
import math
from copy import deepcopy
from train_utils import full_to_partial_masking

def sinusoidal_positional_encoding(seq_len, d_model, device):
    """Standard sinusoidal PE (Vaswani et al., 2017)."""
    position = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) *
                         (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, seq_len, d_model)
# end sinusoidal_positional_encoding

class TransformerEncoderLayerWithAttn(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights = None  # place to store the weights

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        # same as parent forward, except we intercept attn_weights
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        if not self.training:
            self.last_attn_weights = attn_weights.detach()  # store for later

        # rest of the computation is copied from TransformerEncoderLayer
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
# end TransformerEncoderLayerWithAttn

# ========== Latent Clusters Model ==========
class SELaCModel(nn.Module):
    def __init__(
            self, 
            chord_vocab_size,  # V
            d_model=512, 
            nhead=8, 
            num_layers=8, 
            dim_feedforward=2048,
            pianoroll_dim=13,      # PCP + bars only
            grid_length=80,
            dropout=0.3,
            guidance_dim=None,
            device='cpu'
        ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.grid_length = grid_length

        # Melody projection: pianoroll_dim binary -> d_model
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=self.device)
        # self.melody_proj = nn.Embedding(chord_vocab_size, d_model, device=self.device)
        # Harmony token embedding: V -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=self.device)

        self.seq_len = grid_length + grid_length
        
        # # Positional embeddings (separate for clarity)
        self.shared_pos = sinusoidal_positional_encoding(
            grid_length, d_model, device
        )
        self.full_pos = torch.cat([self.shared_pos[:, :self.grid_length, :],
                            self.shared_pos[:, :self.grid_length, :]], dim=1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Transformer Encoder
        self.encoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            encoder_layer = TransformerEncoderLayerWithAttn(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
            self.encoder_layers.append( encoder_layer )
        # Optional: output head for harmonies
        self.output_head = nn.Linear(d_model, chord_vocab_size, device=self.device)
        # Layer norm at input and output
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.to(device)
    # end init

    def forward(
            self,
            melody_grid,
            harmony_tokens=None,
            get_layers_output=False,
            style_banks=None,
            steering_strengths=None,
            steer_mask=None
        ):
        """
        melody_grid: (B, grid_length, pianoroll_dim)
        harmony_tokens: (B, grid_length), e.g. all-mask tokens except bar tokens.

        style_banks: optional dict {layer_index: bank}, where bank has shape
            (N_style_examples, grid_length, d_model). Each bank contains harmony-side
            hidden states from target-style training examples using the same model/input setup.
        steering_strengths: float or dict {layer_index: float}, in [0, 1].
            0 leaves a layer unchanged; 1 replaces each eligible vector with its
            nearest target-style vector at the same grid position.
        steer_mask: optional bool tensor of shape (grid_length,). True means steer
            this position; use False for bar/control tokens if desired.
        get_layers_output: return harmony-side states after steering at each layer.
        """
        B = melody_grid.size(0)
        device = melody_grid.device

        melody_emb = self.melody_proj(melody_grid)
        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            # Legacy behavior. For style experiments, pass explicit mask-token IDs
            # rather than relying on this zero-vector placeholder.
            harmony_emb = torch.zeros(
                B, self.grid_length, self.d_model,
                device=device, dtype=melody_emb.dtype
            )

        full_seq = torch.cat([melody_emb, harmony_emb], dim=1)
        full_seq = full_seq + self.full_pos.to(device=device, dtype=melody_emb.dtype)
        full_seq = self.input_norm(full_seq)
        full_seq = self.dropout(full_seq)

        layers_output = {} if get_layers_output else None
        pre_intervention_output = {} if get_layers_output and style_banks else None
        encoded = full_seq
        harmony_slice = slice(-self.grid_length, None)

        for i, layer in enumerate(self.encoder_layers):
            encoded = layer(encoded)
            harmony_hidden = encoded[:, harmony_slice, :]

            # Nearest-neighbour target-style steering, applied after this layer and
            # before the next layer. Match each position only against that position
            # in the target-style bank, preserving grid alignment.
            if style_banks is not None and i in style_banks:
                bank = style_banks[i].to(device=device, dtype=encoded.dtype)
                if bank.dim() != 3 or bank.shape[1:] != (self.grid_length, self.d_model):
                    raise ValueError(
                        f"style_banks[{i}] must have shape (N, {self.grid_length}, {self.d_model}); "
                        f"got {tuple(bank.shape)}"
                    )
                q = F.normalize(harmony_hidden, p=2, dim=-1)
                keys = F.normalize(bank, p=2, dim=-1)
                # Similarity: (B, G, N); nearest target example at each grid position.
                similarities = torch.einsum("bgd,ngd->bgn", q, keys)
                nearest_idx = similarities.argmax(dim=-1)  # (B, G)
                pos = torch.arange(self.grid_length, device=device)[None, :]
                targets = bank[nearest_idx, pos]           # (B, G, D)

                strength = (steering_strengths.get(i, 0.0)
                            if isinstance(steering_strengths, dict)
                            else (0.0 if steering_strengths is None else steering_strengths))
                strength = float(strength)
                if not 0.0 <= strength <= 1.0:
                    raise ValueError("Steering strengths must be in [0, 1].")
                delta = targets - harmony_hidden
                if steer_mask is not None:
                    eligible = steer_mask.to(device=device, dtype=torch.bool).view(1, -1, 1)
                    delta = delta * eligible
                harmony_hidden_post = harmony_hidden + strength * delta
                encoded = torch.cat([encoded[:, :self.grid_length, :], harmony_hidden_post], dim=1)

            if get_layers_output and style_banks:
                pre_intervention_output[i] = harmony_hidden
                layers_output[i] = harmony_hidden_post
            elif get_layers_output:
                layers_output[i] = harmony_hidden

        encoded = self.output_norm(encoded)
        harmony_output = self.output_head(encoded[:, harmony_slice, :])
        if get_layers_output and style_banks:
            return harmony_output, layers_output, pre_intervention_output
        elif get_layers_output:
            return harmony_output, layers_output
        return harmony_output
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        for layer in self.encoder.layers:
            self_attns.append(layer.last_attn_weights)
        return self_attns
    # end get_attention_maps
# end class SELaCModel

# ================ Activation steering =======================
class SEASModel(nn.Module):
    def __init__(
            self, 
            chord_vocab_size,  # V
            d_model=512, 
            nhead=8, 
            num_layers=8, 
            dim_feedforward=2048,
            pianoroll_dim=13,      # PCP + bars only
            grid_length=80,
            dropout=0.3,
            guidance_dim=None,
            device='cpu'
        ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.grid_length = grid_length

        # Melody projection: pianoroll_dim binary -> d_model
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=self.device)
        # self.melody_proj = nn.Embedding(chord_vocab_size, d_model, device=self.device)
        # Harmony token embedding: V -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=self.device)

        self.seq_len = grid_length + grid_length
        
        # # Positional embeddings (separate for clarity)
        self.shared_pos = sinusoidal_positional_encoding(
            grid_length, d_model, device
        )
        self.full_pos = torch.cat([self.shared_pos[:, :self.grid_length, :],
                            self.shared_pos[:, :self.grid_length, :]], dim=1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Transformer Encoder
        self.encoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            encoder_layer = TransformerEncoderLayerWithAttn(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
            self.encoder_layers.append( encoder_layer )
        # Optional: output head for harmonies
        self.output_head = nn.Linear(d_model, chord_vocab_size, device=self.device)
        # Layer norm at input and output
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.to(device)
    # end init

    def forward(
            self,
            melody_grid,
            harmony_tokens=None,
            steering_vectors=None,
            alpha=1.0,
            get_layers_output=False
        ):
        """
        melody_grid: (B, grid_length, pianoroll_dim)
        harmony_tokens: (B, grid_length) - optional for training or inference
        """
        B = melody_grid.size(0)
        device = self.device

        # Project melody: (B, grid_length, pianoroll_dim) → (B, grid_length, d_model)
        melody_emb = self.melody_proj(melody_grid)

        # Harmony token embedding (optional for training): (B, grid_length) → (B, grid_length, d_model)
        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            # Placeholder (zeros) if not provided
            harmony_emb = torch.zeros(B, self.grid_length, self.d_model, device=device)

        # Concatenate full input: (B, 1 + grid_length + grid_length, d_model)
        full_seq = torch.cat([melody_emb, harmony_emb], dim=1)

        # Add positional encoding
        full_seq = full_seq + self.full_pos

        full_seq = self.input_norm(full_seq)
        full_seq = self.dropout(full_seq)

        if get_layers_output:
            layers_output = {}
        # Transformer encode
        encoded = full_seq
        for i, layer in enumerate(self.encoder_layers):
            encoded = layer(encoded)
            # capture layer output
            if get_layers_output:
                layers_output[i] = encoded[:, -self.grid_length:, :].mean(axis=1).unsqueeze(1)
            # guide process
            if steering_vectors is not None and i in steering_vectors:
                h = steering_vectors[i]
                # ensure correct shape
                if h.dim() == 2:  # (1, d_model)
                    h = h.unsqueeze(1)  # (1,1,d_model)
                encoded[:, -self.grid_length:, :] += alpha * h
        encoded = self.output_norm(encoded)

        # Optionally decode harmony logits (only last grid_length tokens)
        harmony_output = self.output_head(encoded[:, -self.grid_length:, :])  # (B, grid_length, V)
        
        if get_layers_output:
            return harmony_output, layers_output
        else:
            return harmony_output
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        for layer in self.encoder.layers:
            self_attns.append(layer.last_attn_weights)
        return self_attns
    # end get_attention_maps
# end class SEASModel

@torch.no_grad()
def collect_harmony_layer_states(
    model, dataloader, mask_token_id, bar_token_id, device=None
):
    """Collect unsteered harmony-side states for each layer.

    melody_batches and harmony_batches may each be a tensor or an iterable of batches.
    Returns {layer: Tensor(N, grid_length, d_model)} on CPU.
    The harmony batches should contain explicit mask-token IDs (and preserved bar IDs).
    """
    model.eval()
    device = device or next(model.parameters()).device

    all_states = {i: [] for i in range(len(model.encoder_layers))}
    all_harmony_ids = []
    all_valid_positions = []
    for batch in dataloader:
        harmony_ids = batch["harmony_ids"]
        # introduce masked input to the model
        harmony_input, _ = full_to_partial_masking(
            harmony_ids,
            mask_token_id,
            0,
            bar_token_id=bar_token_id
        )
        _, states = model(
            batch["pianoroll"].to(device),
            harmony_tokens=harmony_input.to(device),
            get_layers_output=True
        )
        for layer_idx, state in states.items():
            all_states[layer_idx].append(state.detach().cpu())
        all_harmony_ids.append( batch["harmony_ids"] )
        valid = (harmony_ids != bar_token_id) & (harmony_ids != mask_token_id)
        all_valid_positions.append(valid)
    return {i: torch.cat(chunks, dim=0) for i, chunks in all_states.items() if chunks}, torch.cat(all_harmony_ids), torch.cat(all_valid_positions)
# end collect_harmony_layer_states

@torch.no_grad()
def tokenwise_knn_accuracy(
    query_states,
    reference_states,
    reference_labels,
    query_labels,
    k=5,
    query_valid=None,
    reference_valid=None,
    metric="cosine",
):
    """Evaluate per-token style classification at matching grid positions.

    Args:
        query_states: {layer: Tensor(N_query, G, D)}.
        reference_states: {layer: Tensor(N_reference, G, D)}.
        reference_labels: Labels for reference pieces, shape (N_reference,).
        query_labels: Labels for query pieces, shape (N_query,).
        k: Number of nearest valid reference tokens to consider.
        query_valid: Optional bool Tensor(N_query, G). True means evaluate
            that query token.
        reference_valid: Optional bool Tensor(N_reference, G). True means
            that reference token may be used as a neighbour.
        metric: "cosine" or "euclidean".

    Returns:
        Dictionary keyed by layer, containing:
            token_accuracy: Accuracy over all evaluated query tokens.
            position_accuracy: List of accuracies by grid position. Invalid
                positions (with no evaluable queries/references) are NaN.
            position_counts: Number of evaluated query tokens per position.
            piece_accuracy: Accuracy per query piece; NaN for pieces with
                no evaluable tokens.
            piece_counts: Number of evaluated tokens per query piece.

    Notes:
        Reference states should come from training pieces only.
        Matching is performed only between tokens at the same grid position.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    if metric not in {"cosine", "euclidean"}:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    # Use CPU tensors for evaluation.
    ref_labels = torch.as_tensor(reference_labels, dtype=torch.long).cpu()
    qry_labels = torch.as_tensor(query_labels, dtype=torch.long).cpu()

    if ref_labels.ndim != 1 or qry_labels.ndim != 1:
        raise ValueError("reference_labels and query_labels must be 1-D")

    # Infer dimensions from the first layer.
    if not query_states:
        return {}

    first_layer = next(iter(query_states))
    n_query, grid, _ = query_states[first_layer].shape
    n_reference = reference_states[first_layer].shape[0]

    if len(qry_labels) != n_query:
        raise ValueError("query_labels length does not match query states")
    if len(ref_labels) != n_reference:
        raise ValueError("reference_labels length does not match reference states")

    def prepare_valid(mask, n_pieces, name):
        if mask is None:
            return torch.ones((n_pieces, grid), dtype=torch.bool)

        mask = torch.as_tensor(mask, dtype=torch.bool).cpu()
        if mask.shape != (n_pieces, grid):
            raise ValueError(
                f"{name} must have shape {(n_pieces, grid)}, "
                f"got {tuple(mask.shape)}"
            )
        return mask

    q_valid = prepare_valid(query_valid, n_query, "query_valid")
    r_valid = prepare_valid(reference_valid, n_reference, "reference_valid")

    # A stable label-to-index mapping avoids assumptions that labels are
    # contiguous non-negative integers.
    classes = torch.unique(torch.cat([ref_labels, qry_labels])).sort().values
    n_classes = len(classes)

    # Map original labels to indices in [0, n_classes).
    ref_class_idx = torch.searchsorted(classes, ref_labels)
    qry_class_idx = torch.searchsorted(classes, qry_labels)

    results = {}

    for layer_idx, q_all in query_states.items():
        if layer_idx not in reference_states:
            raise KeyError(f"Missing reference states for layer {layer_idx}")

        q_all = q_all.detach().cpu()
        r_all = reference_states[layer_idx].detach().cpu()

        if q_all.ndim != 3 or r_all.ndim != 3:
            raise ValueError("All state tensors must have shape (N, G, D)")
        if q_all.shape[1:] != r_all.shape[1:]:
            raise ValueError(
                f"Query/reference grid or feature dimensions differ "
                f"at layer {layer_idx}"
            )
        if q_all.shape[0] != n_query or r_all.shape[0] != n_reference:
            raise ValueError(f"Piece counts differ at layer {layer_idx}")

        piece_correct = torch.zeros(n_query, dtype=torch.long)
        piece_count = torch.zeros(n_query, dtype=torch.long)

        position_accuracy = []
        position_counts = []

        correct_total = 0
        count_total = 0

        for p in range(grid):
            # Validity is piece-specific: query and reference masks differ.
            q_idx = torch.where(q_valid[:, p])[0]
            r_idx = torch.where(r_valid[:, p])[0]

            n_valid_q = len(q_idx)
            n_valid_r = len(r_idx)

            # No query tokens to evaluate at this position.
            if n_valid_q == 0:
                position_accuracy.append(float("nan"))
                position_counts.append(0)
                continue

            # Cannot classify without any reference token at this position.
            if n_valid_r == 0:
                position_accuracy.append(float("nan"))
                position_counts.append(0)
                continue

            q = q_all[q_idx, p, :]
            r = r_all[r_idx, p, :]

            if metric == "cosine":
                q = F.normalize(q, dim=-1)
                r = F.normalize(r, dim=-1)
                distances = 1.0 - q @ r.T
            else:
                distances = torch.cdist(q, r, p=2)

            k_eff = min(k, n_valid_r)
            nn_idx = distances.topk(
                k_eff, largest=False, dim=1
            ).indices

            # Neighbour indices are local to the filtered reference set.
            nn_labels = ref_class_idx[r_idx][nn_idx]

            # Majority vote; ties resolve to the lowest class label.
            votes = F.one_hot(
                nn_labels, num_classes=n_classes
            ).sum(dim=1)
            pred_class_idx = votes.argmax(dim=-1)

            correct = pred_class_idx.eq(qry_class_idx[q_idx])

            # Aggregate only over valid query tokens.
            n_correct = int(correct.sum())
            correct_total += n_correct
            count_total += n_valid_q

            piece_correct[q_idx] += correct.long()
            piece_count[q_idx] += 1

            position_accuracy.append(float(correct.float().mean()))
            position_counts.append(n_valid_q)

        piece_accuracy = torch.full((n_query,), float("nan"))
        has_valid_tokens = piece_count > 0
        piece_accuracy[has_valid_tokens] = (
            piece_correct[has_valid_tokens].float()
            / piece_count[has_valid_tokens].float()
        )

        results[layer_idx] = {
            "token_accuracy": correct_total / count_total
                if count_total > 0 else float("nan"),
            "position_accuracy": position_accuracy,
            "position_counts": position_counts,
            "piece_accuracy": piece_accuracy.tolist(),
            "piece_counts": piece_count.tolist(),
        }

    return results

# end tokenwise_knn_accuracy

# def tokenwise_knn_accuracy(
#     query_states, reference_states, reference_labels, query_labels,
#     k=5, query_valid=None, reference_valid=None, metric="cosine"
# ):
#     """Evaluate per-token style classification, matching the same grid position.

#     query_states/reference_states: {layer: Tensor(N, G, D)}.
#     Labels are 1-D tensors/lists, e.g. 0=Nottingham, 1=Jazz.
#     Returns {layer: dict with overall token accuracy, per-position accuracy,
#     and per-piece accuracy}. Reference states must come from training data only.
#     """
#     if k < 1:
#         raise ValueError("k must be >= 1")
#     ref_labels = torch.as_tensor(reference_labels, dtype=torch.long)
#     qry_labels = torch.as_tensor(query_labels, dtype=torch.long)
#     results = {}
#     for layer_idx, q_all in query_states.items():
#         r_all = reference_states[layer_idx].cpu()
#         q_all = q_all.cpu()
#         n_q, grid, dim = q_all.shape
#         positions = (torch.ones(grid, dtype=torch.bool) if valid_positions is None
#                      else torch.as_tensor(valid_positions, dtype=torch.bool))
#         position_acc, correct_total, count_total = [], 0, 0
#         piece_correct = torch.zeros(n_q, dtype=torch.float32)
#         piece_count = 0
#         for p in range(grid):
#             if not positions[p]:
#                 continue
#             q = q_all[:, p, :]
#             r = r_all[:, p, :]
#             if metric == "cosine":
#                 q = F.normalize(q, dim=-1)
#                 r = F.normalize(r, dim=-1)
#                 distances = 1.0 - q @ r.T
#             elif metric == "euclidean":
#                 distances = torch.cdist(q, r)
#             else:
#                 raise ValueError("metric must be 'cosine' or 'euclidean'")
#             nn_idx = distances.topk(min(k, r.shape[0]), largest=False, dim=1).indices
#             nn_labels = ref_labels[nn_idx]
#             # Majority vote; ties resolve to the lower label.
#             votes = F.one_hot(nn_labels, num_classes=int(max(ref_labels.max(), qry_labels.max()).item()) + 1).sum(dim=1)
#             pred = votes.argmax(dim=-1)
#             correct = pred.eq(qry_labels)
#             correct_total += int(correct.sum())
#             count_total += n_q
#             piece_correct += correct.float()
#             piece_count += 1
#             position_acc.append(float(correct.float().mean()))
#         results[layer_idx] = {
#             "token_accuracy": correct_total / max(count_total, 1),
#             "position_accuracy": position_acc,
#             "piece_accuracy": (piece_correct / max(piece_count, 1)).tolist(),
#         }
#     return results
# # end tokenwise_knn_accuracy


@torch.no_grad()
def make_target_style_bank(layer_states, target_indices=None):
    """Select target-style training examples from {layer: Tensor(N, G, D)}."""
    if target_indices is None:
        return {i: x.detach().clone() for i, x in layer_states.items()}
    idx = torch.as_tensor(target_indices, dtype=torch.long)
    return {i: x[idx].detach().clone() for i, x in layer_states.items()}
