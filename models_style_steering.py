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

# ========== Nearest-Neighbor Activation Steering Model ==========
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
            bank_key=None,
            num_neighbors=1,
            steering_strengths=None,
            steer_mask=None
        ):
        """Run the model, optionally steering harmony-side hidden states.

        Args:
            melody_grid: Tensor (B, grid_length, pianoroll_dim).
            harmony_tokens: Tensor (B, grid_length), e.g. mask tokens except bar tokens.
            style_banks: Optional dict {layer_index: bank}. The bank rank selects
                the retrieval strategy:
                  * (N_pieces, grid_length, d_model): position-constrained NN;
                    each query position searches only references at that position.
                  * (N_reference_states, d_model): position-independent NN; each
                    query position searches every reference state in the flattened bank.
                  * (d_model,): global-vector steering; every query position is
                    steered toward this single vector (no nearest-neighbour search).
                Banks should contain states extracted from target-style training examples
                using the same model and input/masking setup.
            steering_strengths: Float or dict {layer_index: float} in [0, 1].
                0 leaves a layer unchanged; 1 fully replaces eligible states with targets.
            steer_mask: Optional bool Tensor (grid_length,); True means steer that position.
                Useful for excluding bar/control tokens.
            get_layers_output: If True, also return post-intervention states by layer and
                pre-intervention states (third return value; None if no style banks supplied).

        Returns:
            Without get_layers_output: (harmony_logits, None, None).
            With get_layers_output: (harmony_logits, post_states, pre_states).
        """
        B = melody_grid.size(0)
        device = melody_grid.device

        melody_emb = self.melody_proj(melody_grid)
        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            # For style experiments, explicit mask-token IDs are recommended.
            harmony_emb = torch.zeros(
                B, self.grid_length, self.d_model,
                device=device, dtype=melody_emb.dtype
            )

        full_seq = torch.cat([melody_emb, harmony_emb], dim=1)
        full_seq = full_seq + self.full_pos.to(device=device, dtype=melody_emb.dtype)
        full_seq = self.input_norm(full_seq)
        full_seq = self.dropout(full_seq)

        layers_output = {} if get_layers_output else None
        has_banks = style_banks is not None and len(style_banks) > 0
        pre_intervention_output = {} if get_layers_output and has_banks else None
        encoded = full_seq
        harmony_slice = slice(-self.grid_length, None)

        for i, layer in enumerate(self.encoder_layers):
            encoded = layer(encoded)
            harmony_hidden = encoded[:, harmony_slice, :]
            harmony_hidden_post = harmony_hidden

            if style_banks is not None and i in style_banks:
                bank = torch.as_tensor(
                    style_banks[i][bank_key], device=device, dtype=encoded.dtype
                )
                q = harmony_hidden
                if bank.ndim == 3:
                    # Position-constrained retrieval: bank (N, G, D), query (B, G, D).
                    if bank.shape[1:] != (self.grid_length, self.d_model):
                        raise ValueError(
                            f"3D style_banks[{i}][{bank_key}] must have shape "
                            f"(N, {self.grid_length}, {self.d_model}); got {tuple(bank.shape)}"
                        )
                    q_norm = F.normalize(q, p=2, dim=-1)
                    bank_norm = F.normalize(bank, p=2, dim=-1)
                    # Similarity (B, G, N): each position searches its matching position.
                    similarities = torch.einsum("bgd,ngd->bgn", q_norm, bank_norm)
                    # nearest_idx = similarities.argmax(dim=-1)  # (B, G)
                    # pos = torch.arange(self.grid_length, device=device)[None, :]
                    # targets = bank[nearest_idx, pos]           # (B, G, D)
                    k = min(num_neighbors, similarities.shape[-1])
                    top_idx = similarities.topk(k, dim=-1).indices  # (B, G, k)
                    pos = torch.arange(self.grid_length, device=device)[None, :, None]
                    neighbors = bank[top_idx, pos]                  # (B, G, k, D)
                    targets = neighbors.mean(dim=2)                 # (B, G, D)
                elif bank.ndim == 2:
                    # Position-independent retrieval: each query state searches all rows.
                    if bank.shape[-1] != self.d_model or bank.shape[0] < 1:
                        raise ValueError(
                            f"2D style_banks[{i}][{bank_key}] must have shape (N_reference_states, "
                            f"{self.d_model}) with N_reference_states >= 1; got {tuple(bank.shape)}"
                        )
                    q_flat = q.reshape(-1, self.d_model)
                    q_norm = F.normalize(q_flat, p=2, dim=-1)
                    bank_norm = F.normalize(bank, p=2, dim=-1)
                    similarities = q_norm @ bank_norm.transpose(0, 1)  # (B*G, N_ref)
                    # nearest_idx = similarities.argmax(dim=-1)
                    # targets = bank[nearest_idx].reshape(B, self.grid_length, self.d_model)
                    k = min(num_neighbors, similarities.shape[-1])
                    top_idx = similarities.topk(k, dim=-1).indices  # (B*G, k)
                    neighbors = bank[top_idx].reshape(B, self.grid_length, k, self.d_model) # (B, G, k, D)
                    targets = neighbors.mean(dim=2)                 # (B, G, D)
                elif bank.ndim == 1:
                    # Global-vector steering: same target vector for every position.
                    if bank.shape[0] != self.d_model:
                        raise ValueError(
                            f"1D style_banks[{i}][{bank_key}] must have shape ({self.d_model},); "
                            f"got {tuple(bank.shape)}"
                        )
                    targets = bank.view(1, 1, self.d_model).expand(B, self.grid_length, -1)
                else:
                    raise ValueError(
                        f"style_banks[{i}][{bank_key}] must be 1D, 2D, or 3D; got shape {tuple(bank.shape)}"
                    )

                strength = (
                    steering_strengths.get(i, 0.0)
                    if isinstance(steering_strengths, dict)
                    else (0.0 if steering_strengths is None else steering_strengths)
                )
                strength = float(strength)
                if not 0.0 <= strength <= 1.0:
                    raise ValueError("Steering strengths must be in [0, 1].")

                delta = targets - harmony_hidden
                if steer_mask is not None:
                    eligible = torch.as_tensor(
                        steer_mask, device=device, dtype=torch.bool
                    )
                    if eligible.shape != (self.grid_length,):
                        raise ValueError(
                            f"steer_mask must have shape ({self.grid_length},); "
                            f"got {tuple(eligible.shape)}"
                        )
                    delta = delta * eligible.view(1, -1, 1)

                harmony_hidden_post = harmony_hidden + strength * delta
                encoded = torch.cat(
                    [encoded[:, :self.grid_length, :], harmony_hidden_post], dim=1
                )

            if get_layers_output:
                layers_output[i] = harmony_hidden_post
                if has_banks:
                    pre_intervention_output[i] = harmony_hidden

        encoded = self.output_norm(encoded)
        harmony_output = self.output_head(encoded[:, harmony_slice, :])
        if get_layers_output:
            return harmony_output, layers_output, pre_intervention_output
        return harmony_output, None, None
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
    model, dataloader, mask_token_id, bar_token_id, device=None,
    include_raw_states=False,
):
    """Collect layer states and build several reference-bank formats in one pass.

    The model is run on harmony inputs where chord tokens are masked and bar
    tokens are preserved. For each layer, the returned bank dictionary contains:

        state_banks[layer]["position_bank"]: (N_pieces, G, D)
            Position-constrained bank. Grid positions are preserved.

        state_banks[layer]["flatten_bank"]: (N_pieces * G, D)
            Position-independent bank containing every grid position, in
            piece-major then position-major order. This matches a literal flatten
            of position_bank and includes control-token positions.

        state_banks[layer]["valid_flatten_bank"]: (N_valid_states, D)
            Position-independent bank containing only valid chord positions.
            Prefer this bank for retrieval when bar/control states should not be
            nearest-neighbour candidates.

        state_banks[layer]["global_vector"]: (D,)
            Mean of all valid chord-position states at that layer.

        state_banks[layer]["valid_flat_indices"]: (N_valid_states, 2)
            Optional-use metadata: [piece_index, grid_position] for each row in
            valid_flatten_bank, in the same order as that filtered bank.

        state_banks[layer]["valid_positions"]: (N_pieces, G)
            Boolean mask indicating valid chord positions in the original labels.

        state_banks[layer]["raw_states"]: (N_pieces, G, D), optional
            Included only if include_raw_states=True. This is the same tensor as
            position_bank, provided for readability/compatibility in analysis.

    Returns:
        state_banks, harmony_ids, valid_positions

    Notes:
        - All collected tensors are returned on CPU.
        - The global vector is a mean over valid token states (not a mean of
          per-piece means), so pieces with more valid chord positions contribute
          proportionally more.
        - `flatten_bank` has exactly N*G rows. `valid_flatten_bank` has only
          valid chord-position rows and is generally the safer unrestricted bank.
        - `position_bank` retains the rectangular grid, including states at
          control-token positions. Use a query steer_mask to avoid intervening
          on control positions. If control tokens can occur inconsistently at
          the same grid position across pieces, consider adding reference-valid
          masking to the position-constrained retrieval implementation.
    """
    model.eval()
    device = device or next(model.parameters()).device

    all_states = {i: [] for i in range(len(model.encoder_layers))}
    all_harmony_ids = []
    all_valid_positions = []

    for batch in dataloader:
        harmony_ids = batch["harmony_ids"]
        harmony_input, _ = full_to_partial_masking(
            harmony_ids,
            mask_token_id,
            0,
            bar_token_id=bar_token_id,
        )
        _, states, _ = model(
            batch["pianoroll"].to(device),
            harmony_tokens=harmony_input.to(device),
            get_layers_output=True,
        )

        for layer_idx, state in states.items():
            all_states[layer_idx].append(state.detach().cpu())

        all_harmony_ids.append(harmony_ids.detach().cpu())
        valid = (harmony_ids != bar_token_id) & (harmony_ids != mask_token_id)
        all_valid_positions.append(valid.detach().cpu())

    harmony_ids = torch.cat(all_harmony_ids, dim=0)
    valid_positions = torch.cat(all_valid_positions, dim=0).bool()

    state_banks = {}
    for layer_idx, chunks in all_states.items():
        if not chunks:
            continue

        position_bank = torch.cat(chunks, dim=0)  # (N, G, D)
        if position_bank.ndim != 3:
            raise ValueError(
                f"Expected layer {layer_idx} states to have shape (N, G, D); "
                f"got {tuple(position_bank.shape)}"
            )
        if valid_positions.shape != position_bank.shape[:2]:
            raise ValueError(
                f"Validity mask shape {tuple(valid_positions.shape)} does not match "
                f"layer {layer_idx} grid shape {tuple(position_bank.shape[:2])}"
            )

        # Literal flatten preserves every grid position and has shape (N*G, D).
        flatten_bank = position_bank.reshape(-1, position_bank.shape[-1])

        # The filtered alternative prevents bar/control states from becoming
        # nearest-neighbour targets in position-independent retrieval.
        valid_flatten_bank = position_bank[valid_positions]  # (N_valid_states, D)
        if valid_flatten_bank.shape[0] == 0:
            raise ValueError(
                f"No valid chord positions found while building bank for layer {layer_idx}"
            )

        # Record where each valid flattened row originated.
        valid_flat_indices = valid_positions.nonzero(as_tuple=False)
        # Indices for every state, including bar and mask positions.
        # Shape: (N_pieces * grid_length, 2).
        # Each row is [piece_index, grid_position].
        n_pieces, grid_length, _ = position_bank.shape
        flat_indices = torch.cartesian_prod(
            torch.arange(n_pieces),
            torch.arange(grid_length),
        )

        bank = {
            "position_bank": position_bank,
            "flatten_bank": flatten_bank.contiguous(),
            "valid_flatten_bank": valid_flatten_bank.contiguous(),
            "global_vector": valid_flatten_bank.mean(dim=0),
            "valid_flat_indices": valid_flat_indices,
            "flat_indices": flat_indices,
            "valid_positions": valid_positions,
        }
        if include_raw_states:
            bank["raw_states"] = position_bank
        state_banks[layer_idx] = bank

    return state_banks, harmony_ids, valid_positions
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
# endj tokenwise_knn_accuracy



import torch
import torch.nn.functional as F


@torch.no_grad()
def statewise_knn_accuracy(
    query_states,
    reference_states,
    query_state_labels,
    reference_state_labels,
    query_piece_indices,
    query_piece_labels,
    k=5,
    metric="cosine",
    chunk_size=2048,
):
    """
    Position-independent k-NN classification of individual states,
    with per-piece accuracy aggregation.

    Args:
        query_states: {layer: Tensor(N_query_states, D)}.
        reference_states: {layer: Tensor(N_reference_states, D)}.
        query_state_labels: Style label for each query state.
        reference_state_labels: Style label for each reference state.
        query_piece_indices: Original piece index for each query state.
        query_piece_labels: Style label for each query piece.
        k: Number of nearest reference states.
        metric: "cosine" or "euclidean".
        chunk_size: Number of query states processed per batch.

    Returns:
        Dictionary keyed by layer, containing:
            state_accuracy: Accuracy over all query states.
            piece_accuracy: Accuracy per query piece (NaN if no states).
            piece_counts: Number of query states per piece.
            piece_labels: Style label per query piece.
            nott_accuracy: Mean per-piece accuracy for label 0.
            jazz_accuracy: Mean per-piece accuracy for label 1.
            per_class_accuracy: Mean per-piece accuracy for each class.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if metric not in {"cosine", "euclidean"}:
        raise ValueError("metric must be 'cosine' or 'euclidean'")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    q_labels = torch.as_tensor(
        query_state_labels, dtype=torch.long
    ).cpu()
    r_labels = torch.as_tensor(
        reference_state_labels, dtype=torch.long
    ).cpu()
    piece_indices = torch.as_tensor(
        query_piece_indices, dtype=torch.long
    ).cpu()
    piece_labels = torch.as_tensor(
        query_piece_labels, dtype=torch.long
    ).cpu()

    if q_labels.ndim != 1 or r_labels.ndim != 1:
        raise ValueError("State labels must be 1-D")
    if piece_indices.ndim != 1 or piece_labels.ndim != 1:
        raise ValueError("Piece indices and labels must be 1-D")
    if len(q_labels) != len(piece_indices):
        raise ValueError(
            "query_state_labels and query_piece_indices must have equal length"
        )
    if len(piece_indices) > 0 and (
        piece_indices.min() < 0 or piece_indices.max() >= len(piece_labels)
    ):
        raise ValueError("query_piece_indices contains an invalid piece index")

    classes = torch.unique(
        torch.cat([q_labels, r_labels])
    ).sort().values
    n_classes = len(classes)

    r_class_idx = torch.searchsorted(classes, r_labels)
    q_class_idx = torch.searchsorted(classes, q_labels)

    # Verify that each state has the style label of its parent piece.
    if len(piece_indices) > 0:
        if not torch.equal(q_labels, piece_labels[piece_indices]):
            raise ValueError(
                "Some query-state labels do not match their piece labels"
            )

    n_pieces = len(piece_labels)
    results = {}

    for layer, q in query_states.items():
        if layer not in reference_states:
            raise KeyError(f"Missing reference layer {layer}")

        q = q.detach().float().cpu()
        r = reference_states[layer].detach().float().cpu()

        if q.ndim != 2 or r.ndim != 2:
            raise ValueError(
                "State tensors must have shape (N_states, D)"
            )
        if q.shape[1] != r.shape[1]:
            raise ValueError("Query/reference feature dimensions differ")
        if len(q) != len(q_labels) or len(r) != len(r_labels):
            raise ValueError("State-label count does not match bank size")
        if len(r) == 0:
            raise ValueError("Reference bank is empty")
        if k > len(r):
            raise ValueError("k cannot exceed the reference bank size")

        if metric == "cosine":
            q = F.normalize(q, dim=-1)
            r = F.normalize(r, dim=-1)

        correct_by_piece = torch.zeros(n_pieces, dtype=torch.long)
        count_by_piece = torch.zeros(n_pieces, dtype=torch.long)
        all_correct = []

        for start in range(0, len(q), chunk_size):
            q_batch = q[start:start + chunk_size]

            if metric == "cosine":
                distances = 1.0 - q_batch @ r.T
            else:
                distances = torch.cdist(q_batch, r, p=2)

            nn_idx = distances.topk(
                k, largest=False, dim=1
            ).indices

            nn_labels = r_class_idx[nn_idx]
            votes = F.one_hot(
                nn_labels, num_classes=n_classes
            ).sum(dim=1)

            pred_class_idx = votes.argmax(dim=-1)
            correct = pred_class_idx.eq(
                q_class_idx[start:start + len(q_batch)]
            )

            all_correct.append(correct)

            batch_piece_indices = piece_indices[
                start:start + len(q_batch)
            ]

            # Aggregate correct predictions and counts per piece.
            correct_by_piece.scatter_add_(
                0, batch_piece_indices, correct.long()
            )
            count_by_piece.scatter_add_(
                0, batch_piece_indices, torch.ones_like(
                    batch_piece_indices
                )
            )

        piece_accuracy = torch.full(
            (n_pieces,), float("nan"), dtype=torch.float
        )
        has_states = count_by_piece > 0
        piece_accuracy[has_states] = (
            correct_by_piece[has_states].float()
            / count_by_piece[has_states].float()
        )

        state_correct = (
            torch.cat(all_correct)
            if all_correct
            else torch.empty(0, dtype=torch.bool)
        )

        per_class_accuracy = {}
        for label in classes.tolist():
            mask = (piece_labels == label) & has_states
            per_class_accuracy[label] = (
                float(piece_accuracy[mask].mean())
                if mask.any() else float("nan")
            )

        def mean_piece_accuracy(label):
            mask = (piece_labels == label) & has_states
            return (
                float(piece_accuracy[mask].mean())
                if mask.any() else float("nan")
            )

        results[layer] = {
            "state_accuracy": (
                float(state_correct.float().mean())
                if len(state_correct) else float("nan")
            ),
            "piece_accuracy": piece_accuracy.tolist(),
            "piece_counts": count_by_piece.tolist(),
            "piece_labels": piece_labels.tolist(),
            "per_class_accuracy": per_class_accuracy,
            "nott_accuracy": mean_piece_accuracy(0),
            "jazz_accuracy": mean_piece_accuracy(1),
        }

    return results
# end statewise_knn_accuracy

# @torch.no_grad()
# def statewise_knn_accuracy(
#     query_states,
#     reference_states,
#     query_state_labels,
#     reference_state_labels,
#     k=5,
#     metric="cosine",
#     chunk_size=2048,
# ):
#     """
#     Position-independent k-NN classification of individual states.

#     Args:
#         query_states: {layer: Tensor(N_query_states, D)}.
#         reference_states: {layer: Tensor(N_reference_states, D)}.
#         query_state_labels: Tensor(N_query_states,).
#         reference_state_labels: Tensor(N_reference_states,).
#         k: Number of nearest reference states.
#         metric: "cosine" or "euclidean".
#         chunk_size: Number of query states processed per batch.

#     Returns:
#         Dictionary keyed by layer, with accuracy and per-class accuracy.
#     """
#     if k < 1:
#         raise ValueError("k must be >= 1")
#     if metric not in {"cosine", "euclidean"}:
#         raise ValueError("metric must be 'cosine' or 'euclidean'")

#     q_labels = torch.as_tensor(
#         query_state_labels, dtype=torch.long
#     ).cpu()
#     r_labels = torch.as_tensor(
#         reference_state_labels, dtype=torch.long
#     ).cpu()

#     if q_labels.ndim != 1 or r_labels.ndim != 1:
#         raise ValueError("State labels must be 1-D")

#     classes = torch.unique(
#         torch.cat([q_labels, r_labels])
#     ).sort().values

#     r_class_idx = torch.searchsorted(classes, r_labels)
#     q_class_idx = torch.searchsorted(classes, q_labels)

#     results = {}

#     for layer, q in query_states.items():
#         if layer not in reference_states:
#             raise KeyError(f"Missing reference layer {layer}")

#         q = q.detach().float().cpu()
#         r = reference_states[layer].detach().float().cpu()

#         if q.ndim != 2 or r.ndim != 2:
#             raise ValueError(
#                 "State tensors must have shape (N_states, D)"
#             )
#         if q.shape[1] != r.shape[1]:
#             raise ValueError("Query/reference feature dimensions differ")
#         if len(q) != len(q_labels) or len(r) != len(r_labels):
#             raise ValueError("State-label count does not match bank size")

#         if k > len(r):
#             raise ValueError("k cannot exceed the reference bank size")

#         if metric == "cosine":
#             q = F.normalize(q, dim=-1)
#             r = F.normalize(r, dim=-1)

#         predictions = []

#         for start in range(0, len(q), chunk_size):
#             q_batch = q[start:start + chunk_size]

#             if metric == "cosine":
#                 distances = 1.0 - q_batch @ r.T
#             else:
#                 distances = torch.cdist(q_batch, r, p=2)

#             nn_idx = distances.topk(
#                 k, largest=False, dim=1
#             ).indices

#             nn_labels = r_class_idx[nn_idx]
#             votes = F.one_hot(
#                 nn_labels, num_classes=len(classes)
#             ).sum(dim=1)

#             predictions.append(votes.argmax(dim=-1))

#         pred_class_idx = torch.cat(predictions)
#         correct = pred_class_idx.eq(q_class_idx)

#         per_class_accuracy = {}
#         for class_idx, label in enumerate(classes.tolist()):
#             mask = q_labels == label
#             per_class_accuracy[label] = (
#                 float(correct[mask].float().mean())
#                 if mask.any() else float("nan")
#             )

#         piece_accuracy = torch.full((n_query,), float("nan"))
#         has_valid_tokens = piece_count > 0
#         piece_accuracy[has_valid_tokens] = (
#             piece_correct[has_valid_tokens].float()
#             / piece_count[has_valid_tokens].float()
#         )

#         results[layer] = {
#             "token_accuracy": float(correct.float().mean()),
#             "per_class_accuracy": per_class_accuracy,
#             "n_query_states": len(q),
#             "n_reference_states": len(r),
#         }

#     return results
# # end statewise_knn_accuracy


@torch.no_grad()
def make_target_style_bank(layer_states, target_indices=None):
    """Select target-style training examples from {layer: Tensor(N, G, D)}."""
    if target_indices is None:
        return {i: x.detach().clone() for i, x in layer_states.items()}
    idx = torch.as_tensor(target_indices, dtype=torch.long)
    return {i: x[idx].detach().clone() for i, x in layer_states.items()}
