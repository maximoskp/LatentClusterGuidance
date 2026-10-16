import torch
import numpy as np
from train_utils import full_to_partial_masking

def get_results_for_individual_layer(
        model,
        device,
        valLoader,
        target_states,
        mask_token_id,
        bar_token_id
):
    strengths = {
        layer: 0
        for layer in range(len(model.encoder_layers))
    }
    results_inidividual_layer = {}
    layer_tested = 0
    print('layer tested: ', layer_tested)
    for strength in np.round(np.linspace(0, 1, 11), 1):
        print('strength value: ', strength)
        for layer in range(0, len(model.encoder_layers)):
            strengths[layer] = 0.0
        strengths[6] = strength
        print('strengths: ', strengths)

        model.eval()

        steered_states_all = {i: [] for i in range(len(model.encoder_layers))}
        pre_out_states_all = {i: [] for i in range(len(model.encoder_layers))}

        # nott melodies to jazz harmony
        for batch in valLoader:
            melodies = batch['pianoroll']
            harmony_ids = batch['harmony_ids']
            harmony_input, _ = full_to_partial_masking(
                harmony_ids,
                mask_token_id,
                0,
                bar_token_id=bar_token_id
            )
            # valid_positions = harmony_input[ harmony_input != mask_token_id ]

            with torch.no_grad():
                logits, steered_states, pre_out = model(
                    melodies.to(device),
                    harmony_tokens=harmony_input.to(device),
                    style_banks=target_states,
                    steering_strengths=strengths,
                    steer_mask=None,# valid_positions.to(device),
                    get_layers_output=True,
                )
                for layer_idx, state in steered_states.items():
                    steered_states_all[layer_idx].append(state.detach().cpu())
                for layer_idx, state in pre_out.items():
                    pre_out_states_all[layer_idx].append(state.detach().cpu())
                # end for
            # end with
        # end batch
        results_inidividual_layer[strength] = {
            'steered': {i: torch.cat(chunks, dim=0) for i, chunks in steered_states_all.items() if chunks},
            'pre': {i: torch.cat(chunks, dim=0) for i, chunks in pre_out_states_all.items() if chunks}
        }
    return results_inidividual_layer
# end get_results_for_individual_layer

@torch.no_grad()
def latent_steering_progress(
    target_states,
    original_states,
    generated_states,
    eps=1e-12,
):
    """
    Compute per-piece, per-position steering progress toward a target-style bank.

    Parameters
    ----------
    target_states : dict[int, Tensor]
        Target bank, e.g. jazz_train_states.
        Each tensor has shape [num_target_pieces, num_steps, model_dim].

    original_states : dict[int, Tensor]
        Unsteered latent states for the generated pieces.
        Each tensor has shape [num_generated_pieces, num_steps, model_dim].

    generated_states : dict[int, Tensor]
        Latent states to evaluate, either `steered` or `pre`.
        Same shape as original_states.

    eps : float
        Numerical tolerance for zero-distance cases.

    Returns
    -------
    dict[int, np.ndarray]
        One array per layer, shape
        [num_generated_pieces, num_steps], with values in [0, 1].

        0 = identical to the original latent.
        1 = identical to the selected target neighbor.
    """
    results = {}

    for layer in original_states:
        # Use the original latent to select a fixed target neighbor.
        original = torch.as_tensor(
            original_states[layer]
        ).detach().float()

        generated = torch.as_tensor(
            generated_states[layer]
        ).detach().float()

        bank = torch.as_tensor(
            target_states[layer]
        ).detach().to(device=original.device, dtype=torch.float32)

        if original.shape != generated.shape:
            raise ValueError(
                f"Layer {layer}: original and generated shapes differ."
            )

        if original.ndim != 3 or bank.ndim != 3:
            raise ValueError(
                "All state tensors must have shape [pieces, steps, dim]."
            )

        if original.shape[1:] != bank.shape[1:]:
            raise ValueError(
                f"Layer {layer}: target bank and query dimensions differ."
            )

        # Normalize for cosine-similarity-based neighbor selection.
        # q = F.normalize(original, p=2, dim=-1)
        q = F.normalize(generated, p=2, dim=-1)
        k = F.normalize(bank, p=2, dim=-1)

        # Similarities: [generated pieces, steps, target pieces].
        similarities = torch.einsum("pgd,ngd->pgn", q, k)

        # Select one target neighbor per generated piece and position.
        nearest_idx = similarities.argmax(dim=-1)  # [P, G]

        # Gather the selected target vectors: [P, G, D].
        positions = torch.arange(
            original.shape[1], device=original.device
        ).unsqueeze(0)

        targets = bank[nearest_idx, positions]

        # Euclidean distances to the fixed target neighbor.
        d_original = torch.linalg.vector_norm(
            original - targets, dim=-1
        )
        d_generated = torch.linalg.vector_norm(
            generated - targets, dim=-1
        )

        # Relative position between original and target, and generated
        # and target. Exact original -> 0; exact target -> 1.
        d_from_original = torch.linalg.vector_norm(
            generated - original, dim=-1
        )

        denominator = d_from_original + d_generated

        score = torch.where(
            denominator > eps,
            d_from_original / denominator.clamp_min(eps),
            torch.zeros_like(denominator),
        )

        results[layer] = score.clamp(0.0, 1.0).cpu().numpy()

    return results
# end latent_steering_progress