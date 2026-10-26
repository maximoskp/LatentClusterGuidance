import torch
import numpy as np
from train_utils import full_to_partial_masking
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
# def latent_steering_progress(
#     target_states,
#     original_states,
#     generated_states,
#     bank_key,
#     eps=1e-12,
# ):
#     """
#     Compute per-piece, per-position steering progress toward a target-style bank.

#     Parameters
#     ----------
#     target_states : dict[int, Tensor]
#         Target bank, e.g. jazz_train_states.
#         Each tensor has shape [num_target_pieces, num_steps, model_dim].

#     original_states : dict[int, Tensor]
#         Unsteered latent states for the generated pieces.
#         Each tensor has shape [num_generated_pieces, num_steps, model_dim].

#     generated_states : dict[int, Tensor]
#         Latent states to evaluate, either `steered` or `pre`.
#         Same shape as original_states.

#     eps : float
#         Numerical tolerance for zero-distance cases.

#     Returns
#     -------
#     dict[int, np.ndarray]
#         One array per layer, shape
#         [num_generated_pieces, num_steps], with values in [0, 1].

#         0 = identical to the original latent.
#         1 = identical to the selected target neighbor.
#     """
#     results = {}

#     for layer in original_states:
#         # Use the original latent to select a fixed target neighbor.
#         original = torch.as_tensor(
#             original_states[layer]
#         ).detach().float()

#         generated = torch.as_tensor(
#             generated_states[layer]
#         ).detach().float()

#         bank = torch.as_tensor(
#             target_states[layer][bank_key]
#         ).detach().to(device=original.device, dtype=torch.float32)

#         if original.shape != generated.shape:
#             raise ValueError(
#                 f"Layer {layer}: original and generated shapes differ."
#             )

#         if original.ndim != 3 or bank.ndim != 3:
#             raise ValueError(
#                 "All state tensors must have shape [pieces, steps, dim]."
#             )

#         if original.shape[1:] != bank.shape[1:]:
#             raise ValueError(
#                 f"Layer {layer}: target bank and query dimensions differ."
#             )

#         # Normalize for cosine-similarity-based neighbor selection.
#         # q = F.normalize(original, p=2, dim=-1)
#         q = F.normalize(generated, p=2, dim=-1)
#         k = F.normalize(bank, p=2, dim=-1)

#         # Similarities: [generated pieces, steps, target pieces].
#         similarities = torch.einsum("pgd,ngd->pgn", q, k)

#         # Select one target neighbor per generated piece and position.
#         nearest_idx = similarities.argmax(dim=-1)  # [P, G]

#         # Gather the selected target vectors: [P, G, D].
#         positions = torch.arange(
#             original.shape[1], device=original.device
#         ).unsqueeze(0)

#         targets = bank[nearest_idx, positions]

#         # Euclidean distances to the fixed target neighbor.
#         d_original = torch.linalg.vector_norm(
#             original - targets, dim=-1
#         )
#         d_generated = torch.linalg.vector_norm(
#             generated - targets, dim=-1
#         )

#         # Relative position between original and target, and generated
#         # and target. Exact original -> 0; exact target -> 1.
#         d_from_original = torch.linalg.vector_norm(
#             generated - original, dim=-1
#         )

#         denominator = d_from_original + d_generated

#         score = torch.where(
#             denominator > eps,
#             d_from_original / denominator.clamp_min(eps),
#             torch.zeros_like(denominator),
#         )

#         results[layer] = score.clamp(0.0, 1.0).cpu().numpy()

#     return results
# # end latent_steering_progress

def latent_steering_progress(
    target_states,
    original_states,
    generated_states,
    bank_key,
    eps=1e-12,
):
    """
    Compute per-piece, per-position steering progress toward a target-style bank.

    Parameters
    ----------
    target_states : dict[int, Tensor]
        Target bank, e.g. jazz_train_states.
        Each tensor has shape [num_target_pieces, num_steps, model_dim],
        [num_target_pieces * num_steps, model_dim], or [model_dim].

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
            target_states[layer][bank_key]
        ).detach().to(device=original.device, dtype=torch.float32)

        if original.shape != generated.shape:
            raise ValueError(
                f"Layer {layer}: original and generated shapes differ."
            )

        # if original.ndim != 3 or bank.ndim != 3:
        #     raise ValueError(
        #         "All state tensors must have shape [pieces, steps, dim]."
        #     )

        # if original.shape[1:] != bank.shape[1:]:
        #     raise ValueError(
        #         f"Layer {layer}: target bank and query dimensions differ."
        #     )

        # Normalize for cosine-similarity-based neighbor selection.
        # q = F.normalize(original, p=2, dim=-1)
        q = F.normalize(generated, p=2, dim=-1)
        k = F.normalize(bank, p=2, dim=-1)
        
        if k.shape[-1] != q.shape[-1]:
            raise ValueError(
                f"Layer {layer}: query and target dimensions differ."
            )

        if k.ndim == 3:
            if k.shape[1] != q.shape[1]:
                raise ValueError(
                    f"Layer {layer}: query and target step counts differ."
                )
            similarities = torch.einsum("pgd,ngd->pgn", q, k)
            nearest_idx = similarities.argmax(dim=-1)
            positions = torch.arange(
                q.shape[1], device=q.device
            ).unsqueeze(0)
            targets = bank[nearest_idx, positions]
        elif k.ndim == 2:
            similarities = torch.einsum("pgd,nd->pgn", q, k)
            nearest_idx = similarities.argmax(dim=-1)
            targets = bank[nearest_idx]
        elif k.ndim == 1:
            similarities = torch.einsum("pgd,d->pg", q, k).unsqueeze(-1)
            targets = bank
        else:
            raise ValueError(
                f"Layer {layer}: target bank must be 1D, 2D, or 3D; "
                f"got shape {tuple(k.shape)}."
            )

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

def single_layer_latent_influence(
    model,
    device,
    valLoader,
    target_train_states,
    bank_key,
    num_neighbors,
    mask_token_id,
    bar_token_id,
    examined_strengths,
    num_layers=8
):
    strength_values_num = len(examined_strengths)
    individual_layer_results = np.zeros((num_layers-1, strength_values_num, num_layers))
    for starting_layer in range(num_layers-1):
        # Test different strengths for layers
        # The unsteered version corresponds to strength 0
        # initialize all strengths as zero
        strengths = {
            layer: 0.0
            for layer in range(len(model.encoder_layers))
        }
        results_inidividual_layer = {}
        layer_tested = 0
        print('layer tested: ', layer_tested)
        for strength in examined_strengths:
            print('strength value: ', strength)
            for layer in range(0, len(model.encoder_layers)):
                strengths[layer] = 0.0
            strengths[starting_layer] = strength
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
                valid_positions = harmony_input[ harmony_input != mask_token_id ]

                with torch.no_grad():
                    _, states_output = model(
                        melodies.to(device),
                        harmony_tokens=harmony_input.to(device),
                        style_banks=target_train_states,
                        bank_key=bank_key,
                        num_neighbors=num_neighbors,
                        steering_strengths=strengths,
                        steer_mask=None,# valid_positions.to(device),
                        get_layers_output=True,
                    )
                    steered_states = states_output['layers_output']
                    pre_out = states_output['pre_intervention_output']
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

        strength_idx = 0
        for strength in np.round(np.linspace(0, 1, 11), 1):
            # Evaluate the post-intervention states.
            scores_post = latent_steering_progress(
                target_states=target_train_states,
                bank_key=bank_key,
                original_states=results_inidividual_layer[0]['steered'],
                generated_states=results_inidividual_layer[strength]['steered'],
            )

            # Evaluate the pre-intervention states.
            scores_pre = latent_steering_progress(
                target_states=target_train_states,
                bank_key=bank_key,
                original_states=results_inidividual_layer[0]['steered'],
                generated_states=results_inidividual_layer[strength]['pre'],
            )
            print('Strength in first layer', 10*'=')
            for layer in range(8):
                # layer_scores = scores_post[layer]
                layer_scores = scores_pre[layer]
                print(f"Mean for layer {layer}:", layer_scores.mean())
                individual_layer_results[starting_layer, strength_idx, layer] = layer_scores.mean()
            strength_idx += 1
    return individual_layer_results
# end single_layer_latent_influence

def all_layers_latent_influence(
    model,
    device,
    valLoader,
    target_train_states,
    bank_key,
    num_neighbors,
    mask_token_id,
    bar_token_id,
    examined_strengths,
    num_layers=8
):
    strength_values_num = len(examined_strengths)
    strengths = {
        layer: 0.0
        for layer in range(len(model.encoder_layers))
    }
    every_layer_results = np.zeros((strength_values_num, num_layers))
    results_inidividual_layer = {}
    for strength in examined_strengths:
        print('strength value: ', strength)
        for layer in range(0, len(model.encoder_layers)):
            strengths[layer] = strength
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
            valid_positions = harmony_input[ harmony_input != mask_token_id ]

            with torch.no_grad():
                _, states_output = model(
                    melodies.to(device),
                    harmony_tokens=harmony_input.to(device),
                    style_banks=target_train_states,
                    bank_key=bank_key,
                    num_neighbors=num_neighbors,
                    steering_strengths=strengths,
                    steer_mask=None,# valid_positions.to(device),
                    get_layers_output=True,
                )
                steered_states = states_output['layers_output']
                pre_out = states_output['pre_intervention_output']
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

    strength_idx = 0
    for strength in np.round(np.linspace(0, 1, 11), 1):
        # Evaluate the post-intervention states.
        scores_post = latent_steering_progress(
            target_states=target_train_states,
            bank_key=bank_key,
            original_states=results_inidividual_layer[0]['steered'],
            generated_states=results_inidividual_layer[strength]['steered'],
        )

        # Evaluate the pre-intervention states.
        scores_pre = latent_steering_progress(
            target_states=target_train_states,
            bank_key=bank_key,
            original_states=results_inidividual_layer[0]['steered'],
            generated_states=results_inidividual_layer[strength]['pre'],
        )
        print('Strength in first layer', 10*'=')
        for layer in range(8):
            # layer_scores = scores_post[layer]
            layer_scores = scores_pre[layer]
            print(f"Mean for layer {layer}:", layer_scores.mean())
            every_layer_results[strength_idx, layer] = layer_scores.mean()
        strength_idx += 1
    return every_layer_results
# end all_layers_latent_influence

def make_imshow_of_inividual_layer_results(
    individual_layer_results,
    fig_path
):
    concat_matrix = np.concatenate(individual_layer_results, axis=1)

    fig, ax = plt.subplots(figsize=(14, 4))
    image = ax.imshow(concat_matrix, cmap='gray_r', aspect='auto', vmin=0, vmax=1)

    block_width = individual_layer_results.shape[2]
    block_count = individual_layer_results.shape[0]
    for block_idx in range(1, block_count):
        ax.axvline(block_idx * block_width - 0.5, color='black', linewidth=1.5)

    ax.set_yticks(np.arange(concat_matrix.shape[0]))
    ax.set_yticklabels(np.round(np.linspace(0,1,11), 1))
    ax.set_xticks(np.arange(concat_matrix.shape[1]))
    ax.set_xticklabels(np.tile(np.arange(block_width), block_count))
    ax.set_xlabel('Layer within block')
    ax.set_ylabel('Strength')
    fig.colorbar(image, ax=ax)
    plt.savefig(fig_path, dpi=300)
# end make_imshow_of_inividual_layer_results

def make_imshow_of_all_layers_results(
    every_layer_results,
    fig_path
):
    fig, ax = plt.subplots(figsize=(3, 4))
    image = ax.imshow(every_layer_results, cmap='gray_r', aspect='auto', vmin=0, vmax=1)

    ax.set_yticks(np.arange(every_layer_results.shape[0]))
    ax.set_yticklabels(np.round(np.linspace(0,1,11), 1))
    ax.set_xticks(np.arange(every_layer_results.shape[1]))
    ax.set_xticklabels(np.arange(every_layer_results.shape[1]))
    ax.set_xlabel('Layer')
    ax.set_ylabel('Strength')
    plt.colorbar(image, ax=ax)
    plt.savefig(fig_path, dpi=300)
# end make_imshow_of_all_layers_results


def layer_latent_dists(
    model,
    device,
    valLoader,
    target_train_states,
    bank_key,
    num_neighbors,
    mask_token_id,
    bar_token_id,
    strength
):
    strengths = {
        layer: strength
        for layer in range(len(model.encoder_layers))
    }
    results_inidividual_layer = {}
    model.eval()

    dists = {
        layer: 0
        for layer in range(len(model.encoder_layers))
    }

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
        valid_positions = harmony_input[ harmony_input != mask_token_id ]

        with torch.no_grad():
            _, states_output = model(
                melodies.to(device),
                harmony_tokens=harmony_input.to(device),
                style_banks=target_train_states,
                bank_key=bank_key,
                num_neighbors=num_neighbors,
                steering_strengths=strengths,
                steer_mask=None,# valid_positions.to(device),
                get_layers_output=True,
            )
            batch_dists = states_output['intervention_dists_per_layer']
            for layer_idx, dist in batch_dists.items():
                dists[layer_idx] += dist.detach().cpu()
            # end for
        # end with
    # end batch
    return dists
# end single_layer_latent_dists