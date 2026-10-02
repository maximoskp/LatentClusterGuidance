import torch
import torch.nn.functional as F
from torcheval.metrics.text import Perplexity
import random
from tqdm import tqdm
from metrics_utils import compute_normalized_token_entropy
import random
import csv
import numpy as np
import os
from transformers import get_cosine_schedule_with_warmup
import sys

perplexity_metric = Perplexity(ignore_index=-100)

contrastive_weight = 0.1

def single_step_progressive_masking(
        harmony_tokens,
        total_stages,
        mask_token_id,
        stage_in=None,
        bar_token_id=None
    ):
    """
    Generate visible input and denoising target for diffusion-style training.
    Visible tokens are always one fewer than the target ones.

    Args:
        harmony_tokens (torch.Tensor): Tensor of shape (B, L) containing target harmony token ids.
        stage (int): Current training stage (0 to total_stages - 1).
        mask_token_id (int): The token ID used to mask hidden positions in visible_harmony.
        device (str or torch.device): Target device.

    Returns:
        visible_harmony (torch.Tensor): Tensor of shape (B, L) with visible tokens (others masked).
        denoising_target (torch.Tensor): Tensor of shape (B, L) with tokens to predict (others = -100).
        stage_indices (torch.Tensor): Tensor of shape (B, 1) with stage per batch item.
        target_indices (torch.Tensor): Tensor of shape (B, 1) with target index per batch item.
    """
    device = harmony_tokens.device
    B, L = harmony_tokens.shape
    total_stages = L # denominator later, make sure there is at least one left

    visible_harmony = torch.full_like(harmony_tokens, fill_value=mask_token_id)
    denoising_target = torch.full_like(harmony_tokens, fill_value=-100)  # -100 is ignored by CrossEntropyLoss

    if bar_token_id is not None:
        # Create a mask for bar token positions
        bar_mask = (harmony_tokens == bar_token_id)
        # Put bar tokens in visible_harmony (always unmasked)
        visible_harmony[bar_mask] = bar_token_id
        # # Also include them in the denoising target (so model predicts them too)
        # denoising_target[bar_mask] = bar_token_id
    
    stage_indices = torch.randint(0, total_stages, (B,), device=device)
    target_indices = torch.zeros((B,), device=device)
    for b in range(B):
        stage = stage_indices[b] if stage_in is None else stage_in

        percent_visible = stage / total_stages
        perm = torch.randperm(L, device=device)
        num_visible = int(L * percent_visible)
        num_predict = 1

        visible_idx = perm[:num_visible]
        predict_idx = perm[:num_visible + num_predict]  # predict includes visible + next unmasked tokens
        target_indices[b] = predict_idx[-1]

        visible_harmony[b, visible_idx] = harmony_tokens[b, visible_idx]
        denoising_target[b, predict_idx] = harmony_tokens[b, predict_idx]

    return visible_harmony, denoising_target, stage_indices, target_indices
# end single_step_progressive_masking

def random_progressive_masking(
        harmony_tokens,
        total_stages,
        mask_token_id,
        stage_in=None,
        bar_token_id=None
    ):
    """
    Generate visible input and denoising target for diffusion-style training.

    Args:
        harmony_tokens (torch.Tensor): Tensor of shape (B, L) containing target harmony token ids.
        stage (int): Current training stage (0 to total_stages - 1).
        total_stages (int): Total number of diffusion stages.
        mask_token_id (int): The token ID used to mask hidden positions in visible_harmony.
        device (str or torch.device): Target device.

    Returns:
        visible_harmony (torch.Tensor): Tensor of shape (B, L) with visible tokens (others masked).
        denoising_target (torch.Tensor): Tensor of shape (B, L) with tokens to predict (others = -100).
    """
    device = harmony_tokens.device
    B, L = harmony_tokens.shape

    visible_harmony = torch.full_like(harmony_tokens, fill_value=mask_token_id)
    denoising_target = torch.full_like(harmony_tokens, fill_value=-100)  # -100 is ignored by CrossEntropyLoss

    if bar_token_id is not None:
        # Create a mask for bar token positions
        bar_mask = (harmony_tokens == bar_token_id)
        # Put bar tokens in visible_harmony (always unmasked)
        visible_harmony[bar_mask] = bar_token_id
        # # Also include them in the denoising target (so model predicts them too)
        # denoising_target[bar_mask] = bar_token_id
    
    stage_indices = torch.randint(0, total_stages, (B,), device=device)
    for b in range(B):
        stage = stage_indices[b] if stage_in is None else stage_in

        percent_visible = stage / total_stages
        percent_predict = 1 / total_stages
        perm = torch.randperm(L, device=device)
        num_visible = int(L * percent_visible)
        num_predict = int(L * percent_predict + 0.5)

        visible_idx = perm[:num_visible]
        predict_idx = perm[num_visible:num_visible + num_predict]  # predict includes visible + next unmasked tokens
        # print('visible_idx: ', visible_idx)
        # print('predict_idx: ', predict_idx)

        visible_harmony[b, visible_idx] = harmony_tokens[b, visible_idx]
        denoising_target[b, predict_idx] = harmony_tokens[b, predict_idx]

    # visible_harmony = harmony_tokens.clone()
    # visible_harmony[:, :] = mask_token_id
    # # visible_harmony[:, 0:10] = mask_token_id
    # denoising_target = harmony_tokens.clone()  # -100 is ignored by CrossEntropyLoss
    # # denoising_target[:, 10:] = -100

    return visible_harmony, denoising_target, stage_indices
# end random_progressive_masking

def full_to_partial_masking(
        harmony_tokens,
        mask_token_id,
        num_visible=0,
        bar_token_id=None
    ):
    """
    Generate visible input and denoising target for diffusion-style training.

    Args:
        harmony_tokens (torch.Tensor): Tensor of shape (B, L) containing target harmony token ids.
        stage (int): Current training stage (0 to total_stages - 1).
        total_stages (int): Total number of diffusion stages.
        mask_token_id (int): The token ID used to mask hidden positions in visible_harmony.
        device (str or torch.device): Target device.

    Returns:
        visible_harmony (torch.Tensor): Tensor of shape (B, L) with visible tokens (others masked).
        denoising_target (torch.Tensor): Tensor of shape (B, L) with tokens to predict (others = -100).
    """
    device = harmony_tokens.device
    B, L = harmony_tokens.shape

    visible_harmony = torch.full_like(harmony_tokens, fill_value=mask_token_id)
    denoising_target = torch.full_like(harmony_tokens, fill_value=-100)  # -100 is ignored by CrossEntropyLoss

    if bar_token_id is not None:
        # Create a mask for bar token positions
        bar_mask = (harmony_tokens == bar_token_id)
        # Put bar tokens in visible_harmony (always unmasked)
        visible_harmony[bar_mask] = bar_token_id
        # # Also include them in the denoising target (so model predicts them too)
        denoising_target[bar_mask] = bar_token_id
    
    perm = torch.randperm(L, device=device)

    visible_idx = perm[:num_visible]
    predict_idx = perm[num_visible:]  # predict all remaining
    # print('visible_idx: ', visible_idx)
    # print('predict_idx: ', predict_idx)

    visible_harmony[:, visible_idx] = harmony_tokens[:, visible_idx]
    denoising_target[:, predict_idx] = harmony_tokens[:, predict_idx]

    # visible_harmony = harmony_tokens.clone()
    # visible_harmony[:, :] = mask_token_id
    # # visible_harmony[:, 0:10] = mask_token_id
    # denoising_target = harmony_tokens.clone()  # -100 is ignored by CrossEntropyLoss
    # # denoising_target[:, 10:] = -100

    return visible_harmony, denoising_target
# end full_to_partial_masking

def structured_progressive_masking(
        harmony_tokens,
        total_stages,
        mask_token_id,
        stage_in=None
    ):
    B, L = harmony_tokens.shape
    device = harmony_tokens.device
    visible_harmony = torch.full_like( harmony_tokens, mask_token_id )
    denoising_target = harmony_tokens.clone()
    input_unmask = torch.full_like( harmony_tokens, 0, dtype=torch.bool, device=device )
    target_to_learn = torch.full_like(
        harmony_tokens,
        False,
        dtype=torch.bool,
        device=device
    )
    stage_indices = torch.randint(0, total_stages, (B,), device=device)
    for i in range(B):
        stage = stage_indices[i] if stage_in is None else stage_in

        spacing_target = min( L, max(1, int((2**(8-stage)))) )
        # Get the indices that will remain unmasked for this step
        target_to_learn[i, ::spacing_target] = True  # reveal tokens at spacing in target
        spacing_input = 2*spacing_target #max(2, int((2**(8*(1-stage/total_stages)))*(ts_num/ts_den)))
        input_unmask[i, ::spacing_input] = spacing_input <= L  # reveal tokens at spacing in harmony input
    visible_harmony[input_unmask] = harmony_tokens[input_unmask]
    target_to_learn[input_unmask] = False
    denoising_target[~torch.logical_or( target_to_learn , input_unmask )] = -100  # ignore tokens that were not shown to the model
    return visible_harmony, denoising_target, stage_indices
# end structured_progressive_masking

def apply_masking(
        harmony_tokens,
        mask_token_id,
        total_stages=10,
        curriculum_type='random',
        stage=None,
        bar_token_id=None
    ):
    if curriculum_type == 'random':
        return random_progressive_masking(
                harmony_tokens,
                total_stages,
                mask_token_id,
                stage,
                bar_token_id
            )
    elif curriculum_type == 'base2':
        return structured_progressive_masking(
                harmony_tokens,
                total_stages,
                mask_token_id,
                stage
            )
    elif curriculum_type == 'step':
        return single_step_progressive_masking(
                harmony_tokens,
                total_stages,
                mask_token_id,
                stage
            )
# end apply_masking

def guidance_contrastive_loss(real_guide_z, random_guide_z, reduction='mean', margin=0.2):
    """
    Contrastive-style loss for guidance embeddings.

    Encourages the real and random guidance encoder outputs to be as far apart
    as possible in embedding space. A margin-based cosine loss gives the
    model a concrete separation target instead of minimizing similarity
    without an explicit margin.
    """
    real_norm = F.normalize(real_guide_z, dim=-1)
    random_norm = F.normalize(random_guide_z, dim=-1)
    sim = torch.sum(real_norm * random_norm, dim=-1)
    loss = F.relu(sim - margin)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

# end guidance_contrastive_loss

def apply_structured_masking(harmony_tokens,
    mask_token_id,
    stage,
    time_sigs,
    total_stages=10,
    curriculum_type='no'):
    """
    harmony_tokens: (B, time_step) - original ground truth tokens
    mask_token_id: int - ID for the special <mask> token
    stage: int - stage of uncovering masks 0, 1, 2 ...
    time_sigs: batch of 16-item binary encodings of time signature for each item
    curriculum_type: how to progress with masking
    - 'no': all tokens are masked and the model needs to learn to unmask all
    - 'random': an increasing number of tokens is masked and the model needs to learn to unmask all
    - 'ts_blank': an increasing number of ts-based tokens is masked
    - 'ts_incr': an increasing number of ts-based tokens is masked while previous are unmasked

    Returns:
        masked_harmony: (B, time_steps) with some tokens replaced with <mask>
        target_harmony: (B, time_steps) with -100 at positions we do NOT want loss
    """
    B, T = harmony_tokens.shape

    # Create masked version that will serve as the input
    masked_harmony = torch.full_like( harmony_tokens, mask_token_id )

    # Create target. Loss computed only on masked positions that we care about learning
    # In incremental learning, we care to learn only a portion of the masked input tokens,
    # while for other masked tokens we don't care.
    target = harmony_tokens.clone()
    device = harmony_tokens.device
    # assume that not tokens need to be masked for learning
    target_to_learn = torch.full_like(
        harmony_tokens,
        curriculum_type=='random' or curriculum_type=='no',
        dtype=torch.bool,
        device=device
    )
    if curriculum_type == 'ts_incr':
        # some tokens need to be revealed in the input for incremental learning
        input_unmask = torch.full_like( harmony_tokens, 0, dtype=torch.bool, device=device )

    if curriculum_type == 'ts_incr' or curriculum_type == 'ts_blank':
        for i in range(B):
            # get ts num and den
            ts_num = torch.nonzero(time_sigs[i, :14])[0] + 2
            ts_den = torch.nonzero(time_sigs[i, 14:])[0]*8 - 4
            spacing_target = min(128, max(1, int((2**(7*stage/total_stages))*(ts_num/ts_den))))

            # Get the indices that will remain unmasked for this step
            target_to_learn[i, ::spacing_target] = True  # reveal tokens at spacing in target
            if curriculum_type == 'ts_incr':
                spacing_input = max(2, int((2**(8*stage/total_stages))*(ts_num/ts_den)))
                input_unmask[i, ::spacing_input] = True  # reveal tokens at spacing in harmony input
    if curriculum_type == 'random':
        for i in range(B):
            stage_ratio = 1. - (stage+1)/total_stages
            valid_indices = (harmony_tokens[i] != -1).nonzero(as_tuple=False).squeeze()
            n_reveal = int(len(valid_indices) * stage_ratio)
            reveal_indices = random.sample(valid_indices.tolist(), n_reveal)
            masked_harmony[i, reveal_indices] = harmony_tokens[i, reveal_indices]
            target_to_learn = masked_harmony == mask_token_id
    if curriculum_type == 'ts_incr':
        masked_harmony[input_unmask] = harmony_tokens[input_unmask]
        target_to_learn[input_unmask] = False
        target[~torch.logical_or( target_to_learn , input_unmask )] = -100  # ignore tokens that were not shown to the model
    if curriculum_type == 'ts_blank':
        target[~target_to_learn] = -100  # ignore tokens that were shown to the model
    return masked_harmony, target
# end apply_structured_masking

def get_stage_linear(epoch, epochs_per_stage, max_stage):
    return min(epoch // epochs_per_stage, max_stage)
# end get_stage_linear

def get_stage_mixed(epoch, max_epoch, max_stage):
    """Returns a random step index, biased toward early stages in early epochs.
    Last epoch is only with maximum stage."""
    if epoch >= max_epoch - 1:
        return max_stage
    progress = epoch / max_epoch
    probs = torch.softmax(torch.tensor([
        (1.0 - abs(progress - (i / max_stage))) * 5 for i in range(max_stage + 1)
    ]), dim=0)
    return torch.multinomial(probs, 1).item()
# end get_stage_mixed

def get_stage_uniform(epoch, max_epoch, max_stage):
    """Returns a random step index, uniform across all stages."""
    return np.random.randint(max_stage + 1)
# end get_stage_uniform

def apply_focal_sharpness(
        melody_grid,
        target_indices,
        focal_sharpness,
        min_sigma=1.0,
        max_sigma=40.0
    ):
    """
    Apply Gaussian attenuation around a focal index per sequence.
    
    Args:
        melody_grid: [B, L, dm] tensor (batch of pianorolls)
        target_indices: [B, 1] tensor with focal indices in [0, L-1]
        focal_sharpness: [B, 1] tensor with values in [0,1]
        min_sigma, max_sigma: range for Gaussian variance mapping
    
    Returns:
        attenuated_grid: [B, L, dm] tensor
    """
    B, L, dm = melody_grid.shape

    # Map sharpness [0,1] -> sigma [max_sigma, min_sigma]
    # low sharpness = wide Gaussian (large sigma), high sharpness = narrow Gaussian (small sigma)
    sigma = max_sigma - focal_sharpness * (max_sigma - min_sigma)  # [B,1]

    # Create positions [0..L-1]
    positions = torch.arange(L, device=melody_grid.device).unsqueeze(0).expand(B, L)  # [B, L]

    # Expand focal indices
    # make sure that target_indices is the correct shape
    if target_indices.dim() == 1:          # shape [B]
        target_indices = target_indices.unsqueeze(-1)  # [B,1]
    elif target_indices.dim() == 2 and target_indices.shape[0] == 1:  
        # case: [1,B] -> transpose to [B,1]
        target_indices = target_indices.t()
    focal = target_indices.expand(-1, L)  # [B, L]

    # Compute squared distance
    dist2 = (positions - focal).float() ** 2  # [B, L]

    # Compute Gaussian weights: exp(-d^2 / (2σ^2))
    weights = torch.exp(-dist2 / (2 * sigma**2))  # [B, L]

    # Normalize weights (optional, keeps focal=1 and remote->0 without collapsing scale)
    weights = weights / weights.max(dim=1, keepdim=True).values

    # Expand to match pianoroll dims
    weights = weights.unsqueeze(-1)  # [B, L, 1]

    # Apply attenuation
    attenuated_grid = melody_grid * weights

    return attenuated_grid
# end apply_focal_sharpness

def validation_curriculum_loop(curriculum_type, model, valloader, mask_token_id, bar_token_id, \
                    num_visible, condition_dim, total_stages, loss_fn, epoch, step, \
                    train_loss, train_accuracy, \
                    train_perplexity, train_token_entropy,
                    best_val_loss, saving_version, results_path=None, \
                    transformer_path=None, tqdm_position=0, save_every_epoch=True):
    device = model.device
    model.eval()
    with torch.no_grad():
        val_loss = 0
        running_loss = 0
        batch_num = 0
        running_accuracy = 0
        val_accuracy = 0
        running_perplexity = 0
        val_perplexity = 0
        running_token_entropy = 0
        val_token_entropy = 0
        print('validation')
        with tqdm(valloader, unit='batch', position=tqdm_position) as tepoch:
            tepoch.set_description(f'Epoch {epoch}@{step}| val')
            for batch in tepoch:
                perplexity_metric.reset()
                melody_grid = batch["pianoroll"].to(device)           # (B, 256, 140)
                harmony_gt = batch["harmony_ids"].to(device)         # (B, 256)
                if condition_dim is not None:
                    conditioning_vec = batch["time_signature"].to(device)  # (B, condDim)
                else:
                    conditioning_vec = None
                
                # Apply masking to harmony
                if curriculum_type == 'f2f':
                    harmony_input, harmony_target = full_to_partial_masking(
                        harmony_gt,
                        mask_token_id,
                        num_visible,
                        bar_token_id=bar_token_id
                    )
                    stage_indices = None
                else:
                    # Apply masking to harmony
                    harmony_input, harmony_target, stage_indices = apply_masking(
                        harmony_gt,
                        mask_token_id,
                        total_stages=total_stages,
                        curriculum_type=curriculum_type
                    )
                    num_visible = -1
                
                # Forward pass
                logits = model(
                    melody_grid.to(device),
                    harmony_input.to(device),
                    None,
                    None
                )

                # Compute loss only on masked tokens
                loss = loss_fn(logits.view(-1, logits.size(-1)), harmony_target.view(-1))

                # update loss and accuracy
                batch_num += 1
                running_loss += loss.item()
                val_loss = running_loss/batch_num
                # accuracy
                predictions = logits.argmax(dim=-1)
                # mask = harmony_target != harmony_input # harmony_target != -100
                # mask = torch.logical_and(harmony_target != harmony_input, harmony_target != -100)
                mask = harmony_target != -100
                running_accuracy += (predictions[mask] == harmony_target[mask]).sum().item()/mask.sum().item()
                val_accuracy = running_accuracy/batch_num
                # perplexity
                running_perplexity += perplexity_metric.update(logits, harmony_target).compute().item()
                val_perplexity = running_perplexity/batch_num
                # token entropy
                _, entropy_per_batch = compute_normalized_token_entropy(logits, harmony_target, pad_token_id=-100)
                running_token_entropy += entropy_per_batch
                val_token_entropy = running_token_entropy/batch_num

                tepoch.set_postfix(loss=val_loss, accuracy=val_accuracy)
            # end for batch
    # end with tqdm
    if transformer_path is not None:
        if  save_every_epoch or (best_val_loss > val_loss):
            print('saving!')
            saving_version += 1
            best_val_loss = val_loss
            torch.save(model.state_dict(), transformer_path)
        if  epoch in [168, 177, 182, 189, 196, 200, 203]:
            print(f'saving copy of epoch {epoch} with num_visible {num_visible}')
            torch.save(model.state_dict(), transformer_path.replace('.pt', f'_epoch{epoch}_nvis{num_visible}.pt'))
    print(f'validation: accuracy={val_accuracy}, loss={val_loss}')
    print('results_path: ', results_path)
    if results_path is not None:
        with open( results_path, 'a' ) as f:
            writer = csv.writer(f)
            writer.writerow( [epoch, step, num_visible, train_loss, train_accuracy, \
                            train_perplexity, train_token_entropy, \
                            val_loss, val_accuracy, \
                            val_perplexity, val_token_entropy, \
                            saving_version] )
    return best_val_loss, saving_version
# end validation_loop

def train_with_curriculum(
    model, optimizer, trainloader, valloader, loss_fn, mask_token_id,
    curriculum_type='random',
    epochs=100,
    condition_dim=None,
    exponent=5,
    total_stages=None,
    results_path=None,
    transformer_path=None,
    bar_token_id=None,
    validations_per_epoch=1,
    tqdm_position=0,
    save_every_epoch=True
):
    # device = next(model.parameters()).device
    device = model.device
    perplexity_metric.to(device)
    best_val_loss = np.inf
    saving_version = 0

    # save results and model
    print('results_path:', results_path)
    if results_path is not None:
        result_fields = ['epoch', 'step', 'n_vis', 'train_loss', 'train_acc', \
                        'train_ppl', 'train_te', 'val_loss', \
                        'val_acc', 'val_ppl', 'val_te', 'sav_version']
        with open( results_path, 'w' ) as f:
            writer = csv.writer(f)
            writer.writerow( result_fields )

    # Compute total training steps
    total_steps = len(trainloader) * epochs
    # Define the scheduler
    # warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    step = 0

    for epoch in range(epochs):
        train_loss = 0
        running_loss = 0
        batch_num = 0
        running_accuracy = 0
        train_accuracy = 0
        running_perplexity = 0
        train_perplexity = 0
        running_token_entropy = 0
        train_token_entropy = 0
        
        with tqdm(trainloader, unit='batch', position=tqdm_position) as tepoch:
            tepoch.set_description(f'Epoch {epoch}@{step} | trn')
            for batch in tepoch:
                perplexity_metric.reset()
                model.train()
                melody_grid = batch["pianoroll"].to(device)    # (B, L, prDim)
                harmony_gt = batch["harmony_ids"].to(device)     # (B, L)
                if condition_dim is not None:
                    conditioning_vec = batch["time_signature"].to(device)  # (B, condDim)
                else:
                    conditioning_vec = None

                # Apply masking to harmony
                if curriculum_type == 'f2f':
                    if exponent == -1:
                        percent_visible = 0.0
                    else:
                        # percent_visible = min(1.0, (step+1)/total_steps)**exponent  # 5th power goes around half way near zero
                        percent_visible = min(1.0, np.exp( (-1/2)*np.power( (step-(total_steps/2))/(total_steps/15) , 2) ))
                    L = harmony_gt.shape[1]
                    num_visible = min( int(L * percent_visible), L-1 )  # ensure at least one token is predicted
                    harmony_input, harmony_target = full_to_partial_masking(
                        harmony_gt,
                        mask_token_id,
                        num_visible,
                        bar_token_id=bar_token_id
                    )
                    stage_indices = None
                else:
                    # Apply masking to harmony
                    harmony_input, harmony_target, stage_indices = apply_masking(
                        harmony_gt,
                        mask_token_id,
                        total_stages=total_stages,
                        curriculum_type=curriculum_type
                    )
                    num_visible = -1
                # Forward pass
                logits = model(
                    melody_grid.to(device),
                    harmony_input.to(device),
                    None,
                    None
                )

                # Compute loss only on masked tokens
                loss = loss_fn(logits.view(-1, logits.size(-1)), harmony_target.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()

                # update loss and accuracy
                batch_num += 1
                running_loss += loss.item()
                train_loss = running_loss/batch_num
                # accuracy
                predictions = logits.argmax(dim=-1)
                # mask = torch.logical_and(harmony_target != harmony_input, harmony_target != -100)
                mask = harmony_target != -100
                running_accuracy += (predictions[mask] == harmony_target[mask]).sum().item()/max(1,mask.sum().item())
                train_accuracy = running_accuracy/batch_num
                # perplexity
                running_perplexity += perplexity_metric.update(logits, harmony_target).compute().item()
                train_perplexity = running_perplexity/batch_num
                # token entropy
                _, entropy_per_batch = compute_normalized_token_entropy(logits, harmony_target, pad_token_id=-100)
                running_token_entropy += entropy_per_batch
                train_token_entropy = running_token_entropy/batch_num

                tepoch.set_postfix(loss=train_loss, accuracy=train_accuracy)
                step += 1
                if step%(total_steps//(epochs*validations_per_epoch)) == 0 or step == total_steps:
                    best_val_loss, saving_version = validation_curriculum_loop(
                        curriculum_type,
                        model,
                        valloader,
                        mask_token_id,
                        bar_token_id,
                        num_visible,
                        condition_dim,
                        total_stages,
                        loss_fn,
                        epoch,
                        step,
                        train_loss,
                        train_accuracy,
                        train_perplexity,
                        train_token_entropy,
                        best_val_loss,
                        saving_version,
                        results_path=results_path,
                        transformer_path=transformer_path,
                        tqdm_position=tqdm_position,
                        save_every_epoch=save_every_epoch
                    )
            # end for batch
        # end with tqdm
    # end for epoch
# end train_with_curriculum

# ================================================================
# Graph training
# ================================================================

def validation_graph_loop(
        transformer_model, graph_model,
        valloader,
        mask_token_id, bar_token_id,
        logits_loss_fn,
        epoch,
        step,
        total_train_loss,
        real_train_loss,
        recomposed_train_loss,
        contrastive_train_loss,
        real_train_accuracy,
        recomposed_train_accuracy,
        real_train_top5,
        recomposed_train_top5,
        best_val_loss, saving_version,
        results_path=None, transformer_path=None, 
        graph_model_path=None, tqdm_position=0
    ):
    device = transformer_model.device
    transformer_model.eval()
    graph_model.eval()
    with torch.no_grad():
        running_total_loss = 0
        total_val_loss = 0
        running_real_loss = 0
        real_val_loss = 0
        running_recomposed_loss = 0
        recomposed_val_loss = 0
        running_contra_loss = 0
        val_contra_loss = 0

        real_val_accuracy = 0
        running_real_accuracy = 0
        recomposed_val_accuracy = 0
        running_recomposed_accuracy = 0

        real_val_top5 = 0
        running_real_top5 = 0
        recomposed_val_top5 = 0
        running_recomposed_top5 = 0

        batch_num = 0
        print('validation')
        with tqdm(valloader, unit='batch', position=tqdm_position) as tepoch:
            tepoch.set_description(f'Epoch {epoch}@{step}| val')
            for batch in tepoch:
                real_guide_z = graph_model(batch['real_graph'].to(device))
                real_constraints = batch['real_harmony_ids'].clone()
                real_constraints[batch['mask_token_positions']] = mask_token_id

                recomposed_guide_z = graph_model(batch['recomposed_graph'].to(device))
                recomposed_constraints = batch['recomposed_harmony_ids'].clone()
                recomposed_constraints[batch['mask_token_positions']] = mask_token_id

                real_logits = transformer_model(
                    batch['pianoroll'].to(device),
                    real_constraints.to(device),
                    real_guide_z.to(device)
                )

                recomposed_logits = transformer_model(
                    batch['pianoroll'].to(device),
                    recomposed_constraints.to(device),
                    recomposed_guide_z.to(device)
                )

                mask = batch['mask_token_positions'].to(device)
                real_target = batch['real_harmony_ids'].clone().to(device)
                # real_target[~mask] = -100
                recomposed_target = batch['recomposed_harmony_ids'].clone().to(device)
                # recomposed_target[~mask] = -100

                real_logits_loss = logits_loss_fn(
                    real_logits.view(-1, real_logits.size(-1)),
                    real_target.view(-1)
                )

                recomposed_logits_loss = logits_loss_fn(
                    recomposed_logits.view(-1, recomposed_logits.size(-1)),
                    recomposed_target.view(-1)
                )

                contrastive_val_loss = guidance_contrastive_loss(real_guide_z, recomposed_guide_z)

                loss = real_logits_loss + recomposed_logits_loss + contrastive_weight * contrastive_val_loss
                # loss = real_logits_loss + recomposed_logits_loss
                # loss = recomposed_logits_loss

                # update loss and accuracy
                # loss
                batch_num += 1
                running_total_loss += loss.item()
                total_val_loss = running_total_loss/batch_num
                running_real_loss += real_logits_loss.item()
                real_val_loss = running_real_loss/batch_num
                running_recomposed_loss += recomposed_logits_loss.item()
                recomposed_val_loss = running_recomposed_loss/batch_num
                running_contra_loss += contrastive_val_loss.item()
                val_contra_loss = running_contra_loss/batch_num

                # accuracy
                real_predictions = real_logits.argmax(dim=-1)
                recomposed_predictions = recomposed_logits.argmax(dim=-1)
                mask_acc = batch['real_harmony_ids'] != -100
                running_real_accuracy += (real_predictions[mask_acc] == batch['real_harmony_ids'][mask_acc].to(device)).sum().item()/max(1,mask_acc.sum().item())
                real_val_accuracy = running_real_accuracy/batch_num
                running_recomposed_accuracy += (recomposed_predictions[mask_acc] == batch['recomposed_harmony_ids'][mask_acc].to(device)).sum().item()/max(1,mask_acc.sum().item())
                recomposed_val_accuracy = running_recomposed_accuracy/batch_num

                # top5
                k = 5

                topk_real = torch.topk(real_logits, k, dim=-1).indices
                real_target = batch['real_harmony_ids'].to(device).unsqueeze(-1)
                correct_topk_real = (topk_real == real_target).any(dim=-1)

                topk_recomposed = torch.topk(recomposed_logits, k, dim=-1).indices
                recomposed_target = batch['recomposed_harmony_ids'].to(device).unsqueeze(-1)
                correct_topk_recomposed = (topk_recomposed == recomposed_target).any(dim=-1)

                mask = batch['mask_token_positions'].to(device)
                running_real_top5 += correct_topk_real[mask].sum().item() / max(1, mask.sum().item())
                real_val_top5 = running_real_top5/batch_num

                running_recomposed_top5 += correct_topk_recomposed[mask].sum().item() / max(1, mask.sum().item())
                recomposed_val_top5 = running_recomposed_top5/batch_num

                tepoch.set_postfix(
                    loss=total_val_loss,
                    acc_real=real_val_accuracy,
                    acc_rcn=recomposed_val_accuracy
                )
            # end for batch
        # end with tqdm
    # end with no grad
    if transformer_path is not None:
        if  best_val_loss > total_val_loss:
            print('saving!')
            saving_version += 1
            best_val_loss = total_val_loss
            torch.save(transformer_model.state_dict(), transformer_path)
            torch.save(graph_model.state_dict(), graph_model_path)
    print(f'validation: loss={total_val_loss}, acc_real={real_val_accuracy}, acc_rnd={recomposed_val_accuracy}')
    print('results_path: ', results_path)
    if results_path is not None:
        with open( results_path, 'a' ) as f:
            writer = csv.writer(f)
            writer.writerow( [epoch, step, \
                        total_train_loss, real_train_loss,  recomposed_train_loss, contrastive_train_loss, \
                        real_train_accuracy, recomposed_train_accuracy, real_train_top5, recomposed_train_top5, \
                        total_val_loss, real_val_loss,  recomposed_val_loss, val_contra_loss, \
                        real_val_accuracy, recomposed_val_accuracy, real_val_top5, recomposed_val_top5, \
                        saving_version] )
    return best_val_loss, saving_version
# end validation_graph_loop

def train_graph_loop(
        transformer_model, graph_model,
        logits_loss_fn,
        optimizer, trainloader, valloader, mask_token_id,
        use_contra=False,
        epochs=100,
        results_path=None,
        transformer_path=None,
        graph_model_path=None,
        bar_token_id=None,
        validations_per_epoch=1,
        tqdm_position=0,
        freeze_base=True
    ):
    device = transformer_model.device
    print('device: ', device)
    graph_model.to(device)
    best_val_loss = np.inf
    saving_version = 0

    # save results and model
    print('results_path:', results_path)
    if results_path is not None:
        result_fields = ['epoch', 'step', \
                        'total_train_loss', 'real_train_loss',  'rec_train_loss', 'contra_train_loss', \
                        'real_train_acc', 'rec_train_acc', \
                        'real_train_top5', 'rec_train_top5', \
                        'total_val_loss', 'real_val_loss',  'rec_val_loss', 'contra_val_loss', \
                        'real_val_acc', 'rec_val_acc', \
                        'real_val_top5', 'rec_val_top5', \
                        'sav_version']
        with open( results_path, 'w' ) as f:
            writer = csv.writer(f)
            writer.writerow( result_fields )

    # Compute total training steps
    total_steps = len(trainloader) * epochs
    # Define the scheduler
    # warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    step = 0

    for epoch in range(epochs):
        running_total_loss = 0
        total_train_loss = 0
        running_real_loss = 0
        real_train_loss = 0
        running_recomposed_loss = 0
        recomposed_train_loss = 0
        running_contra_loss = 0
        contra_train_loss = 0

        real_accuracy = 0
        running_real_accuracy = 0
        recomposed_accuracy = 0
        running_recomposed_accuracy = 0

        real_top5 = 0
        running_real_top5 = 0
        recomposed_top5 = 0
        running_recomposed_top5 = 0

        batch_num = 0
        
        with tqdm(trainloader, unit='batch', position=tqdm_position) as tepoch:
            tepoch.set_description(f'Epoch {epoch} | trn')
            for batch in tepoch:
                transformer_model.train()
                graph_model.train()
                if freeze_base:
                    transformer_model.freeze_base()
                
                real_guide_z = graph_model(batch['real_graph'].to(device))
                real_constraints = batch['real_harmony_ids'].clone()
                real_constraints[batch['mask_token_positions']] = mask_token_id

                recomposed_guide_z = graph_model(batch['recomposed_graph'].to(device))
                recomposed_constraints = batch['recomposed_harmony_ids'].clone()
                recomposed_constraints[batch['mask_token_positions']] = mask_token_id

                real_logits = transformer_model(
                    batch['pianoroll'].to(device),
                    real_constraints.to(device),
                    real_guide_z.to(device)
                )

                recomposed_logits = transformer_model(
                    batch['pianoroll'].to(device),
                    recomposed_constraints.to(device),
                    recomposed_guide_z.to(device)
                )

                mask = batch['mask_token_positions'].to(device)
                real_target = batch['real_harmony_ids'].clone().to(device)
                # real_target[~mask] = -100
                recomposed_target = batch['recomposed_harmony_ids'].clone().to(device)
                # recomposed_target[~mask] = -100

                real_logits_loss = logits_loss_fn(
                    real_logits.view(-1, real_logits.size(-1)),
                    real_target.view(-1)
                )
                
                recomposed_logits_loss = logits_loss_fn(
                    recomposed_logits.view(-1, recomposed_logits.size(-1)),
                    recomposed_target.view(-1)
                )

                contrastive_loss = guidance_contrastive_loss(real_guide_z, recomposed_guide_z)

                optimizer.zero_grad()
                loss = real_logits_loss + recomposed_logits_loss + use_contra * contrastive_weight * contrastive_loss
                # loss = real_logits_loss + recomposed_logits_loss
                # loss = recomposed_logits_loss
                loss.backward()
                optimizer.step()
                # scheduler.step()

                # update loss and accuracy
                # loss
                batch_num += 1
                running_total_loss += loss.item()
                total_train_loss = running_total_loss/batch_num
                running_real_loss += real_logits_loss.item()
                real_train_loss = running_real_loss/batch_num
                running_recomposed_loss += recomposed_logits_loss.item()
                recomposed_train_loss = running_recomposed_loss/batch_num
                running_contra_loss += contrastive_loss.item()
                contra_train_loss = running_contra_loss/batch_num

                # accuracy
                real_predictions = real_logits.argmax(dim=-1)
                recomposed_predictions = recomposed_logits.argmax(dim=-1)
                mask_acc = batch['real_harmony_ids'] != -100
                running_real_accuracy += (real_predictions[mask_acc] == batch['real_harmony_ids'][mask_acc].to(device)).sum().item()/max(1,mask_acc.sum().item())
                real_accuracy = running_real_accuracy/batch_num
                running_recomposed_accuracy += (recomposed_predictions[mask_acc] == batch['recomposed_harmony_ids'][mask_acc].to(device)).sum().item()/max(1,mask_acc.sum().item())
                recomposed_accuracy = running_recomposed_accuracy/batch_num

                # top5
                k = 5

                topk_real = torch.topk(real_logits, k, dim=-1).indices
                real_target = batch['real_harmony_ids'].to(device).unsqueeze(-1)
                correct_topk_real = (topk_real == real_target).any(dim=-1)

                topk_recomposed = torch.topk(recomposed_logits, k, dim=-1).indices
                recomposed_target = batch['recomposed_harmony_ids'].to(device).unsqueeze(-1)
                correct_topk_recomposed = (topk_recomposed == recomposed_target).any(dim=-1)

                mask = batch['mask_token_positions'].to(device)
                running_real_top5 += correct_topk_real[mask].sum().item() / max(1, mask.sum().item())
                real_top5 = running_real_top5/batch_num

                running_recomposed_top5 += correct_topk_recomposed[mask].sum().item() / max(1, mask.sum().item())
                recomposed_top5 = running_recomposed_top5/batch_num

                tepoch.set_postfix(
                    loss=total_train_loss,
                    acc_real=real_accuracy,
                    acc_rcn=recomposed_accuracy
                )
                step += 1
                if step%(total_steps//(epochs*validations_per_epoch)) == 0 or step == total_steps:
                    best_val_loss, saving_version = validation_graph_loop(
                        transformer_model, graph_model,
                        valloader,
                        mask_token_id, bar_token_id,
                        logits_loss_fn,
                        epoch,
                        step,
                        total_train_loss,
                        real_train_loss,
                        recomposed_train_loss,
                        contra_train_loss,
                        real_accuracy,
                        recomposed_accuracy,
                        real_top5,
                        recomposed_top5,
                        best_val_loss, saving_version,
                        results_path=results_path,
                        transformer_path=transformer_path,
                        graph_model_path=graph_model_path,
                        tqdm_position=tqdm_position
                    )
            # end for batch
        # end with tqdm
    # end for epoch
# end train_graph_loop

# ================================================================
# BiLSTM training
# ================================================================

def validation_bilstm_loop(
        transformer_model, bilstm_model,
        valloader,
        mask_token_id, bar_token_id,
        logits_loss_fn,
        epoch,
        step,
        total_train_loss,
        real_train_loss,
        recomposed_train_loss,
        contrastive_train_loss,
        real_train_accuracy,
        recomposed_train_accuracy,
        real_train_top5,
        recomposed_train_top5,
        best_val_loss, saving_version,
        results_path=None, transformer_path=None, 
        bilstm_model_path=None, tqdm_position=0
    ):
    device = transformer_model.device
    transformer_model.eval()
    bilstm_model.eval()
    with torch.no_grad():
        running_total_loss = 0
        total_val_loss = 0
        running_real_loss = 0
        real_val_loss = 0
        running_recomposed_loss = 0
        recomposed_val_loss = 0
        running_contra_loss = 0
        val_contra_loss = 0

        real_val_accuracy = 0
        running_real_accuracy = 0
        recomposed_val_accuracy = 0
        running_recomposed_accuracy = 0

        real_val_top5 = 0
        running_real_top5 = 0
        recomposed_val_top5 = 0
        running_recomposed_top5 = 0

        batch_num = 0
        print('validation')
        with tqdm(valloader, unit='batch', position=tqdm_position) as tepoch:
            tepoch.set_description(f'Epoch {epoch}@{step}| val')
            for batch in tepoch:
                real_guide_z = bilstm_model(batch['real_bilstm'].to(device), batch['real_lengths'].to(device))
                real_constraints = batch['real_harmony_ids'].clone()
                real_constraints[batch['mask_token_positions']] = mask_token_id

                recomposed_guide_z = bilstm_model(batch['recomposed_bilstm'].to(device), batch['recomposed_lengths'].to(device))
                recomposed_constraints = batch['recomposed_harmony_ids'].clone()
                recomposed_constraints[batch['mask_token_positions']] = mask_token_id

                real_logits = transformer_model(
                    batch['pianoroll'].to(device),
                    real_constraints.to(device),
                    real_guide_z.to(device)
                )

                recomposed_logits = transformer_model(
                    batch['pianoroll'].to(device),
                    recomposed_constraints.to(device),
                    recomposed_guide_z.to(device)
                )

                mask = batch['mask_token_positions'].to(device)
                real_target = batch['real_harmony_ids'].clone().to(device)
                # real_target[~mask] = -100
                recomposed_target = batch['recomposed_harmony_ids'].clone().to(device)
                # recomposed_target[~mask] = -100

                real_logits_loss = logits_loss_fn(
                    real_logits.view(-1, real_logits.size(-1)),
                    real_target.view(-1)
                )

                recomposed_logits_loss = logits_loss_fn(
                    recomposed_logits.view(-1, recomposed_logits.size(-1)),
                    recomposed_target.view(-1)
                )

                contrastive_val_loss = guidance_contrastive_loss(real_guide_z, recomposed_guide_z)

                loss = real_logits_loss + recomposed_logits_loss + contrastive_weight * contrastive_val_loss
                # loss = real_logits_loss + recomposed_logits_loss
                # loss = recomposed_logits_loss

                # update loss and accuracy
                # loss
                batch_num += 1
                running_total_loss += loss.item()
                total_val_loss = running_total_loss/batch_num
                running_real_loss += real_logits_loss.item()
                real_val_loss = running_real_loss/batch_num
                running_recomposed_loss += recomposed_logits_loss.item()
                recomposed_val_loss = running_recomposed_loss/batch_num
                running_contra_loss += contrastive_val_loss.item()
                val_contra_loss = running_contra_loss/batch_num

                # accuracy
                real_predictions = real_logits.argmax(dim=-1)
                recomposed_predictions = recomposed_logits.argmax(dim=-1)
                mask_acc = batch['real_harmony_ids'] != -100
                running_real_accuracy += (real_predictions[mask_acc] == batch['real_harmony_ids'][mask_acc].to(device)).sum().item()/max(1,mask_acc.sum().item())
                real_val_accuracy = running_real_accuracy/batch_num
                running_recomposed_accuracy += (recomposed_predictions[mask_acc] == batch['recomposed_harmony_ids'][mask_acc].to(device)).sum().item()/max(1,mask_acc.sum().item())
                recomposed_val_accuracy = running_recomposed_accuracy/batch_num

                # top5
                k = 5

                topk_real = torch.topk(real_logits, k, dim=-1).indices
                real_target = batch['real_harmony_ids'].to(device).unsqueeze(-1)
                correct_topk_real = (topk_real == real_target).any(dim=-1)

                topk_recomposed = torch.topk(recomposed_logits, k, dim=-1).indices
                recomposed_target = batch['recomposed_harmony_ids'].to(device).unsqueeze(-1)
                correct_topk_recomposed = (topk_recomposed == recomposed_target).any(dim=-1)

                mask = batch['mask_token_positions'].to(device)
                running_real_top5 += correct_topk_real[mask].sum().item() / max(1, mask.sum().item())
                real_val_top5 = running_real_top5/batch_num

                running_recomposed_top5 += correct_topk_recomposed[mask].sum().item() / max(1, mask.sum().item())
                recomposed_val_top5 = running_recomposed_top5/batch_num

                tepoch.set_postfix(
                    loss=total_val_loss,
                    acc_real=real_val_accuracy,
                    acc_rcn=recomposed_val_accuracy
                )
            # end for batch
        # end with tqdm
    # end with no grad
    if transformer_path is not None:
        if  best_val_loss > total_val_loss:
            print('saving!')
            saving_version += 1
            best_val_loss = total_val_loss
            torch.save(transformer_model.state_dict(), transformer_path)
            torch.save(bilstm_model.state_dict(), bilstm_model_path)
    print(f'validation: loss={total_val_loss}, acc_real={real_val_accuracy}, acc_rnd={recomposed_val_accuracy}')
    print('results_path: ', results_path)
    if results_path is not None:
        with open( results_path, 'a' ) as f:
            writer = csv.writer(f)
            writer.writerow( [epoch, step, \
                        total_train_loss, real_train_loss,  recomposed_train_loss, contrastive_train_loss, \
                        real_train_accuracy, recomposed_train_accuracy, real_train_top5, recomposed_train_top5, \
                        total_val_loss, real_val_loss,  recomposed_val_loss, val_contra_loss, \
                        real_val_accuracy, recomposed_val_accuracy, real_val_top5, recomposed_val_top5, \
                        saving_version] )
    return best_val_loss, saving_version
# end validation_bilstm_loop

def train_bilstm_loop(
        transformer_model, bilstm_model,
        logits_loss_fn,
        optimizer, trainloader, valloader, mask_token_id,
        use_contra=False,
        epochs=100,
        results_path=None,
        transformer_path=None,
        bilstm_model_path=None,
        bar_token_id=None,
        validations_per_epoch=1,
        tqdm_position=0,
        freeze_base=True
    ):
    device = transformer_model.device
    print('device: ', device)
    bilstm_model.to(device)
    best_val_loss = np.inf
    saving_version = 0

    # save results and model
    print('results_path:', results_path)
    if results_path is not None:
        result_fields = ['epoch', 'step', \
                        'total_train_loss', 'real_train_loss',  'rec_train_loss', 'contra_train_loss', \
                        'real_train_acc', 'rec_train_acc', \
                        'real_train_top5', 'rec_train_top5', \
                        'total_val_loss', 'real_val_loss',  'rec_val_loss', 'contra_val_loss', \
                        'real_val_acc', 'rec_val_acc', \
                        'real_val_top5', 'rec_val_top5', \
                        'sav_version']
        with open( results_path, 'w' ) as f:
            writer = csv.writer(f)
            writer.writerow( result_fields )

    # Compute total training steps
    total_steps = len(trainloader) * epochs
    # Define the scheduler
    # warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    step = 0

    for epoch in range(epochs):
        running_total_loss = 0
        total_train_loss = 0
        running_real_loss = 0
        real_train_loss = 0
        running_recomposed_loss = 0
        recomposed_train_loss = 0
        running_contra_loss = 0
        contra_train_loss = 0

        real_accuracy = 0
        running_real_accuracy = 0
        recomposed_accuracy = 0
        running_recomposed_accuracy = 0

        real_top5 = 0
        running_real_top5 = 0
        recomposed_top5 = 0
        running_recomposed_top5 = 0

        batch_num = 0
        
        with tqdm(trainloader, unit='batch', position=tqdm_position) as tepoch:
            tepoch.set_description(f'Epoch {epoch} | trn')
            for batch in tepoch:
                transformer_model.train()
                bilstm_model.train()
                if freeze_base:
                    transformer_model.freeze_base()
                
                real_guide_z = bilstm_model(batch['real_bilstm'].to(device), batch['real_lengths'].to(device))
                real_constraints = batch['real_harmony_ids'].clone()
                real_constraints[batch['mask_token_positions']] = mask_token_id


                recomposed_guide_z = bilstm_model(batch['recomposed_bilstm'].to(device), batch['recomposed_lengths'].to(device))
                recomposed_constraints = batch['recomposed_harmony_ids'].clone()
                recomposed_constraints[batch['mask_token_positions']] = mask_token_id

                real_logits = transformer_model(
                    batch['pianoroll'].to(device),
                    real_constraints.to(device),
                    real_guide_z.to(device)
                )

                recomposed_logits = transformer_model(
                    batch['pianoroll'].to(device),
                    recomposed_constraints.to(device),
                    recomposed_guide_z.to(device)
                )

                mask = batch['mask_token_positions'].to(device)
                real_target = batch['real_harmony_ids'].clone().to(device)
                # real_target[~mask] = -100
                recomposed_target = batch['recomposed_harmony_ids'].clone().to(device)
                # recomposed_target[~mask] = -100

                real_logits_loss = logits_loss_fn(
                    real_logits.view(-1, real_logits.size(-1)),
                    real_target.view(-1)
                )
                
                recomposed_logits_loss = logits_loss_fn(
                    recomposed_logits.view(-1, recomposed_logits.size(-1)),
                    recomposed_target.view(-1)
                )

                contrastive_loss = guidance_contrastive_loss(real_guide_z, recomposed_guide_z)

                optimizer.zero_grad()
                loss = real_logits_loss + recomposed_logits_loss + use_contra * contrastive_weight * contrastive_loss
                # loss = real_logits_loss + recomposed_logits_loss
                # loss = recomposed_logits_loss
                loss.backward()
                optimizer.step()
                # scheduler.step()

                # update loss and accuracy
                # loss
                batch_num += 1
                running_total_loss += loss.item()
                total_train_loss = running_total_loss/batch_num
                running_real_loss += real_logits_loss.item()
                real_train_loss = running_real_loss/batch_num
                running_recomposed_loss += recomposed_logits_loss.item()
                recomposed_train_loss = running_recomposed_loss/batch_num
                running_contra_loss += contrastive_loss.item()
                contra_train_loss = running_contra_loss/batch_num

                # accuracy
                real_predictions = real_logits.argmax(dim=-1)
                recomposed_predictions = recomposed_logits.argmax(dim=-1)
                mask_acc = batch['real_harmony_ids'] != -100
                running_real_accuracy += (real_predictions[mask_acc] == batch['real_harmony_ids'][mask_acc].to(device)).sum().item()/max(1,mask_acc.sum().item())
                real_accuracy = running_real_accuracy/batch_num
                running_recomposed_accuracy += (recomposed_predictions[mask_acc] == batch['recomposed_harmony_ids'][mask_acc].to(device)).sum().item()/max(1,mask_acc.sum().item())
                recomposed_accuracy = running_recomposed_accuracy/batch_num

                # top5
                k = 5

                topk_real = torch.topk(real_logits, k, dim=-1).indices
                real_target = batch['real_harmony_ids'].to(device).unsqueeze(-1)
                correct_topk_real = (topk_real == real_target).any(dim=-1)

                topk_recomposed = torch.topk(recomposed_logits, k, dim=-1).indices
                recomposed_target = batch['recomposed_harmony_ids'].to(device).unsqueeze(-1)
                correct_topk_recomposed = (topk_recomposed == recomposed_target).any(dim=-1)

                mask = batch['mask_token_positions'].to(device)
                running_real_top5 += correct_topk_real[mask].sum().item() / max(1, mask.sum().item())
                real_top5 = running_real_top5/batch_num

                running_recomposed_top5 += correct_topk_recomposed[mask].sum().item() / max(1, mask.sum().item())
                recomposed_top5 = running_recomposed_top5/batch_num

                tepoch.set_postfix(
                    loss=total_train_loss,
                    acc_real=real_accuracy,
                    acc_rcn=recomposed_accuracy
                )
                step += 1
                if step%(total_steps//(epochs*validations_per_epoch)) == 0 or step == total_steps:
                    best_val_loss, saving_version = validation_bilstm_loop(
                        transformer_model, bilstm_model,
                        valloader,
                        mask_token_id, bar_token_id,
                        logits_loss_fn,
                        epoch,
                        step,
                        total_train_loss,
                        real_train_loss,
                        recomposed_train_loss,
                        contra_train_loss,
                        real_accuracy,
                        recomposed_accuracy,
                        real_top5,
                        recomposed_top5,
                        best_val_loss, saving_version,
                        results_path=results_path,
                        transformer_path=transformer_path,
                        bilstm_model_path=bilstm_model_path,
                        tqdm_position=tqdm_position
                    )
            # end for batch
        # end with tqdm
    # end for epoch
# end train_bilstm_loop

# ================================================================
# Adapter training
# ================================================================

def validation_adapter_loop(
        transformer_model, graph_model, bilstm_model, adapter_model,
        valloader,
        mask_token_id, bar_token_id,
        logits_loss_fn,
        epoch,
        step,
        total_train_loss,
        real_train_loss,
        recomposed_train_loss,
        contrastive_train_loss,
        real_train_accuracy,
        recomposed_train_accuracy,
        real_train_top5,
        recomposed_train_top5,
        best_val_loss, saving_version,
        results_path=None,
        transformer_path=None, 
        graph_model_path=None,
        bilstm_model_path=None,
        adapter_model_path=None,
        tqdm_position=0
    ):
    device = transformer_model.device
    transformer_model.eval()
    graph_model.eval()
    bilstm_model.eval()
    adapter_model.eval()
    with torch.no_grad():
        running_total_loss = 0
        total_val_loss = 0
        running_real_loss = 0
        real_val_loss = 0
        running_recomposed_loss = 0
        recomposed_val_loss = 0
        running_contra_loss = 0
        val_contra_loss = 0

        real_val_accuracy = 0
        running_real_accuracy = 0
        recomposed_val_accuracy = 0
        running_recomposed_accuracy = 0

        real_val_top5 = 0
        running_real_top5 = 0
        recomposed_val_top5 = 0
        running_recomposed_top5 = 0

        batch_num = 0
        print('validation')
        with tqdm(valloader, unit='batch', position=tqdm_position) as tepoch:
            tepoch.set_description(f'Epoch {epoch}@{step}| val')
            for batch in tepoch:
                real_graph_guide_z = graph_model(batch['real_graph'].to(device))
                real_bilstm_guide_z = bilstm_model(batch['real_bilstm'].to(device), batch['real_lengths'].to(device))
                real_constraints = batch['real_harmony_ids'].clone()
                real_constraints[batch['mask_token_positions']] = mask_token_id
                real_guide_z = adapter_model(real_graph_guide_z, real_bilstm_guide_z)

                recomposed_graph_guide_z = graph_model(batch['recomposed_graph'].to(device))
                recomposed_bilstm_guide_z = bilstm_model(batch['recomposed_bilstm'].to(device), batch['recomposed_lengths'].to(device))
                recomposed_constraints = batch['recomposed_harmony_ids'].clone()
                recomposed_constraints[batch['mask_token_positions']] = mask_token_id
                recomposed_guide_z = adapter_model(recomposed_graph_guide_z, recomposed_bilstm_guide_z)

                real_logits = transformer_model(
                    batch['pianoroll'].to(device),
                    real_constraints.to(device),
                    real_guide_z.to(device)
                )

                recomposed_logits = transformer_model(
                    batch['pianoroll'].to(device),
                    recomposed_constraints.to(device),
                    recomposed_guide_z.to(device)
                )

                mask = batch['mask_token_positions'].to(device)
                real_target = batch['real_harmony_ids'].clone().to(device)
                # real_target[~mask] = -100
                recomposed_target = batch['recomposed_harmony_ids'].clone().to(device)
                # recomposed_target[~mask] = -100

                real_logits_loss = logits_loss_fn(
                    real_logits.view(-1, real_logits.size(-1)),
                    real_target.view(-1)
                )

                recomposed_logits_loss = logits_loss_fn(
                    recomposed_logits.view(-1, recomposed_logits.size(-1)),
                    recomposed_target.view(-1)
                )

                contrastive_val_loss = guidance_contrastive_loss(real_guide_z, recomposed_guide_z)

                loss = real_logits_loss + recomposed_logits_loss + contrastive_weight * contrastive_val_loss
                # loss = real_logits_loss + recomposed_logits_loss
                # loss = recomposed_logits_loss

                # update loss and accuracy
                # loss
                batch_num += 1
                running_total_loss += loss.item()
                total_val_loss = running_total_loss/batch_num
                running_real_loss += real_logits_loss.item()
                real_val_loss = running_real_loss/batch_num
                running_recomposed_loss += recomposed_logits_loss.item()
                recomposed_val_loss = running_recomposed_loss/batch_num
                running_contra_loss += contrastive_val_loss.item()
                val_contra_loss = running_contra_loss/batch_num

                # accuracy
                real_predictions = real_logits.argmax(dim=-1)
                recomposed_predictions = recomposed_logits.argmax(dim=-1)
                mask_acc = batch['real_harmony_ids'] != -100
                running_real_accuracy += (real_predictions[mask_acc] == batch['real_harmony_ids'][mask_acc].to(device)).sum().item()/max(1,mask_acc.sum().item())
                real_val_accuracy = running_real_accuracy/batch_num
                running_recomposed_accuracy += (recomposed_predictions[mask_acc] == batch['recomposed_harmony_ids'][mask_acc].to(device)).sum().item()/max(1,mask_acc.sum().item())
                recomposed_val_accuracy = running_recomposed_accuracy/batch_num

                # top5
                k = 5

                topk_real = torch.topk(real_logits, k, dim=-1).indices
                real_target = batch['real_harmony_ids'].to(device).unsqueeze(-1)
                correct_topk_real = (topk_real == real_target).any(dim=-1)

                topk_recomposed = torch.topk(recomposed_logits, k, dim=-1).indices
                recomposed_target = batch['recomposed_harmony_ids'].to(device).unsqueeze(-1)
                correct_topk_recomposed = (topk_recomposed == recomposed_target).any(dim=-1)

                mask = batch['mask_token_positions'].to(device)
                running_real_top5 += correct_topk_real[mask].sum().item() / max(1, mask.sum().item())
                real_val_top5 = running_real_top5/batch_num

                running_recomposed_top5 += correct_topk_recomposed[mask].sum().item() / max(1, mask.sum().item())
                recomposed_val_top5 = running_recomposed_top5/batch_num

                tepoch.set_postfix(
                    loss=total_val_loss,
                    acc_real=real_val_accuracy,
                    acc_rcn=recomposed_val_accuracy
                )
            # end for batch
        # end with tqdm
    # end with no grad
    if transformer_path is not None:
        if  best_val_loss > total_val_loss:
            print('saving!')
            saving_version += 1
            best_val_loss = total_val_loss
            torch.save(transformer_model.state_dict(), transformer_path)
            torch.save(graph_model.state_dict(), graph_model_path)
            torch.save(bilstm_model.state_dict(), bilstm_model_path)
            torch.save(adapter_model.state_dict(), adapter_model_path)
    print(f'validation: loss={total_val_loss}, acc_real={real_val_accuracy}, acc_rnd={recomposed_val_accuracy}')
    print('results_path: ', results_path)
    if results_path is not None:
        with open( results_path, 'a' ) as f:
            writer = csv.writer(f)
            writer.writerow( [epoch, step, \
                        total_train_loss, real_train_loss,  recomposed_train_loss, contrastive_train_loss, \
                        real_train_accuracy, recomposed_train_accuracy, real_train_top5, recomposed_train_top5, \
                        total_val_loss, real_val_loss,  recomposed_val_loss, val_contra_loss, \
                        real_val_accuracy, recomposed_val_accuracy, real_val_top5, recomposed_val_top5, \
                        saving_version] )
    return best_val_loss, saving_version
# end validation_graph_loop

def train_adapter_loop(
        transformer_model, graph_model, bilstm_model, adapter_model,
        logits_loss_fn,
        optimizer, trainloader, valloader, mask_token_id,
        use_contra=False,
        epochs=100,
        results_path=None,
        transformer_path=None,
        graph_model_path=None,
        bilstm_model_path=None,
        adapter_model_path=None,
        bar_token_id=None,
        validations_per_epoch=1,
        tqdm_position=0,
        freeze_base=True
    ):
    device = transformer_model.device
    print('device: ', device)
    graph_model.to(device)
    best_val_loss = np.inf
    saving_version = 0

    # save results and model
    print('results_path:', results_path)
    if results_path is not None:
        result_fields = ['epoch', 'step', \
                        'total_train_loss', 'real_train_loss',  'rec_train_loss', 'contra_train_loss', \
                        'real_train_acc', 'rec_train_acc', \
                        'real_train_top5', 'rec_train_top5', \
                        'total_val_loss', 'real_val_loss',  'rec_val_loss', 'contra_val_loss', \
                        'real_val_acc', 'rec_val_acc', \
                        'real_val_top5', 'rec_val_top5', \
                        'sav_version']
        with open( results_path, 'w' ) as f:
            writer = csv.writer(f)
            writer.writerow( result_fields )

    # Compute total training steps
    total_steps = len(trainloader) * epochs
    # Define the scheduler
    # warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    step = 0

    for epoch in range(epochs):
        running_total_loss = 0
        total_train_loss = 0
        running_real_loss = 0
        real_train_loss = 0
        running_recomposed_loss = 0
        recomposed_train_loss = 0
        running_contra_loss = 0
        contra_train_loss = 0

        real_accuracy = 0
        running_real_accuracy = 0
        recomposed_accuracy = 0
        running_recomposed_accuracy = 0

        real_top5 = 0
        running_real_top5 = 0
        recomposed_top5 = 0
        running_recomposed_top5 = 0

        batch_num = 0
        
        with tqdm(trainloader, unit='batch', position=tqdm_position) as tepoch:
            tepoch.set_description(f'Epoch {epoch} | trn')
            for batch in tepoch:
                transformer_model.train()
                graph_model.train()
                bilstm_model.train()
                adapter_model.train()
                if freeze_base:
                    transformer_model.freeze_base()
                
                real_graph_guide_z = graph_model(batch['real_graph'].to(device))
                real_bilstm_guide_z = bilstm_model(batch['real_bilstm'].to(device), batch['real_lengths'].to(device))
                real_constraints = batch['real_harmony_ids'].clone()
                real_constraints[batch['mask_token_positions']] = mask_token_id
                real_guide_z = adapter_model(real_graph_guide_z, real_bilstm_guide_z)

                recomposed_graph_guide_z = graph_model(batch['recomposed_graph'].to(device))
                recomposed_bilstm_guide_z = bilstm_model(batch['recomposed_bilstm'].to(device), batch['recomposed_lengths'].to(device))
                recomposed_constraints = batch['recomposed_harmony_ids'].clone()
                recomposed_constraints[batch['mask_token_positions']] = mask_token_id
                recomposed_guide_z = adapter_model(recomposed_graph_guide_z, recomposed_bilstm_guide_z)


                real_logits = transformer_model(
                    batch['pianoroll'].to(device),
                    real_constraints.to(device),
                    real_guide_z.to(device)
                )

                recomposed_logits = transformer_model(
                    batch['pianoroll'].to(device),
                    recomposed_constraints.to(device),
                    recomposed_guide_z.to(device)
                )

                mask = batch['mask_token_positions'].to(device)
                real_target = batch['real_harmony_ids'].clone().to(device)
                # real_target[~mask] = -100
                recomposed_target = batch['recomposed_harmony_ids'].clone().to(device)
                # recomposed_target[~mask] = -100

                real_logits_loss = logits_loss_fn(
                    real_logits.view(-1, real_logits.size(-1)),
                    real_target.view(-1)
                )
                
                recomposed_logits_loss = logits_loss_fn(
                    recomposed_logits.view(-1, recomposed_logits.size(-1)),
                    recomposed_target.view(-1)
                )

                contrastive_loss = guidance_contrastive_loss(real_guide_z, recomposed_guide_z)

                optimizer.zero_grad()
                loss = real_logits_loss + recomposed_logits_loss + use_contra * contrastive_weight * contrastive_loss
                # loss = real_logits_loss + recomposed_logits_loss
                # loss = recomposed_logits_loss
                loss.backward()
                optimizer.step()
                # scheduler.step()

                # update loss and accuracy
                # loss
                batch_num += 1
                running_total_loss += loss.item()
                total_train_loss = running_total_loss/batch_num
                running_real_loss += real_logits_loss.item()
                real_train_loss = running_real_loss/batch_num
                running_recomposed_loss += recomposed_logits_loss.item()
                recomposed_train_loss = running_recomposed_loss/batch_num
                running_contra_loss += contrastive_loss.item()
                contra_train_loss = running_contra_loss/batch_num

                # accuracy
                real_predictions = real_logits.argmax(dim=-1)
                recomposed_predictions = recomposed_logits.argmax(dim=-1)
                mask_acc = batch['real_harmony_ids'] != -100
                running_real_accuracy += (real_predictions[mask_acc] == batch['real_harmony_ids'][mask_acc].to(device)).sum().item()/max(1,mask_acc.sum().item())
                real_accuracy = running_real_accuracy/batch_num
                running_recomposed_accuracy += (recomposed_predictions[mask_acc] == batch['recomposed_harmony_ids'][mask_acc].to(device)).sum().item()/max(1,mask_acc.sum().item())
                recomposed_accuracy = running_recomposed_accuracy/batch_num

                # top5
                k = 5

                topk_real = torch.topk(real_logits, k, dim=-1).indices
                real_target = batch['real_harmony_ids'].to(device).unsqueeze(-1)
                correct_topk_real = (topk_real == real_target).any(dim=-1)

                topk_recomposed = torch.topk(recomposed_logits, k, dim=-1).indices
                recomposed_target = batch['recomposed_harmony_ids'].to(device).unsqueeze(-1)
                correct_topk_recomposed = (topk_recomposed == recomposed_target).any(dim=-1)

                mask = batch['mask_token_positions'].to(device)
                running_real_top5 += correct_topk_real[mask].sum().item() / max(1, mask.sum().item())
                real_top5 = running_real_top5/batch_num

                running_recomposed_top5 += correct_topk_recomposed[mask].sum().item() / max(1, mask.sum().item())
                recomposed_top5 = running_recomposed_top5/batch_num

                tepoch.set_postfix(
                    loss=total_train_loss,
                    acc_real=real_accuracy,
                    acc_rcn=recomposed_accuracy
                )
                step += 1
                if step%(total_steps//(epochs*validations_per_epoch)) == 0 or step == total_steps:
                    best_val_loss, saving_version = validation_adapter_loop(
                        transformer_model, graph_model, bilstm_model, adapter_model,
                        valloader,
                        mask_token_id, bar_token_id,
                        logits_loss_fn,
                        epoch,
                        step,
                        total_train_loss,
                        real_train_loss,
                        recomposed_train_loss,
                        contra_train_loss,
                        real_accuracy,
                        recomposed_accuracy,
                        real_top5,
                        recomposed_top5,
                        best_val_loss, saving_version,
                        results_path=results_path,
                        transformer_path=transformer_path,
                        graph_model_path=graph_model_path,
                        bilstm_model_path=bilstm_model_path,
                        adapter_model_path=adapter_model_path,
                        tqdm_position=tqdm_position
                    )
            # end for batch
        # end with tqdm
    # end for epoch
# end train_adapter_loop