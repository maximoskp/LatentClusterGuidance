from models_style_steering import (
    collect_harmony_layer_states,
    tokenwise_knn_accuracy,
)
from generate_utils import load_SELaC, nucleus_token_by_token_generate, generate_files_with_nucleus
from train_utils import full_to_partial_masking
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import pickle
from tqdm import tqdm
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset, CSGridMLM_collate_fn
import matplotlib.pyplot as plt
from copy import deepcopy
import os
from dotenv import load_dotenv

load_dotenv()

# helper functions for activation steering
def _piece_balanced_mean(states, valid_mask=None):
    """
    states:     (N, G, D)
    valid_mask: optional bool tensor (N, G); True = include position.

    First average positions within each piece, then average pieces.
    Thus, pieces receive equal weight even if their valid-position
    counts differ.
    """
    states = states.detach().float().cpu()

    if valid_mask is None:
        return states.mean(dim=(0, 1))

    valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool).cpu()

    if valid_mask.shape != states.shape[:2]:
        raise ValueError(
            f"valid_mask must have shape {tuple(states.shape[:2])}"
        )

    weights = valid_mask.unsqueeze(-1).float()
    position_counts = weights.sum(dim=1)  # (N, 1)

    piece_means = (states * weights).sum(dim=1)
    piece_means = piece_means / position_counts.clamp_min(1.0)

    # Exclude pieces with no valid positions.
    has_valid = position_counts.squeeze(-1) > 0

    if not has_valid.any():
        raise ValueError("No valid positions available.")

    return piece_means[has_valid].mean(dim=0)
# end _piece_balanced_means

def make_activation_steering_vectors(
    jazz_train_states,
    nottingham_train_states,
    jazz_valid=None,
    nottingham_valid=None,
):
    """
    Returns dictionaries keyed by layer index. Each value has shape (D,).

    jazz_valid / nottingham_valid:
        Optional masks shaped (N_pieces, G), True for positions to include.
        For example, exclude <bar> and other control-token positions.
    """
    if set(jazz_train_states) != set(nottingham_train_states):
        raise ValueError("Jazz and Nottingham must contain the same layers.")

    jazz_means = {}
    nottingham_means = {}
    contrastive_directions = {}
    unit_directions = {}

    for layer in sorted(jazz_train_states):
        jazz_mean = _piece_balanced_mean(
            jazz_train_states[layer],
            None if jazz_valid is None else jazz_valid,
        )
        nottingham_mean = _piece_balanced_mean(
            nottingham_train_states[layer],
            None if nottingham_valid is None else nottingham_valid,
        )

        # Direction from the Nottingham mean toward the Jazz mean.
        direction = jazz_mean - nottingham_mean

        jazz_means[layer] = jazz_mean
        nottingham_means[layer] = nottingham_mean
        contrastive_directions[layer] = direction
        unit_directions[layer] = F.normalize(direction, p=2, dim=0)

    return {
        "jazz_mean": jazz_means,
        "nottingham_mean": nottingham_means,
        "contrastive": contrastive_directions,
        "contrastive_unit": unit_directions,
    }
# end make_activation_steering_vectors

# tokenizer, special tokens and batch size =====================================
tokenizer = CSGridMLMTokenizer(
    fixed_length=80,
    quantization='4th',
    intertwine_bar_info=True,
    trim_start=False,
    use_pc_roll=True,
    use_full_range_melody=False
)

mask_token_id = tokenizer.mask_token_id
bar_token_id = tokenizer.bar_token_id

batchsize = 8

# load data ===================================================
print('loading datasets')
train_gjt = os.getenv('TRAIN_GJT')
train_nott = os.getenv('TRAIN_NOTT')

val_gjt = os.getenv('VAL_GJT')
val_nott = os.getenv('VAL_NOTT')

train_dataset_gjt = CSGridMLMDataset(train_gjt, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
train_dataset_nott = CSGridMLMDataset(train_nott, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

val_dataset_gjt = CSGridMLMDataset(val_gjt, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
val_dataset_nott = CSGridMLMDataset(val_nott, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

# dataloaders ===================================================
gjtTrainLoader = DataLoader(
    train_dataset_gjt,
    batch_size=batchsize,
    shuffle=False,
    drop_last=False,
    collate_fn=CSGridMLM_collate_fn
)

nottTrainLoader = DataLoader(
    train_dataset_nott,
    batch_size=batchsize,
    shuffle=False,
    drop_last=False,
    collate_fn=CSGridMLM_collate_fn
)

gjtValLoader = DataLoader(
    val_dataset_gjt,
    batch_size=batchsize,
    shuffle=False,
    drop_last=False,
    collate_fn=CSGridMLM_collate_fn
)

nottValLoader = DataLoader(
    val_dataset_nott,
    batch_size=batchsize,
    shuffle=False,
    drop_last=False,
    collate_fn=CSGridMLM_collate_fn
)

# device and model ==========================================
device_name = 'cuda:1'

if device_name == 'cpu':
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        print('Selected device not available: ' + device_name)
# end device selection

print('loading model')
model = load_SELaC(
    tokenizer,
    device,
    guidance_dim=512,
    d_model=512,
    checkpoint_path=None,
)
model.eval()

# make banks ================================================
print('making banks')
# These must contain explicit mask-token IDs, preserving bar tokens.
nt_train_states, nt_train_harmony_ids, nt_train_valid = collect_harmony_layer_states(
    model, nottTrainLoader, mask_token_id, bar_token_id, device=device
)
jazz_train_states, jazz_train_harmony_ids, jazz_train_valid = collect_harmony_layer_states(
    model, gjtTrainLoader, mask_token_id, bar_token_id, device=device
)

nt_test_states, nt_test_harmony_ids, nt_test_valid = collect_harmony_layer_states(
    model, nottValLoader, mask_token_id, bar_token_id, device=device
)
jazz_test_states, jazz_test_harmony_ids, jazz_test_valid = collect_harmony_layer_states(
    model, gjtValLoader, mask_token_id, bar_token_id, device=device
)

banks = {
    'jazz': {
        'train': {
            'states': jazz_train_states,
            'harmony_ids': jazz_train_harmony_ids,
            'valid': jazz_train_valid
        },
        'test': {
            'states': jazz_test_states,
            'harmony_ids': jazz_test_harmony_ids,
            'valid': jazz_test_valid
        }
    },
    'nott': {
        'train': {
            'states': nt_train_states,
            'harmony_ids': nt_train_harmony_ids,
            'valid': nt_train_valid
        },
        'test': {
            'states': nt_test_states,
            'harmony_ids': nt_test_harmony_ids,
            'valid': nt_test_valid
        }
    }
}

# making activation steering vectors ===============================
print('making activation steering vectors')
vectors = make_activation_steering_vectors(
    jazz_train_states,
    nt_train_states,
    jazz_valid=None,            # optional
    nottingham_valid=None # optional
)

with open('data/banks.pickle', 'wb') as f:
    pickle.dump(banks, f)

with open('data/vectors.pickle', 'wb') as f:
    pickle.dump(vectors, f)