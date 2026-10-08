from generate_utils import load_SELaC, nucleus_token_by_token_generate
import torch
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import pickle
from tqdm import tqdm
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset, CSGridMLM_collate_fn
import matplotlib.pyplot as plt
from copy import deepcopy
import os
import umap
import numba
numba.set_num_threads(4)
from dotenv import load_dotenv

batchsize = 8

# Define data ========================================
load_dotenv()
train_gjt = os.getenv('TRAIN_GJT')
train_nott = os.getenv('TRAIN_NOTT')

val_gjt = os.getenv('VAL_GJT')
val_nott = os.getenv('VAL_NOTT')

# Load tokenizer ========================================
tokenizer = CSGridMLMTokenizer(
    fixed_length=80,
    quantization='4th',
    intertwine_bar_info=True,
    trim_start=False,
    use_pc_roll=True,
    use_full_range_melody=False
)

# Load dataset ========================================
train_dataset_gjt = CSGridMLMDataset(train_gjt, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
train_dataset_nott = CSGridMLMDataset(train_nott, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

val_dataset_gjt = CSGridMLMDataset(val_gjt, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
val_dataset_nott = CSGridMLMDataset(val_nott, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

# Make data loaders ========================================
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

# Define device ========================================
device_name = 'cuda:2'

if device_name == 'cpu':
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        print('Selected device not available: ' + device_name)
# end device selection

# Define models ===============================================
models_data = []
models_data.append({
    'path': 'saved_models/SELaC_pretrained/pretrained_epoch89_nvis0.pt',
    'name': 'epoch89_nvis0'
})
models_data.append({
    'path': 'saved_models/SELaC_pretrained/pretrained_epoch130_nvis50.pt',
    'name': 'epoch130_nvis50'
})
models_data.append({
    'path': 'saved_models/SELaC_pretrained/pretrained_epoch168_nvis50.pt',
    'name': 'epoch168_nvis50'
})
models_data.append({
    'path': 'saved_models/SELaC_pretrained/pretrained_epoch200_nvis3.pt',
    'name': 'epoch200_nvis3'
})

# Helper functions =======================================================
def update_latents_for_batch(model, batch, latent_steps_list, next_piece_idx):
    harmony_ids = batch["harmony_ids"]
    batch_size, seq_len = harmony_ids.shape

    # One ID per piece, repeated for each of its sequence positions.
    piece_ids = torch.arange(
        next_piece_idx,
        next_piece_idx + batch_size,
        dtype=torch.long,
    ).unsqueeze(1).expand(-1, seq_len)

    # Position within the piece; useful for ordering the path on hover.
    step_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    next_piece_idx += batch_size

    _, latent_steps = model(
        batch["pianoroll"].to(device),
        harmony_ids.to(device),
        get_layers_output=True,
    )

    latent_steps_list.append({
        "activations": {
            layer: activations.squeeze(1).cpu()
            for layer, activations in latent_steps.items()
        },
        "harmony_ids": harmony_ids.cpu(),
        "attention_mask": batch["attention_mask"].cpu(),
        "piece_ids": piece_ids.cpu(),
        "step_indices": step_indices.cpu(),
    })

    return next_piece_idx
# end update_latents_for_batch

# Run for each model =====================================================
for model_data in models_data:
    model_name = model_data['name']
    print(f'loading model: {model_name}')
    model = load_SELaC(
        tokenizer,
        device,
        guidance_dim=512,
        d_model=512,
        checkpoint_path=model_data['path'],
    )
    model.eval()

    # making folders =====================================================
    os.makedirs('figs', exist_ok=True)
    os.makedirs('figs/umap/', exist_ok=True)
    os.makedirs(f'figs/umap/{model_name}', exist_ok=True)

    os.makedirs('data', exist_ok=True)
    os.makedirs(f'data/{model_name}', exist_ok=True)
    os.makedirs(f'data/{model_name}/umap/', exist_ok=True)

    # start with empty steps list ==========================================
    gjt_train_latent_steps_list = []
    gjt_val_latent_steps_list = []

    nott_train_latent_steps_list = []
    nott_val_latent_steps_list = []

    # extracting latents from batches ===========================================
    print('extracting latents from batches')
    next_piece_idx = 0  # Reset for each dataset/split extraction loop
    with torch.inference_mode():
        for batch in gjtTrainLoader:
            next_piece_idx = update_latents_for_batch(model, batch, gjt_train_latent_steps_list, next_piece_idx)
        
        for batch in gjtValLoader:
            next_piece_idx = update_latents_for_batch(model, batch, gjt_val_latent_steps_list, next_piece_idx)
            
        for batch in nottTrainLoader:
            next_piece_idx = update_latents_for_batch(model, batch, nott_train_latent_steps_list, next_piece_idx)
            
        for batch in nottValLoader:
            next_piece_idx = update_latents_for_batch(model, batch, nott_val_latent_steps_list, next_piece_idx)
    # end torch.inference_mode()

    # extract harmony_ids =========================================
    print('extracting harmony_ids')
    # GJT TRAIN
    gjt_train_harmony_ids = torch.cat([
        item["harmony_ids"].reshape(-1)
        for item in gjt_train_latent_steps_list
    ])
    gjt_train_valid_mask = torch.cat([
        item["attention_mask"].reshape(-1).bool()
        for item in gjt_train_latent_steps_list
    ])
    # gjt_train_harmony_ids = gjt_train_harmony_ids[gjt_train_valid_mask].numpy()

    # GJT VAL
    gjt_val_harmony_ids = torch.cat([
        item["harmony_ids"].reshape(-1)
        for item in gjt_val_latent_steps_list
    ])
    gjt_val_valid_mask = torch.cat([
        item["attention_mask"].reshape(-1).bool()
        for item in gjt_val_latent_steps_list
    ])
    # gjt_val_harmony_ids = gjt_val_harmony_ids[gjt_val_valid_mask].numpy()

    # NOTT TRAIN
    nott_train_harmony_ids = torch.cat([
        item["harmony_ids"].reshape(-1)
        for item in nott_train_latent_steps_list
    ])
    nott_train_valid_mask = torch.cat([
        item["attention_mask"].reshape(-1).bool()
        for item in nott_train_latent_steps_list
    ])
    # nott_train_harmony_ids = nott_train_harmony_ids[nott_train_valid_mask].numpy()

    # NOTT VAL
    nott_val_harmony_ids = torch.cat([
        item["harmony_ids"].reshape(-1)
        for item in nott_val_latent_steps_list
    ])
    nott_val_valid_mask = torch.cat([
        item["attention_mask"].reshape(-1).bool()
        for item in nott_val_latent_steps_list
    ])
    # nott_val_harmony_ids = nott_val_harmony_ids[nott_val_valid_mask].numpy()

    # saving harmony_ids ===========================================
    print('saving harmony_ids')
    os.makedirs('data', exist_ok=True)

    with open(f'data/{model_name}/gjt_train_harmony_ids.pickle', 'wb') as handle:
        pickle.dump(gjt_train_harmony_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'data/{model_name}/gjt_val_harmony_ids.pickle', 'wb') as handle:
        pickle.dump(gjt_val_harmony_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'data/{model_name}/nott_train_harmony_ids.pickle', 'wb') as handle:
        pickle.dump(nott_train_harmony_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'data/{model_name}/nott_val_harmony_ids.pickle', 'wb') as handle:
        pickle.dump(nott_val_harmony_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # activations ====================================================
    print('extracting activations')
    gjt_train_latent_steps_dict = {}
    gjt_val_latent_steps_dict = {}

    nott_train_latent_steps_dict = {}
    nott_val_latent_steps_dict = {}


    for k in range(8):
        # GJT TRAIN
        activations = torch.cat([
            item["activations"][k].reshape(
                -1, item["activations"][k].shape[-1]
            )
            for item in gjt_train_latent_steps_list
        ], dim=0)

        # gjt_train_latent_steps_dict[k] = activations[gjt_train_valid_mask].numpy()
        gjt_train_latent_steps_dict[k] = activations.numpy()

        assert len(gjt_train_latent_steps_dict[k]) == len(gjt_train_harmony_ids)
        # GJT VAL
        activations = torch.cat([
            item["activations"][k].reshape(
                -1, item["activations"][k].shape[-1]
            )
            for item in gjt_val_latent_steps_list
        ], dim=0)

        # gjt_val_latent_steps_dict[k] = activations[gjt_val_valid_mask].numpy()
        gjt_val_latent_steps_dict[k] = activations.numpy()
        assert len(gjt_val_latent_steps_dict[k]) == len(gjt_val_harmony_ids)

        # NOTT TRAIN
        activations = torch.cat([
            item["activations"][k].reshape(
                -1, item["activations"][k].shape[-1]
            )
            for item in nott_train_latent_steps_list
        ], dim=0)

        # nott_train_latent_steps_dict[k] = activations[nott_train_valid_mask].numpy()
        nott_train_latent_steps_dict[k] = activations.numpy()
        assert len(nott_train_latent_steps_dict[k]) == len(nott_train_harmony_ids)
        
        # NOTT VAL
        activations = torch.cat([
            item["activations"][k].reshape(
                -1, item["activations"][k].shape[-1]
            )
            for item in nott_val_latent_steps_list
        ], dim=0)

        # nott_val_latent_steps_dict[k] = activations[nott_val_valid_mask].numpy()
        nott_val_latent_steps_dict[k] = activations.numpy()
        assert len(nott_val_latent_steps_dict[k]) == len(nott_val_harmony_ids)

    # saving activations dictionary ============================================
    print('saving activations dictionary')
    os.makedirs('data', exist_ok=True)

    with open(f'data/{model_name}/gjt_train_latent_steps_dict.pickle', 'wb') as handle:
        pickle.dump(gjt_train_latent_steps_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'data/{model_name}/gjt_val_latent_steps_dict.pickle', 'wb') as handle:
        pickle.dump(gjt_val_latent_steps_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'data/{model_name}/nott_train_latent_steps_dict.pickle', 'wb') as handle:
        pickle.dump(nott_train_latent_steps_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'data/{model_name}/nott_val_latent_steps_dict.pickle', 'wb') as handle:
        pickle.dump(nott_val_latent_steps_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # saving activations lists =================================================
    print('saving activations lists')
    with open(f'data/{model_name}/gjt_train_latent_steps_list.pickle', 'wb') as handle:
        pickle.dump(gjt_train_latent_steps_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'data/{model_name}/gjt_val_latent_steps_list.pickle', 'wb') as handle:
        pickle.dump(gjt_val_latent_steps_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'data/{model_name}/nott_train_latent_steps_list.pickle', 'wb') as handle:
        pickle.dump(nott_train_latent_steps_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'data/{model_name}/nott_val_latent_steps_list.pickle', 'wb') as handle:
        pickle.dump(nott_val_latent_steps_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # making umap data ==============================================================
    print('making umap data')
    umap_2d_per_layer = {}
    for k in range(8):
        tmp_umap = umap.UMAP(
            n_components=2,
            n_jobs=4,
            random_state=None,
            n_neighbors=100,
            min_dist=0.5,
            init="random"
        )

        gjt_train = gjt_train_latent_steps_dict[k]
        gjt_val = gjt_val_latent_steps_dict[k]
        nott_train = nott_train_latent_steps_dict[k]
        nott_val = nott_val_latent_steps_dict[k]

        group_sizes = [len(gjt_train), len(gjt_val), len(nott_train), len(nott_val)]
        all_latent_steps = np.concatenate(
            [gjt_train, gjt_val, nott_train, nott_val],
            axis=0
        )
        all_latent_steps_2d = tmp_umap.fit_transform(all_latent_steps)
        gjt_train_2d, gjt_val_2d, nott_train_2d, nott_val_2d = np.split(
            all_latent_steps_2d,
            np.cumsum(group_sizes)[:-1],
            axis=0
        )

        umap_2d_per_layer[k] = {
            'gjt_train_2d': gjt_train_2d,
            'gjt_val_2d': gjt_val_2d,
            'nott_train_2d': nott_train_2d,
            'nott_val_2d': nott_val_2d
        }

        plt.figure(figsize=(8, 6))
        plt.scatter(gjt_train_2d[:, 0], gjt_train_2d[:, 1], c='firebrick', marker='|', label='GJT train', alpha=0.01)
        plt.scatter(gjt_val_2d[:, 0], gjt_val_2d[:, 1], c='fuchsia', marker='1', label='GJT validation', alpha=0.1)
        plt.scatter(nott_train_2d[:, 0], nott_train_2d[:, 1], c='blue', marker='_', label='NoTT train', alpha=0.01)
        plt.scatter(nott_val_2d[:, 0], nott_val_2d[:, 1], c='cyan', marker='3', label='NoTT validation', alpha=0.1)
        plt.title(f'UMAP of latent layer {k}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figs/umap/{model_name}/umap_{k}.png', dpi=300)
        plt.close()
    # end for k

    # saving umap data ==================================================
    print('saving umap data')
    with open(f'data/{model_name}/umap/umap_2d_per_layer.pickle', 'wb') as handle:
        pickle.dump(umap_2d_per_layer, handle, protocol=pickle.HIGHEST_PROTOCOL)