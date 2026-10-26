from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset, CSGridMLM_collate_fn
from generate_utils import load_SELaC
import torch
from torch.utils.data import DataLoader
import pickle
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from eval_utils import layer_latent_dists

load_dotenv()

batchsize = 8

# tokenizer, special tokens and batch size =====================================
tokenizer = CSGridMLMTokenizer(
    fixed_length=80,
    quantization='4th',
    intertwine_bar_info=True,
    trim_start=False,
    use_pc_roll=True,
    use_full_range_melody=False
)

# load data ===================================================
print('loading datasets')
val_gjt = os.getenv('VAL_GJT')
val_nott = os.getenv('VAL_NOTT')

val_dataset_gjt = CSGridMLMDataset(val_gjt, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
val_dataset_nott = CSGridMLMDataset(val_nott, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

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

# load model ================================================
print('loading model')
device_name = 'cuda:1'

if device_name == 'cpu':
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        print('Selected device not available: ' + device_name)
# end device selection

model = load_SELaC(
    tokenizer,
    device,
    d_model=512,
    checkpoint_path=None,
)
model.eval()

# load banks and vectors =====================================
with open('data/banks.pickle', 'rb') as f:
    banks = pickle.load(f)

print('loading banks and vectors')
nt_train_states = banks['nott']['train']['states']
jazz_train_states = banks['jazz']['train']['states']

nt_test_states = banks['nott']['test']['states']
jazz_test_states = banks['jazz']['test']['states']

# run ========================================================

base_path = 'results/latex/'
os.makedirs(base_path, exist_ok=True)

inidividual_results_jazz = base_path + 'individual_jazz_dists.tex'
cascade_results_jazz = base_path + 'cascade_jazz_dists.tex'
inidividual_results_nott = base_path + 'individual_nott_dists.tex'
cascade_results_nott = base_path + 'cascade_nott_dists.tex'

columns = ["layer", "position_bank", "flatten_bank", "global_vector"]
inidividual_jazz_df = pd.DataFrame(columns=columns)
inidividual_nott_df = pd.DataFrame(columns=columns)
cascade_jazz_df = pd.DataFrame(columns=columns)
cascade_nott_df = pd.DataFrame(columns=columns)

print('individual', 20*'-')
print('nott2jazz', 20*'-')
for bank_key in ['position_bank', 'flatten_bank', 'global_vector']:
    # individual
    tmp_dists = layer_latent_dists(
        model,
        device,
        nottValLoader,
        jazz_train_states,
        bank_key,
        1,
        tokenizer.mask_token_id,
        tokenizer.bar_token_id,
        0.0
    )
    print(bank_key)
    print(tmp_dists)

print('jazz2nott', 20*'-')
for bank_key in ['position_bank', 'flatten_bank', 'global_vector']:
    tmp_dists = layer_latent_dists(
        model,
        device,
        gjtValLoader,
        nt_train_states,
        bank_key,
        1,
        tokenizer.mask_token_id,
        tokenizer.bar_token_id,
        0.0
    )
    print(bank_key)
    print(tmp_dists)

print('cascade', 20*'-')
print('nott2jazz', 20*'-')
for bank_key in ['position_bank', 'flatten_bank', 'global_vector']:
    # cascade
    tmp_dists = layer_latent_dists(
        model,
        device,
        nottValLoader,
        jazz_train_states,
        bank_key,
        1,
        tokenizer.mask_token_id,
        tokenizer.bar_token_id,
        1.0
    )
    print(bank_key)
    print(tmp_dists)

print('jazz2nott', 20*'-')
for bank_key in ['position_bank', 'flatten_bank', 'global_vector']:
    tmp_dists = layer_latent_dists(
        model,
        device,
        gjtValLoader,
        nt_train_states,
        bank_key,
        1,
        tokenizer.mask_token_id,
        tokenizer.bar_token_id,
        1.0
    )
    print(bank_key)
    print(tmp_dists)