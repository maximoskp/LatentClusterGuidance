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
from eval_utils import single_layer_latent_influence, all_layers_latent_influence, make_imshow_of_inividual_layer_results, make_imshow_of_all_layers_results

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

inidividual_results_base_path = 'figs/layer_influence/individual/'
os.makedirs(inidividual_results_base_path, exist_ok=True)
every_results_base_path = 'figs/layer_influence/every/'
os.makedirs(every_results_base_path, exist_ok=True)

examined_strengths = np.round(np.linspace(0, 1, 11), 1)

individual_layer_results = {
    'nott2jazz': {},
    'jazz2nott': {}
}
all_layer_results = {
    'nott2jazz': {},
    'jazz2nott': {}
}

for bank_key in ['position_bank', 'flatten_bank', 'global_vector']:
    # individual
    individual_layer_results['nott2jazz'][bank_key] = single_layer_latent_influence(
        model,
        device,
        nottValLoader,
        jazz_train_states,
        bank_key,
        tokenizer.mask_token_id,
        tokenizer.bar_token_id,
        examined_strengths,
        num_layers=8
    )
    make_imshow_of_inividual_layer_results(
        individual_layer_results['nott2jazz'][bank_key],
        fig_path=inidividual_results_base_path + f'nott2jazz_{bank_key}.png'
    )
    individual_layer_results['jazz2nott'][bank_key] = single_layer_latent_influence(
        model,
        device,
        gjtValLoader,
        nt_train_states,
        bank_key,
        tokenizer.mask_token_id,
        tokenizer.bar_token_id,
        examined_strengths,
        num_layers=8
    )
    make_imshow_of_inividual_layer_results(
        individual_layer_results['jazz2nott'][bank_key],
        fig_path=inidividual_results_base_path + f'jazz2nott_{bank_key}.png'
    )
    # all
    all_layer_results['nott2jazz'][bank_key] = all_layers_latent_influence(
        model,
        device,
        nottValLoader,
        jazz_train_states,
        bank_key,
        tokenizer.mask_token_id,
        tokenizer.bar_token_id,
        examined_strengths,
        num_layers=8
    )
    make_imshow_of_all_layers_results(
        all_layer_results['nott2jazz'][bank_key],
        fig_path=every_results_base_path + f'nott2jazz_{bank_key}.png'
    )
    all_layer_results['jazz2nott'][bank_key] = all_layers_latent_influence(
        model,
        device,
        gjtValLoader,
        nt_train_states,
        bank_key,
        tokenizer.mask_token_id,
        tokenizer.bar_token_id,
        examined_strengths,
        num_layers=8
    )
    make_imshow_of_all_layers_results(
        all_layer_results['jazz2nott'][bank_key],
        fig_path=every_results_base_path + f'jazz2nott_{bank_key}.png'
    )