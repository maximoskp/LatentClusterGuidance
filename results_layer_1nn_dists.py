from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset, CSGridMLM_collate_fn
import torch
from torch.utils.data import DataLoader
import pickle
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

batchsize = 8

with open('data/banks.pickle', 'rb') as f:
    banks = pickle.load(f)

with open('data/vectors.pickle', 'rb') as f:
    vectors = pickle.load(f)

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

# load banks and vectors =====================================
print('loading banks and vectors')
nt_train_states = banks['nott']['train']['states']
jazz_train_states = banks['jazz']['train']['states']

nt_test_states = banks['nott']['test']['states']
jazz_test_states = banks['jazz']['test']['states']