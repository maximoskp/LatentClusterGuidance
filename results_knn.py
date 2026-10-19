from models_style_steering import (
    collect_harmony_layer_states,
    tokenwise_knn_accuracy,
    statewise_knn_accuracy
)
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset
import torch
import pickle
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

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
train_gjt = os.getenv('TRAIN_GJT')
train_nott = os.getenv('TRAIN_NOTT')

val_gjt = os.getenv('VAL_GJT')
val_nott = os.getenv('VAL_NOTT')

train_dataset_gjt = CSGridMLMDataset(train_gjt, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
train_dataset_nott = CSGridMLMDataset(train_nott, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

val_dataset_gjt = CSGridMLMDataset(val_gjt, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
val_dataset_nott = CSGridMLMDataset(val_nott, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

# load banks and vectors =====================================
print('loading banks and vectors')
nt_train_states = banks['nott']['train']['states']
jazz_train_states = banks['jazz']['train']['states']

nt_test_states = banks['nott']['test']['states']
jazz_test_states = banks['jazz']['test']['states']

# Combine the two training sets into a reference bank.
reference_states = {
    layer: torch.cat(
        [nt_train_states[layer], jazz_train_states[layer]], dim=0
    )
    for layer in nt_train_states
}

reference_labels = torch.cat([
    torch.zeros(len(train_dataset_nott), dtype=torch.long),
    torch.ones(len(train_dataset_gjt), dtype=torch.long),
])

# reference_valid = torch.cat([nt_train_valid, jazz_train_valid])

query_states = {
    layer: torch.cat(
        [nt_test_states[layer], jazz_test_states[layer]], dim=0
    )
    for layer in nt_test_states
}

query_labels = torch.cat([
    torch.zeros(len(val_dataset_nott), dtype=torch.long),
    torch.ones(len(val_dataset_gjt), dtype=torch.long),
])

# query_valid = torch.cat([nt_test_valid, jazz_test_valid])

# run knn ============================================================================
print('running knn')
results = tokenwise_knn_accuracy(
    query_states,
    reference_states,
    reference_labels,
    query_labels,
    k=5,
    # reference_valid=reference_valid,
    # query_valid=query_valid,
    reference_valid=None,
    query_valid=None,
    metric="cosine",
)

rows = []
for layer, result in results.items():
    piece_accuracies = np.array(result['piece_accuracy'])
    query_labels_np = query_labels.numpy()
    nott_accuracies = np.mean( piece_accuracies[ query_labels_np == 0 ] )
    jazz_accuracies = np.mean( piece_accuracies[ query_labels_np == 1 ] )
    rows.append({
        'Layer': layer,
        'Total Accuracy': result['token_accuracy'],
        'Nottingham Accuracy': nott_accuracies,
        'Jazz Accuracy': jazz_accuracies,
    })
    print(layer, result["token_accuracy"])
    print(f'layer: {layer} | total_accuracy: {result["token_accuracy"]} - nott_accuracy: {nott_accuracies} - jazz_accuracy: {jazz_accuracies}')

results_df = pd.DataFrame(rows)
latex_path = 'results/latex/knn.tex'
os.makedirs(os.path.dirname(latex_path), exist_ok=True)
results_df.to_latex(latex_path, index=False, float_format='%.4f')