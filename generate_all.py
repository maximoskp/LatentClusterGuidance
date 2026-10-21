from generate_utils import load_SELaC, generate_files_with_nucleus
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import pickle
from tqdm import tqdm
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset, CSGridMLM_collate_fn
import os

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

# load model ================================================
print('loading model')
device_name = 'cuda:0'

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

# generate ===========================================================

intervention_layers = [0,1,2,3,4,5,6,7,'all']
intervention_strengths = [0.25, 0.5, 0.75, 1]
for layer in intervention_layers:
    layer_str = 'L' + str(layer)
    for strength in intervention_strengths:
        strength_str = 'S' + str(strength)
        # layers and strengths
        if layer == 'all':
            strengths = {
                layer: strength
                for layer in range(len(model.encoder_layers))
            }
        else:
            strengths = {
                layer: 0
                for layer in range(len(model.encoder_layers))
            }
            strengths[layer] = strength
        print(f'{layer_str}_{strength_str}')
        # generate for bank keys
        for bank_key in ['position_bank', 'flatten_bank', 'global_vector']:
            print(f'{bank_key}')
            # not to jazz
            print('not to jazz')
            files_path = 'MIDIs/nottingham/real/'
            files = os.listdir(files_path)
            for file in tqdm(files):
                input_f_path = files_path + file
                piece_name = input_f_path.split('/')[-1].split('.')[0]
                gen_out = generate_files_with_nucleus(
                    model,
                    tokenizer,
                    input_f_path=input_f_path,
                    mxl_folder_out=None,
                    midi_folder_out=f'MIDIs/nott2jazz/gen/{bank_key}/{layer_str}_{strength_str}',
                    name_suffix=f'{piece_name}',
                    style_banks=jazz_train_states,
                    bank_key=bank_key,
                    steering_strengths=strengths,
                    steer_mask=None,# valid_positions.to(device),
                    use_constraints=False,
                    intertwine_bar_info=True,
                    normalize_tonality=True,
                    temperature=1.0,
                    p=0.9,
                    unmasking_order='certain',
                    create_gen=True,
                    create_real=False
                )
            # jazz to nott
            print('jazz to nott')
            files_path = 'MIDIs/jazz/real/'
            files = os.listdir(files_path)
            for file in tqdm(files):
                input_f_path = files_path + file
                piece_name = input_f_path.split('/')[-1].split('.')[0]
                gen_out = generate_files_with_nucleus(
                    model,
                    tokenizer,
                    input_f_path=input_f_path,
                    mxl_folder_out=None,
                    midi_folder_out=f'MIDIs/jazz2nott/gen/{bank_key}/{layer_str}_{strength_str}',
                    name_suffix=f'{piece_name}',
                    style_banks=jazz_train_states,
                    bank_key=bank_key,
                    steering_strengths=strengths,
                    steer_mask=None,# valid_positions.to(device),
                    use_constraints=False,
                    intertwine_bar_info=True,
                    normalize_tonality=True,
                    temperature=1.0,
                    p=0.9,
                    unmasking_order='certain',
                    create_gen=True,
                    create_real=False
                )