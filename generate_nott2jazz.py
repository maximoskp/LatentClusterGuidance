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
import argparse

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script for training a selected contrastive space.')

    # Define arguments
    parser.add_argument('-l', '--layer', type=int, help='Intervention layer. -1 means all layers.', required=True)
    parser.add_argument('-s', '--strength', type=float, help='Strength to apply to intervention - from 0 in 1.', required=True)
    parser.add_argument('-g', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)

    device_name = 'cpu'

    # Parse the arguments
    args = parser.parse_args()
    
    layer = args.layer
    strength = args.strength
    if args.gpu is not None:
        if args.gpu > -1:
            device_name = 'cuda:' + str(args.gpu)

    tokenizer = CSGridMLMTokenizer(
        fixed_length=80,
        quantization='4th',
        intertwine_bar_info=True,
        trim_start=False,
        use_pc_roll=True,
        use_full_range_melody=False
    )

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
    jazz_train_states = banks['jazz']['train']['states']

    # generate ===========================================================

    layer_str = 'L' + str(layer)
    strength_str = 'S' + str(strength)
    # layers and strengths
    if layer == -1:
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
# end main

if __name__ == '__main__':
    main()