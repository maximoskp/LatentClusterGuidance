from data_utils import CSGridMLMDataset
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
from dotenv import load_dotenv

load_dotenv()

# 12 transpositions
train_hook12 = os.getenv('TRAIN_HOOK12')
train_wiki12 = os.getenv('TRAIN_WIKI12')
train_gjt12 = os.getenv('TRAIN_GJT12')
train_nott12 = os.getenv('TRAIN_NOTT12')

val_hook12 = os.getenv('VAL_HOOK12')
val_wiki12 = os.getenv('VAL_WIKI12')
val_gjt12 = os.getenv('VAL_GJT12')
val_nott12 = os.getenv('VAL_NOTT12')

folders_12 = [
    train_hook12, train_wiki12, train_gjt12, train_nott12,
    val_hook12, val_wiki12, val_gjt12, val_nott12
]

tokenizer = CSGridMLMTokenizer(
    fixed_length=80,
    quantization='4th',
    intertwine_bar_info=True,
    trim_start=False,
    use_pc_roll=True,
    use_full_range_melody=False
)

for folder in folders_12:
    _ = CSGridMLMDataset(folder, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')