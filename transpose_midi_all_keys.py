import os
from pathlib import Path
import music21 as m21
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

train_hook = os.getenv('TRAIN_HOOK')
train_wiki = os.getenv('TRAIN_WIKI')
train_gjt = os.getenv('TRAIN_GJT')
train_nott = os.getenv('TRAIN_NOTT')

val_hook = os.getenv('VAL_HOOK')
val_wiki = os.getenv('VAL_WIKI')
val_gjt = os.getenv('VAL_GJT')
val_nott = os.getenv('VAL_NOTT')

folders_CA = [
    train_hook, train_wiki, train_gjt, train_nott,
    val_hook, val_wiki, val_gjt, val_nott
]

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

for in_folder, out_folder in zip(folders_CA, folders_12):
    print(f'runing for: {in_folder} - {out_folder}')
    # Define transposition intervals (-5 to +6)
    transposition_intervals = range(-5, 7)  # Includes -5, -4, ..., 0, ..., +6

    # Define input and output directories
    input_root = Path(in_folder)  # Change this to your directory
    output_root = Path(out_folder)        # Change this to where you want the results

    # Ensure output directory exists
    output_root.mkdir(parents=True, exist_ok=True)

    # Find all MusicXML files
    midi_files = [file for ext in ["*.mid", "*.midi", "*.xml", "*.mxl", "*.musicxml"] for file in input_root.rglob(ext)]

    # Setup progress bar
    with tqdm(total=len(midi_files) * (len(transposition_intervals) - 1), desc="Processing Files") as pbar:
        for input_path in midi_files:
            rel_path = input_path.relative_to(input_root)  # Preserve subfolder structure
            
            try:
                score = m21.converter.parse(input_path)  # Load the MIDI file
            except Exception as e:
                print(f"❌ Error loading {input_path}: {e}")
                continue  # Skip to the next file
            # Ensure output directory exists
            output_root.mkdir(parents=True, exist_ok=True)
            
            midi_extensions = {".mid", ".midi"}
            musicxml_extensions = {".xml", ".mxl", ".musicxml"}
            extension = input_path.suffix.lower()
            if extension in midi_extensions:
                output_ext = '.mid'
                write_type = 'midi'
            elif extension in musicxml_extensions:
                output_ext = '.xml'
                write_type = 'xml'
            else:
                print('ERROR: unknown extension')
            
            # Process transpositions
            for interval in transposition_intervals:
                try:
                    # Transpose and create output path
                    transposed_score = score.transpose(interval)
                    output_file = output_root / rel_path.with_stem(f"{input_path.stem}_tr_{interval}")
                    output_file = output_file.with_suffix(output_ext)
                    
                    # Create necessary subdirectories
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write to file
                    transposed_score.write(write_type, output_file)
                    pbar.update(1)  # Update progress bar
                    
                except Exception as e:
                    print(f"❌ Error transposing {input_path} by {interval}: {e}")
