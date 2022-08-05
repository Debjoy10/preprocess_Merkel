import pandas as pd
from tqdm import tqdm
import pickle
import random
import argparse
from random import shuffle
import os

parser = argparse.ArgumentParser(description = "Metadata_Writer");
parser.add_argument('--dataset_dir', type=str, default='/raid/dsaha/Merkel_one', help='Dataset with text-audio-videos')
opt = parser.parse_args();

def main():
    metadata_arr = []
    dataset_dir = opt.dataset_dir

    # Collect File-IDs and Texts
    with open(os.path.join(dataset_dir,"metadata.csv"), 'w') as s:
        for v in [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]:
            if os.path.exists(os.path.join(dataset_dir, v, 'text.txt')) and os.path.join(dataset_dir, v, 'audio.wav') and \
               os.path.exists(os.path.join(dataset_dir, v, v+'.avi')) and os.path.join(dataset_dir, v, v+'_lips.avi'):
                textf = os.path.join(dataset_dir, v, 'text.txt')
                with open(textf, 'r') as f:
                    text = f.read()
                s.write(str(v)+"|"+text+"\n")
    print("Completed writing metafile")

    # Breaking into train and val (500-rest)
    with open(os.path.join(dataset_dir,"metadata.csv"), 'r') as s:
        lines = s.readlines()
        shuffle(lines)
        train_shuf = lines[200:]
        val_shuf = lines[:200]
        # Write Train Metadata
        with open(os.path.join(dataset_dir,"metadata_train.csv"), 'w') as ts:
            ts.writelines(train_shuf)
        # Write Val Metadata
        with open(os.path.join(dataset_dir,"metadata_val.csv"), 'w') as vs:
            vs.writelines(val_shuf)
    print("Completed writing metafile train and validation")  
    
if __name__ == '__main__':
    main()