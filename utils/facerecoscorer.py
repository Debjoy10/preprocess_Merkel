# ========== ========== ========== ==========
# Obtain Face-Recognition Scores parallely
# ========== ========== ========== ==========

from scipy.ndimage.morphology import binary_dilation
import numpy as np
import webrtcvad
import librosa
import struct
from pathlib import Path
import face_recognition
import cv2
import glob
import subprocess
from resemblyzer import VoiceEncoder
import json
import pandas as pd
from tqdm import tqdm
import pickle
import random
import os
import argparse
from speechface import Recognizer
from compare_videos import video_avg_score
from talknet_utils import talknet_scores

# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description = "FaceRecoScorer");
parser.add_argument('--reference_file_ids', type=str, default='files/references.txt', help='File containing ids of correct manually annotated files')
parser.add_argument('--video_dir', type=str, default='/raid/nayak/cropped/pycrop/', help='Video Directory having predefined structure')
parser.add_argument('--face_reco_file', type=str, default='files/all_scenes.pickle', help='Face reco scores stored')
parser.add_argument('--videolist', type=str, default='', help='Path to file containing list of video files')
parser.add_argument('--out_reco_file', type=str, default='files/all_scenes_new.pickle', help='Output Face reco scores')
opt = parser.parse_args();

def main():
    # Read Reference Files
    with open(opt.reference_file_ids) as f:
        ref = [l.strip() for l in f.readlines()]
    
    # Reference file paths from IDs
    ref_files = []
    for id_ in ref:
        # Verify File Structure and change if necessary
        ref_files.append(os.path.join(opt.video_dir, '{}/{}.avi'.format(id_, id_)))
    
    # Recognizer Object
    recognizer = Recognizer(audio_eps = 0.65, video_eps = 0.50)
    
    # Read Face reco files
    face_reco_scoredict = None
    if opt.face_reco_file != '':
        with open(opt.face_reco_file, 'rb') as handle:
            face_reco_scoredict = pickle.load(handle)
    
    # Get reference files features to compare other videos with
    reference_feats = []
    for file in ref_files:
        fts = recognizer.get_features_video(file)
        reference_feats.append(fts)
    video_cluster_reps = np.array(reference_feats)

    # Run
    with open(opt.videolist, 'r') as f:
        videolist = [fi.strip() for fi in f.readlines()]
    
    # Loop
    for v in tqdm(videolist):
        if face_reco_scoredict and v in face_reco_scoredict.keys():
            face_reco_score = face_reco_scoredict[v]
        else:
            try:
                face_reco_score = video_avg_score(v, video_cluster_reps)
            except AssertionError as e:
                print("{}: {}".format(v, e))
                face_reco_score = 0
            face_reco_scoredict[v] = face_reco_score
    
    # Save Pickle
    with open(opt.out_reco_file, 'wb') as handle:
        pickle.dump(face_reco_scoredict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()