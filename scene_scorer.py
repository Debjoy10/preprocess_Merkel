# ========== ========== ========== ==========
# Write the 1. Face-Recognition Scores and 2. TalkNet Scores for future processing
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
from utils.speechface import Recognizer
from utils.compare_videos import video_avg_score
from utils.talknet_utils import talknet_scores

# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description = "SceneScorer");
parser.add_argument('--reference_file_ids', type=str, default='files/references.txt', help='File containing ids of correct manually annotated files')
parser.add_argument('--video_dir', type=str, default='', help='Video Directory having predefined structure')
parser.add_argument('--talknet_dir', type=str, default='', help='Talknet files stored')
parser.add_argument('--face_reco_file', type=str, default='', help='Face reco scores stored')
parser.add_argument('--video', type=str, default='', help='Video File Path, if need to run score generator for one video file')
parser.add_argument('--videolist', type=str, default='', help='Path to file containing list of video files')
parser.add_argument('--outpath', type=str, default='files/facetalk_scores.csv', help='CSV File to store Scores')
opt = parser.parse_args();


def main():
    # Read Reference Files
    with open(opt.reference_file_ids) as f:
        ref_files = [l.strip() for l in f.readlines()]
    
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
    
    # Check Values
    if len(opt.video) != 0 and len(opt.videolist) != 0:
        return Exception("Video and Videolist both provided values")
    if len(opt.video) == 0 and len(opt.videolist) == 0:
        return Exception("Video and Videolist both empty")
    
    # Runs on file formats .../Data_Id/Scene_id-Spk_id
    if len(opt.video) != 0:
        try:
            face_reco_score = video_avg_score(opt.video, video_cluster_reps)
        except AssertionError as e:
            print("{}: {}".format(opt.video, e))
            face_reco_score = 0
        talknet_filename = opt.video.split('/')[-2] + '_and_' + opt.video.split('/')[-1].strip('.avi')
        talknet_score = talknet_scores(talknet_filename, opt.talknet_dir)
        print("Scores for {}:".format(opt.video))
        print("###################################################")
        print("Talknet Score = {}".format(talknet_score))
        print("Face Recognition Score = {}".format(face_reco_score))
        print("###################################################")
    else:
        with open(opt.videolist, 'r') as f:
            videolist = [fi.strip() for fi in f.readlines()]
        analyzefacetalknet = {
            'filename': [],
            'talknet_score': [],
            'face_reco_score': [],
        }
        for v in tqdm(videolist):
            if face_reco_scoredict and v in face_reco_scoredict.keys():
                face_reco_score = face_reco_scoredict[v]
            else:
                try:
                    face_reco_score = video_avg_score(v, video_cluster_reps)
                except AssertionError as e:
                    print("{}: {}".format(v, e))
                    face_reco_score = 0
            talknet_filename = v.split('/')[-2] + '_and_' + v.split('/')[-1].strip('.avi')
            talknet_score = talknet_scores(talknet_filename, opt.talknet_dir)
            analyzefacetalknet['filename'].append(v)
            analyzefacetalknet['talknet_score'].append(talknet_score)
            analyzefacetalknet['face_reco_score'].append(face_reco_score)
        pd.DataFrame(analyzefacetalknet).to_csv(opt.outpath, index=False)            

if __name__ == '__main__':
    main()