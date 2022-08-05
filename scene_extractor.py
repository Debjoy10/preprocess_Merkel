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

parser = argparse.ArgumentParser(description = "SceneExtractor");
parser.add_argument('--talknet_score_thresh', type=float, default=0.35, help='Talknet Score Threshold')
parser.add_argument('--face_score_thresh', type=float, default=0.98, help='Face recognition score threshold')
parser.add_argument('--scores_file_path', type=str, default='files/all_scenes_scores.csv', help='CSV File where Scores stored')
parser.add_argument('--out_path', type=str, default='files/include_file_ids.pickle', help='PKL File where final file ids will be saved')
opt = parser.parse_args();

def get_scores_by_fileid(csv_file_path):
    """
    Loads the presaved CSV File (saved by scenescorer) and gets scene scores by File-ID
    """
    df = pd.read_csv(csv_file_path)
    df_dict = df.to_dict(orient = 'list')
    
    # file_id_div - (T: File Key --> array(for scene --> scenes)(file-path_scene, talknet-score_scene, face-reco-scores_scene))
    file_id_div = {}
    for vpath, ts, fs in zip(*[list(df_dict[k]) for k in list(df_dict.keys())]):
        file_key = vpath.split('/')[-2]
        if file_key in file_id_div:
            file_id_div[file_key].append([vpath, ts, fs])
        else:
            file_id_div[file_key] = [[vpath, ts, fs]]
    return file_id_div

def dict_by_scenes(scores):
    """
    Scores = Scores corresponding to the list of scenes
    Returns Scene dict having key as each scene and value as scene speaker scores
    """
    scene_dict = {}
    for score in scores:
        scene_key = int(score[0].split('/')[-1].split('-')[0])
        if scene_key in scene_dict:
            scene_dict[scene_key].append(score)
        else:
            scene_dict[scene_key] = [score]
    return scene_dict

def get_best_spk_each_scene(scene_dict):
    """
    Gets the scene scores order by scene-id
    And determines best speaker for each scene
    """
    # Error magnitude
    # 0 - No error
    # 1 - Short Duration Error
    # 2 - Not present error
    # 3 - Low Score Error
    
    best_speaker_scene = []
    errors = []
    for scene_num in range(1, max(scene_dict.keys())+1):
        if scene_num not in scene_dict.keys():
            # print("Scene missing")
            # Since all videos are there this is not an error
            # best_speaker_scene.append(None)
            continue
            
        scene = scene_dict[scene_num]
        best_spk = None
        error_magnitude = 3
        for spk in scene:
            if spk[1] != "ERR" and spk[1] != "ERR_short_duration" and float(spk[1]) >= opt.talknet_score_thresh and float(spk[2]) >= opt.face_score_thresh:
                error_magnitude = 0
                if best_spk is None:
                    best_spk = spk
                elif float(best_spk[1]) < float(spk[1]):
                    best_spk = spk
                else:
                    pass
            elif spk[1] == "ERR" and error_magnitude > 2:
                error_magnitude = 2
            elif spk[1] == "ERR_short_duration" and error_magnitude > 1:
                error_magnitude = 1
            else:
                pass
        errors.append(error_magnitude)    
        best_speaker_scene.append(best_spk)
    return best_speaker_scene, errors

def get_full_score(file_id_div, k):
    """
    Calls Best Spk each scene and returns results for some File-ID
    """
    scores = file_id_div[k]
    spkscores, errors = get_best_spk_each_scene(dict_by_scenes(scores))
    return scores, spkscores, errors

def get_score(file_id_div, k):
    """
    Calls Best Spk each scene and returns processed results for some File-ID
    """    
    scores = file_id_div[k]
    spkscores, errors = get_best_spk_each_scene(dict_by_scenes(scores))
    if None not in spkscores:
        return 1, [s[0] for s in spkscores]
    else:
        return 0, [s[0] if s is not None else None for s in spkscores]

def main():
    file_id_div = get_scores_by_fileid(opt.scores_file_path)
    scene_total = 0
    scene_merkel = 0
    scene_dict = {}
    with open(opt.out_path.strip('.pickle')+'_readable.txt', 'w') as f:        
        for k in file_id_div.keys():
            score, spkscores = get_score(file_id_div, k)
            scene_total += 1
            if score == 1:
                f.write("{}|{}\n".format(k, spkscores))
                scene_dict[k] = spkscores
                scene_merkel += 1
    print("{}/{} Files included".format(scene_merkel, scene_total))
    with open(opt.out_path, 'wb') as handle:
        pickle.dump(scene_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
if __name__ == '__main__':
    main()