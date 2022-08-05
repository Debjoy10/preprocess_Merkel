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
from tqdm import tqdm
import pickle
import random
import os

def video_avg_score(videofile, video_cluster_reps):
    '''
    Returns the avg features
    Input:
        videofile : Path to the video
    Output:
        Np embeddings
    '''
    video = cv2.VideoCapture(videofile)
    counter = 0
    check_every = 5 
    frame_truths = []

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if counter % check_every != 0:
            counter+=1
            continue
        counter+=1

        # Processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)

        # Checks
        if len(face_encodings) == 0:
            continue
        elif len(face_encodings) > 1:
            raise AssertionError("Two Faces detected in videofile {}".format(videofile))

        face = face_encodings[0]
        
        # Eval - Simplified              
        match = face_recognition.compare_faces(video_cluster_reps, face, tolerance=10.7)
        truth = list(map(int, match))
        frame_truth = np.max(truth)
        frame_truths.append(frame_truth)

    video.release()
    if len(frame_truths) == 0:
        raise AssertionError("No Faces detected in videofile {}".format(videofile))
    score = np.mean(frame_truths)
    return score  