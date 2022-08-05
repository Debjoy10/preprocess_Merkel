import os
import pandas as pd
import numpy as np
import glob
import cv2

def talknet_scores(video_id, talknet_dir = '/raid/dsaha/Merkel_TTS_Dataset/talknet_scores/'):
    '''
    Provides the talknet scores - whether on-screen person is speaking
    Input:
        videofile : Path to audio and video
    Output:
        Mean Score
    '''
    try:
        scores = pd.read_pickle(os.path.join(talknet_dir, video_id, 'pywork/scores.pckl'))
        tracks = pd.read_pickle(os.path.join(talknet_dir, video_id, 'pywork/tracks.pckl'))
        tracklen = max([max(track['track']['frame'].tolist()) for track in tracks]) + 1
        faces = [[] for i in range(tracklen)]

        for tidx, track in enumerate(tracks):
            score = scores[tidx]
            for fidx, frame in enumerate(track['track']['frame'].tolist()):
                s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
                s = np.mean(s)
                faces[frame].append({
                    'track':tidx, 
                    'score':float(s),
                    's':track['proc_track']['s'][fidx], 
                    'x':track['proc_track']['x'][fidx], 
                    'y':track['proc_track']['y'][fidx]
                })
        newfaces = [face for face in faces if face != []]

        # Length Verification ==>
        # cap = cv2.VideoCapture(merkeldir)
        # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check Scores
        scores = []
        for f in faces:
            if len(f) < 1:
                continue
            if len(f) > 1:
                return -np.inf
            scores.append(f[0]['score'])                
        return np.mean(scores)
    except Exception as e: 
        if str(e) == "max() arg is an empty sequence":
            return "ERR_short_duration"
        return "ERR"