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
from shutil import rmtree
import argparse

# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description = "SceneWriter");
parser.add_argument('--include_files', type=str, default='files/include_file_ids.pickle', help='PKL File where final file ids are saved')
parser.add_argument('--lip_videos_dir', type=str, default='', help='Lip cropped videos folder')
parser.add_argument('--alter_lip_videos_dir', type=str, default='', help='Alternate Lip cropped videos folder')
parser.add_argument('--face_videos_dir', type=str, default='', help='Face cropped videos folder')
parser.add_argument('--wavs_dir', type=str, default='', help='Audios folder - pass empty to use video')
parser.add_argument('--metadata', type=str, default='', help='Metadata - Text from metadata only')
parser.add_argument('--text_dir', type=str, default='', help='Text Dir alternate to metadata')
parser.add_argument('--out_dir', type=str, default='Merkel_Single_Speaker', help='Output Directory')
parser.add_argument('--bash_out', type=str, default='files/to_run.sh', help='Bash File to run after running the script')
parser.add_argument('--warn', type=str, default='Y', help='Give warning on missing files(Y/N)')
parser.add_argument('--additional_scenes_file', type=str, default='', help='Additional Video to add (1 scene only, text should be present)')
opt = parser.parse_args();

def main():
    bash_cmds = []
    done = 0
    
    # Currently Extracting Text from metadata file only
    text_for_ids = {}
    if opt.metadata != '':
        with open(opt.metadata, 'r') as f:
            lines = f.readlines()
        for line in lines:
            text_for_ids[line.split('|')[0]] = line.split('|')[1].strip()
      
    # File-IDs to include
    with open(opt.include_files, 'rb') as handle:
        include_ids = pickle.load(handle)
    
    # Iterate on include_ids
    for file_id, scenes in tqdm(include_ids.items(), total = len(include_ids)):
        current_bash_cmds = []
        
        # Output Paths
        text_out_path = os.path.join(opt.out_dir, file_id, 'text.txt')
        audio_out_path = os.path.join(opt.out_dir, file_id, 'audio.wav')
        video_out_path = os.path.join(opt.out_dir, file_id, file_id + '.avi')
        lip_video_out_path = os.path.join(opt.out_dir, file_id, file_id + '_lips.avi')
        
        # Check output path
        if os.path.exists(os.path.join(opt.out_dir, file_id)):
            rmtree(os.path.join(opt.out_dir, file_id))
        os.makedirs(os.path.join(opt.out_dir, file_id))        
        
        # Text
        if file_id not in text_for_ids:
            text_file_path = os.path.join(opt.text_dir, file_id, 'text.txt')
            alt_text_file_path = os.path.join(opt.text_dir, file_id.split('_')[0], file_id.split('_')[1]+'.txt')
            if os.path.exists(text_file_path):
                with open(text_file_path, 'r') as fin:
                    text = fin.read().strip()
            if os.path.exists(alt_text_file_path):
                with open(alt_text_file_path, 'r') as fin:
                    text = fin.read().strip()
            else:
                if opt.warn == 'Y':
                    print("Text not found for {}".format(file_id))
                continue
        else:
            text = text_for_ids[file_id]
        
        # Video
        video_files = scenes
        video_files.sort()
        if len(video_files) == 1:
            current_bash_cmds.append("cp {} {}".format(video_files[0], video_out_path))
        else:
            current_bash_cmds.append("touch mylist.txt")
            for vfile in video_files:
                current_bash_cmds.append("echo 'file {}' >> mylist.txt".format(vfile))
            current_bash_cmds.append("ffmpeg -f concat -safe 0 -i mylist.txt -c copy {}".format(video_out_path))
            current_bash_cmds.append("rm mylist.txt")
            
        # Lip-Video
        lip_video_files = []
        present = True
        for scene in scenes:
            file1 = os.path.join(opt.lip_videos_dir, file_id, os.path.basename(scene))
            file2 = os.path.join(opt.alter_lip_videos_dir, file_id, os.path.basename(scene))
            if os.path.exists(file1):
                lip_video_files.append(file1)
            elif os.path.exists(file2):
                lip_video_files.append(file2)
            else:
                present = False
                break
        if not present:
            if opt.warn == 'Y':
                print("Lip Video not found for {}".format(file_id))
            continue
        video_files.sort()
        if len(lip_video_files) == 1:
            current_bash_cmds.append("cp {} {}".format(lip_video_files[0], lip_video_out_path))
        else:
            current_bash_cmds.append("touch mylist.txt")
            for vfile in lip_video_files:
                current_bash_cmds.append("echo 'file {}' >> mylist.txt".format(vfile))
            current_bash_cmds.append("ffmpeg -f concat -safe 0 -i mylist.txt -c copy {}".format(lip_video_out_path))
            current_bash_cmds.append("rm mylist.txt")
            
        # Audio
        wav_file = os.path.join(opt.wavs_dir, file_id+'.wav')
        if not os.path.isfile(wav_file):
            if opt.warn == 'Y':
                print("Audio not found for {}".format(file_id))
                print("Writing from Video ...")
            current_bash_cmds.append("ffmpeg -i {} -ab 160k -ac 1 -ar 16000 -vn {}".format(video_out_path, audio_out_path))
        else:
            current_bash_cmds.append("cp {} {}".format(wav_file, audio_out_path))
        # High-Pass Audio
        hp_audio_out_path = os.path.join(opt.out_dir, file_id, 'highpassaudio.wav')
        cmd = "sox {} -r 16k -b 16 -c 1 {} highpass 10\n".format(audio_out_path, hp_audio_out_path)
        current_bash_cmds.append(cmd)
        
        # Writing
        with open(text_out_path, 'w') as fout:
            fout.write(text)
        bash_cmds.extend(current_bash_cmds)
        done += 1
    
    # Additional Files
    add_dones = 0
    addscenes = []
    if opt.additional_scenes_file != '':
        with open(opt.additional_scenes_file, 'r') as f:
            addscenes = [x.strip() for x in f.readlines()]

        for ascene in addscenes:
            file_id = ascene.split('/')[-2]
            current_bash_cmds = []

            # Output Paths
            text_out_path = os.path.join(opt.out_dir, file_id, 'text.txt')
            audio_out_path = os.path.join(opt.out_dir, file_id, 'audio.wav')
            video_out_path = os.path.join(opt.out_dir, file_id, file_id + '.avi')
            lip_video_out_path = os.path.join(opt.out_dir, file_id, file_id + '_lips.avi')
            
            # Check output path
            if os.path.exists(os.path.join(opt.out_dir, file_id)):
                rmtree(os.path.join(opt.out_dir, file_id))
            os.makedirs(os.path.join(opt.out_dir, file_id))        
            
            # Video
            current_bash_cmds.append("cp {} {}".format(ascene, video_out_path))
            # Lip Video
            file1 = os.path.join(opt.lip_videos_dir, file_id, os.path.basename(ascene))
            file2 = os.path.join(opt.alter_lip_videos_dir, file_id, os.path.basename(ascene))
            if os.path.exists(file1):
                current_bash_cmds.append("cp {} {}".format(file1, video_out_path))
            elif os.path.exists(file2):
                current_bash_cmds.append("cp {} {}".format(file2, video_out_path))
            else:
                if opt.warn == 'Y':
                    print("Lip Video not found for {}".format(file_id))
                continue
            # Audio
            current_bash_cmds.append("ffmpeg -i {} -ab 160k -ac 1 -ar 16000 -vn {}".format(video_out_path, audio_out_path))
            # High-Pass Audio
            hp_audio_out_path = os.path.join(opt.out_dir, file_id, 'highpassaudio.wav')
            cmd = "sox {} -r 16k -b 16 -c 1 {} highpass 10\n".format(audio_out_path, hp_audio_out_path)
            current_bash_cmds.append(cmd)
            
            # Text
            if file_id not in text_for_ids:
                text_file_path = os.path.join(opt.text_dir, file_id, 'text.txt')
                if not os.path.exists(text_file_path) and opt.warn == 'Y':
                    print("Text not found for {}".format(file_id))
                    continue
                else:
                    with open(text_file_path, 'r') as fin:
                        text = fin.read().strip()
            else:
                text = text_for_ids[file_id]
            
            # Writing
            with open(text_out_path, 'w') as fout:
                fout.write(text)
            bash_cmds.extend(current_bash_cmds)
            add_dones += 1
    
    # Finishing up
    print("Total Files Done = {}/{}".format(done, len(include_ids)))
    print("Additional Files Done = {}/{}".format(add_dones, len(addscenes)))
    with open(opt.bash_out, 'w') as f:
        for cmd in bash_cmds:
            f.write(cmd + '\n')
    
if __name__ == '__main__':
    main()