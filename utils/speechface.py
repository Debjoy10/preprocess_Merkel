from scipy.ndimage.morphology import binary_dilation
import numpy as np
import webrtcvad
import librosa
import struct
from pathlib import Path
import face_recognition
import cv2
import glob
from resemblyzer import VoiceEncoder
import json
from tqdm.notebook import tqdm
import pickle
import random
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

class Recognizer:
    '''
    Class for speaker verification in videos.
    '''
    def __init__(self, merkel_audio_embedding_path = 'files/merkel_audio_embeddings_by_day.p', 
                merkel_face_embedding_path = 'files/merkel_video_embeddings_by_day.p', 
                merkel_audio_files = 'files/merkel_audio_files_by_day.json',
                merkel_face_files = 'files/merkel_video_files_by_day.json',
                merkel_audio_centroids_path = 'files/kmeans_audio_cluster_centers.npy',
                merkel_face_centroids_path = 'files/kmeans_video_cluster_centers.npy',
                audio_norm_target_dBFS = -30, vad_window_length = 30, 
                sampling_rate = 16000, vad_moving_average_width = 8,
                vad_max_silence_length = 6, audio_thresh=0.7, video_thresh=0.95, video_norm = False,
                num_audio_clusters = 5, num_video_clusters = 5, audio_eps = 0.5, video_eps = 0.25, min_samples=5,
                audio_dir = '/raid/nayak/Merkel_TTS_Dataset/wavs/', video_dir = '/raid/nayak/cropped/pycrop/'
        ):
        # audio parameters and models
        self.int16_max = (2 ** 15) - 1
        self.audio_norm_target_dBFS = audio_norm_target_dBFS
        self.vad_window_length = vad_window_length
        self.sampling_rate = sampling_rate
        self.vad_moving_average_width = vad_moving_average_width
        self.vad_max_silence_length = vad_max_silence_length
        self.encoder = VoiceEncoder()
        self.facenorm = video_norm
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        
        # File Names
        self.merkel_audio_embedding_path = merkel_audio_embedding_path 
        self.merkel_face_embedding_path = merkel_face_embedding_path
        self.merkel_audio_files = merkel_audio_files
        self.merkel_face_files = merkel_face_files
    
        # Embeddings
        self.audio_thresh = audio_thresh
        self.video_thresh = video_thresh
        self.get_embeddings_and_filenames(not os.path.isfile(merkel_audio_embedding_path), not os.path.isfile(merkel_face_embedding_path)) 
        self.process_name = ''
        
        # Do outlier detection and clustering
        # clustering utils        
        self.num_audio_clusters = num_audio_clusters 
        self.num_video_clusters = num_video_clusters
        self.audio_eps = audio_eps
        self.video_eps = video_eps
        self.min_samples = min_samples
        
        # Cluster
        if os.path.isfile(merkel_audio_centroids_path):
            print("Loading presaved audio centroids ...")
            self.audio_cluster_centroids = np.load(merkel_audio_centroids_path)
        else:
            print("Getting audio Clusters")
            self.audio_cluster_centroids = self.outlier_detection_and_cluster(self.merkel_audio_embeddings_by_day,
                                                                         self.merkel_audio_embeddings_by_day_files, 
                                                                         num_audio_clusters, audio_eps, min_samples)
            print("Saving centroids")
            np.save(merkel_audio_centroids_path, self.audio_cluster_centroids)
        
        if os.path.isfile(merkel_face_centroids_path):
            print("Loading presaved video centroids ...")
            self.video_cluster_centroids = np.load(merkel_face_centroids_path)
        else:
            print("Getting video Clusters")
            self.video_cluster_centroids = self.outlier_detection_and_cluster(self.merkel_embeddings_by_day, 
                                                                         self.merkel_embeddings_by_day_files,
                                                                         num_video_clusters, video_eps, min_samples)
            print("Saving centroids")
            np.save(merkel_face_centroids_path, self.video_cluster_centroids)

    def get_embeddings_and_filenames(self, writeaudio = True, writevideo = True):
        '''
        Writes the embeddings into files and/or load into class attrs
        '''
        if writeaudio:
            print("Generating Audio Embeddings")
            # Extract files - Audio
            merkel_audios = glob.glob(os.path.join(self.audio_dir, '*.wav'), recursive=False)
            self.merkel_audios_by_day = {}
            for aud in merkel_audios:
                ckey = aud.split('/')[-1].split('_')[0]
                if ckey in self.merkel_audios_by_day.keys():
                    self.merkel_audios_by_day[ckey].append(aud)
                else:
                    self.merkel_audios_by_day[ckey] = [aud]

            # Get Audio Embeddings
            self.merkel_audio_embeddings_by_day = {}
            self.merkel_audio_embeddings_by_day_files = {}
            for k, v in tqdm(self.merkel_audios_by_day.items()):
                try:
                    aud = random.sample(v, 1)[0]
                    self.merkel_audio_embeddings_by_day[k] = self.get_audio_embedding(aud)
                    self.merkel_audio_embeddings_by_day_files[k] = aud
                except:
                    pass
            
            # Write
            assert self.merkel_audio_embedding_path.endswith('.p')
            assert self.merkel_audio_files.endswith('.json')
            with open(self.merkel_audio_embedding_path, 'wb') as f:
                pickle.dump(self.merkel_audio_embeddings_by_day, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.merkel_audio_files, 'w') as f:
                json.dump(self.merkel_audio_embeddings_by_day_files, f)
                
        else:
            print("Loading presaved Audio Embeddings")
            assert self.merkel_audio_embedding_path.endswith('.p')
            assert self.merkel_audio_files.endswith('.json')
            with open(self.merkel_audio_embedding_path, 'rb') as fp:
                self.merkel_audio_embeddings_by_day = pickle.load(fp)
            with open(self.merkel_audio_files, 'r') as f:
                self.merkel_audio_embeddings_by_day_files = json.load(f)
                    
        if writevideo:
            print("Generating Video Embeddings")
            # Extract files - Video
            merkel_vids = glob.glob(os.path.join(self.video_dir, '*/*.avi'), recursive=True)
            merkel_vids = [v for v in merkel_vids if v.split('/')[-2] == v.split('/')[-1].strip('.avi')]
            self.merkel_vids_by_day = {}
            for vid in merkel_vids:
                ckey = vid.split('/')[-1].split('_')[0]
                if ckey in self.merkel_vids_by_day.keys():
                    self.merkel_vids_by_day[ckey].append(vid)
                else:
                    self.merkel_vids_by_day[ckey] = [vid]

            # Get Video Embeddings
            self.merkel_embeddings_by_day = {}
            self.merkel_embeddings_by_day_files = {}
            for k, v in tqdm(self.merkel_vids_by_day.items()):
                try:
                    vid = random.sample(v, 1)[0]
                    self.merkel_embeddings_by_day[k] = self.get_features_video(vid)
                    self.merkel_embeddings_by_day_files[k] = vid
                except:
                    pass

            # Write
            assert self.merkel_face_embedding_path.endswith('.p')
            assert self.merkel_face_files.endswith('.json')
            with open(self.merkel_face_embedding_path, 'wb') as f:
                pickle.dump(self.merkel_embeddings_by_day, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.merkel_face_files, 'w') as f:
                json.dump(self.merkel_embeddings_by_day_files, f)
                
        else:
            print("Loading presaved Video Embeddings")
            assert self.merkel_face_embedding_path.endswith('.p')
            assert self.merkel_face_files.endswith('.json')
            with open(self.merkel_face_embedding_path, 'rb') as fp:
                self.merkel_embeddings_by_day = pickle.load(fp)
            with open(self.merkel_face_files, 'r') as f:
                self.merkel_embeddings_by_day_files = json.load(f)
        
    '''
    Following three functions are borrowed from resemblyzer resemblyzer/audio.py.
    Some changes were needed so I copied them here.
    '''
    def preprocess_wav(self, audiofile):
        '''
        Preprocesses a audio file by removing silence and normalizing it
        Input:
            audiofile: Path to audio file
        Output:
            wav : numpy array which represents the audio
        '''
        wav, source_sr = librosa.load(Path(audiofile), sr=None)
        if source_sr is not None:
            wav = librosa.resample(wav, source_sr, self.sampling_rate)
        if len(wav.shape)==2:
            wav = np.mean(wav, axis=1)

        wav = self.normalize_volume(wav, self.audio_norm_target_dBFS, increase_only=True)
        wav = self.trim_long_silences(wav)
        return wav

    def normalize_volume(self, wav, target_dBFS, increase_only=False, decrease_only=False):
        '''
        Normalises the audio waveform.
        Inputs:
            wav : The audio waveform
            target_dBFS, increase_only, decrease_only : Some audio parameters
        Outputs:
            wav : Normalized audio wave form
        '''
        if increase_only and decrease_only:
            raise ValueError("Both increase only and decrease only are set")
        rms = np.sqrt(np.mean((wav * self.int16_max) ** 2))
        wave_dBFS = 20 * np.log10(rms / self.int16_max)
        dBFS_change = target_dBFS - wave_dBFS
        if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
            return wav
        return wav * (10 ** (dBFS_change / 20))

    def trim_long_silences(self, wav):
        '''
        Removes long silences from the audio
        Input:
            wav : The audio waveform
        Output:
            wav: audio waveform after silences have been removed.
        '''
        samples_per_window = (self.vad_window_length * self.sampling_rate) // 1000

        # Trim the end of the audio to have a multiple of the window size
        wav = wav[:len(wav) - (len(wav) % samples_per_window)]

        # Convert the float waveform to 16-bit mono PCM
        pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * self.int16_max)).astype(np.int16))

        # Perform voice activation detection
        voice_flags = []
        vad = webrtcvad.Vad(mode=3)
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                             sample_rate=self.sampling_rate))
        voice_flags = np.array(voice_flags)

        # Smooth the voice detection with a moving average
        def moving_average(array, width):
            array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
            ret = np.cumsum(array_padded, dtype=float)
            ret[width:] = ret[width:] - ret[:-width]
            return ret[width - 1:] / width

        audio_mask = moving_average(voice_flags, self.vad_moving_average_width)
        audio_mask = np.round(audio_mask).astype(np.bool)

        # Dilate the voiced regions
        audio_mask = binary_dilation(audio_mask, np.ones(self.vad_max_silence_length + 1))
        audio_mask = np.repeat(audio_mask, samples_per_window)

        return wav[audio_mask == True]

    def cosine_similarity(self, vec1, vec2):
        '''
        Calculates the cosine similarity between two vectors
        Input:
            vec1, vec2 : Two vectors
        Output:
            cosine similarity score (vec1.vec2/(norm(vec1)*norm(vec2)))
        '''
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))

    def get_speaker_embedding(self, filenames):
        '''
        Return audio embeddings - group of files
        Input:
            filenames : Audio files containing sound/utterance
        Output:
            None
        '''
        wavs = []
        for file in filenames:
            wavs.append(self.preprocess_wav(file))
        self.merkel_embedding =  self.encoder.embed_speaker(wavs)

    def get_audio_embedding(self, audiofile):
        '''
        Return audio embeddings - One file
        Input:
            filenames : Audio file containing sound/utterance
        Output:
            None
        '''
        wav = self.preprocess_wav(audiofile)
        time = librosa.get_duration(filename=audiofile)
        embedding = self.encoder.embed_utterance(wav)
        return embedding

    def audio(self, audiofile):
        '''
        Compares given audio with existing speaker embeddings and return True if similarity is above a threshold.
        Input:
            audiofile : Audio file containing sound/utterance
        Output:
            True / False
        '''
        wav = self.preprocess_wav(audiofile)
        time = librosa.get_duration(filename=audiofile)
        embedding = self.encoder.embed_utterance(wav)
        max_sim_score = 0
        for audio_centroid in self.audio_cluster_centroids:
            sim_score = self.cosine_similarity(embedding, audio_centroid)
            max_sim_score = max(max_sim_score, sim_score)
        return max_sim_score
    
    def get_features_video(self, videofile):
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
        video_avg_face = None
        video_avg_N = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break
            if counter % check_every != 0:
                counter+=1
                continue
            counter+=1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_frame)
            if len(face_encodings) == 0:
                continue
            elif len(face_encodings) > 1:
                raise AssertionError("Two Faces detected in videofile {}".format(videofile))

            if video_avg_face is None:
                video_avg_face = face_encodings[0]
                video_avg_N += 1
            else:
                video_avg_face += face_encodings[0]
                video_avg_N += 1

        video.release()
        if video_avg_N == 0:
            raise AssertionError("No Faces detected in videofile {}".format(videofile))
        true_video_avg = video_avg_face/video_avg_N
        if self.facenorm:
            norm_video_avg = true_video_avg/np.linalg.norm(true_video_avg)
            return norm_video_avg
        else:
            return true_video_avg
    
    def video(self, videofile):
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
                
            if self.facenorm:
                face = face_encodings[0]/np.linalg.norm(face_encodings[0])
            else:
                face = face_encodings[0]
            
            # Eval - Simplified              
            match = face_recognition.compare_faces(self.video_cluster_centroids, face, tolerance=0.6)
            truth = list(map(int, match))
            frame_truth = np.max(truth)
            frame_truths.append(frame_truth)
        
        video.release()
        if len(frame_truths) == 0:
            raise AssertionError("No Faces detected in videofile {}".format(videofile))
        score = np.min(frame_truths)
        return score

    def recognize(self, audiofile, videofile):
        '''
        Combines the results of both audio and video
        Input:
            audiofile, videofile : Path to audio and video
        Output:
            True/False: True if both audio and video functions return true
        '''
        # Audio Score
        try:
            audio_truth = self.audio(audiofile)
        except:
            audio_truth = 0
        
        # Face Score
        try:
            video_truth = self.video(videofile)
        except:
            video_truth = 0
        
        # Talknet Score
        try:
            tn_truth = self.talknet_scores(videofile)
        except:
            tn_truth = -np.inf
            
        return {"audio_score": audio_truth, "video_score": video_truth, "talknet_score": tn_truth}
            
    def outlier_detection_and_cluster(self, embeddings_per_day, files_per_day, num_clusters = 5, eps = 0.5, min_samples = 5):
        '''
        Clustering the embeddings
        '''
        keys, values = zip(*embeddings_per_day.items())
        X = np.array(values)
        filenames = [files_per_day[k] for k in keys]
        # Clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        
        print("The following are the outlier keys - ")
        outlier_keys = []
        for key, label in zip(filenames, dbscan.labels_):
            if label == -1: 
                print(" > {}".format(key))
                outlier_keys.append(key)
                
        print("The following are the proper keys - ")
        proper_keys = []
        for key, label in zip(filenames, dbscan.labels_):
            if label != -1: 
                print(" > {}".format(key))
                proper_keys.append(key)
        
        upload = False
        if upload:
            for o in random.sample(outlier_keys, 5):
                os.system("gupload {}".format(o))
            for o in random.sample(proper_keys, 5):
                os.system("gupload {}".format(o))
               
        # PCA
        df = pd.DataFrame()
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(values)
        df['pca-one'] = pca_result[:,0]
        df['pca-two'] = pca_result[:,1] 
        df['pca-three'] = pca_result[:,2]
        df['keys'] = [f.split('/')[-1] for f in filenames]
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
        df["y"] = dbscan.labels_
        rndperm = np.random.permutation(df.shape[0])
        
        # Plot
        plt.figure(figsize=(16,10))
        sns.scatterplot(
            x="pca-one", y="pca-two",
            hue="y",
            palette=sns.color_palette("hls", len(np.unique(df["y"]))),
            data=df.loc[rndperm,:],
            legend="full",
            alpha=0.3
        )
        for i in range(df.shape[0]):
            if df['y'][i] != -1: continue
            plt.text(x=df['pca-one'][i]+0.005,y=df['pca-two'][i]+0.005,s=df['keys'][i], 
                  fontdict=dict(color='red',size=10),
                  bbox=dict(facecolor='yellow',alpha=0.5))
        plt.show()
        
        # Cluster non-outlier values
        outlier_keyids = [v.split('/')[-1].split('_')[0] for v in outlier_keys]
        keys, values = zip(*{k: v for k, v in embeddings_per_day.items() if k not in outlier_keyids}.items())
        print("Keys Retained = {}/{}".format(len(keys), len(embeddings_per_day)))
        
        # K-means
        X = np.array(values)
        kmeans = KMeans(n_clusters = num_clusters, random_state = 0).fit(X)
        
        # Plots
        df = pd.DataFrame()
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(np.array(list(values) + list(kmeans.cluster_centers_)))
        df['pca-one'] = pca_result[:len(values),0]
        df['pca-two'] = pca_result[:len(values),1] 
        df['pca-three'] = pca_result[:len(values),2]
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
        df["y"] = kmeans.labels_

        rndperm = np.random.permutation(df.shape[0])
        plt.figure(figsize=(16,10))
        colors = ['red', 'blue', 'purple', 'green']

        # Cluster Centroids
        ax = sns.scatterplot(
            x="pca-one", y="pca-two",
            hue="y",
            palette=sns.color_palette("hls", len(np.unique(df["y"]))),
            data=df.loc[rndperm,:],
            legend="full",
            alpha=0.3
        )
        ax = sns.scatterplot(pca_result[len(values):,0], pca_result[len(values):,1],
                             hue=range(num_clusters), s=50, ec='black', legend=False, ax=ax)
        plt.show()
        return kmeans.cluster_centers_
    
    def talknet_scores(self, videofile):
        '''
        Provides the talknet scores - whether on-screen person is speaking
        Input:
            videofile : Path to audio and video
        Output:
            Mean Score
        '''
        video_id = videofile.split('/')[-1].strip('.avi')
        merkeldir = videofile

        try:
            scores = pd.read_pickle(os.path.join('/raid/dsaha/Merkel_TTS_Dataset/talknet_scores/', video_id, 'pywork/scores.pckl'))
            tracks = pd.read_pickle(os.path.join('/raid/dsaha/Merkel_TTS_Dataset/talknet_scores/', video_id, 'pywork/tracks.pckl'))
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
        except:
            print("Not found {}".format(resultdir))
            return -np.inf