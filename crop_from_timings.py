from pydub import AudioSegment
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import argparse
import sys, subprocess, math
import time
import os

def crop(times, text, audio_path, audio_save_dir, video_path=None, video_save_dir=None):
    """
    Used to crop the audio and video from the time given in lab files.
    Input:
        times: A List[Tuples] containing start and end times.
        audio_path: path to the audio file.
        audio_save_dir: path to the folder where cropped audio files have to be stored.
        video_path: path to the video clip.
        video_save_dir: path to the folder where cropped video files have to be stored.
    Output:
        None.
        Crops and saves audio and video clips.
    """
    audio = AudioSegment.from_wav(audio_path)
    for i, (t1, t2) in enumerate(times):
        id = '%02d' % i
        text_sequence = text[i]
        f = open(audio_save_dir + '/' + id + '.txt', 'w')
        f.write(text_sequence)
        f.close()

        t1_new = t1*1000
        t2_new = t2*1000
        cropped_audio = audio[t1_new:t2_new]
        cropped_audio.export(audio_save_dir + "/" + id +".wav", format="wav", parameters=["-ar", "16000"])
        if video_path is not None:
            extension = video_path.split('.')[-1]
            # ffmpeg command to crop
            command = ("ffmpeg -ss %s -i %s -t %s %s" % (t1, video_path, t2-t1, video_save_dir + "/" + id + "." + extension))
            output = subprocess.call(command, shell=True, stdout=None)

    print("Finished cropping and saving {} files".format(i+1))

def read_lab(path):
    """
    Used to read the lab file and extract time information from it.
    Input:
        path: path to the lab file
    Output:
        times: List[Tupes] containg start and end time of each utterance.
        text: List[String] containing texts of utterance
    """
    times = []
    text = []
    with open(path, 'r') as f:
        data = f.readlines()
        for d in data:
            d = d.split()
            t1, t2 = float(d[0]), float(d[1])
            text.append(' '.join(d[2:]))
            times.append((t1,t2))

    return times, text

def read_times(path):
    """
    Used to read the timings file and extract information.
    Input:
        path: path to the lab file
    Output:
        id: List[String] containing IDs of files (Dates in this case)
        times: List[Tupes] containg start and end time of each utterance.
        text: List[String] containing texts of utterance
    """
    times = []
    text = []
    ids = []
    with open(path, 'r') as f:
        data = f.readlines()
        for d in data:
            d = d.split('|')
            t1, t2 = float(d[1]), float(d[2])
            text.append(d[3])
            ids.append(d[0])
            times.append((t1,t2))

    return ids, times, text    

def create_metafile(path, save_dir):
    """
    Converts the lab file into a format which can be directly read by the TTS system.
    Input:
        path: path to the lab file.
        save_dir: folder in which the metafile is stored.
    Output:
        None
        Save the metafile.txt in the specified folder.
    """
    with open(save_dir + '/' +"metadata.txt", 'w') as s:
        with open(path, 'r') as f:
            data = f.readlines()
            for i,d in enumerate(data):
                d = d.split()
                s.write(str(i)+"|"+" ".join(d[2:])+"\n")

    print("Completed writing metafile")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--timings_file', type=str, default='', required=True)
    parser.add_argument('--data_path', type=str, default='', required=True)
    parser.add_argument('--save_dir', type=str, default='', required=True)
    args = parser.parse_args()

    ids, times, text = read_times(args.timings_file)
    pid = None
    csnip_times = []
    csnip_text = []
    for id_, t, tx in zip(ids, times, text):
        if id_ != pid:
            if len(csnip_times):
                crop(csnip_times, csnip_text, audio_path, audio_save_dir, video_path, video_save_dir)
            pid = id_
            audio_path = os.path.join(args.data_path, pid, 'audio.wav')
            audio_save_dir = os.path.join(args.save_dir, pid)
            video_path = os.path.join(args.data_path, pid, 'video.mp4')
            video_save_dir = os.path.join(args.save_dir, pid)
            # Make dirs
            if not os.path.exists(video_save_dir):
                os.makedirs(video_save_dir)
            if not os.path.exists(audio_save_dir):
                os.makedirs(audio_save_dir)
            csnip_times = []
            csnip_text = []
        else:
            csnip_times.append(t)
            csnip_text.append(tx)

if __name__ == '__main__':
    main()
