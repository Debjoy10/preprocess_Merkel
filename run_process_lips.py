import numpy as np
from shutil import rmtree
import argparse
import glob
import os
import cv2
import time
import subprocess

import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal

import pickle

from detectors import S3FD

# ========== ========== ========== ==========
# # IOU FUNCTION
# ========== ========== ========== ==========

def bb_intersection_over_union(boxA, boxB):

  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interArea = max(0, xB - xA) * max(0, yB - yA)

  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

  iou = interArea / float(boxAArea + boxBArea - interArea)

  return iou

# ========== ========== ========== ==========
# # FACE TRACKING
# ========== ========== ========== ==========

def track_shot(opt,scenefaces,scene_number):

  iouThres  = 0.5     # Minimum IOU between consecutive face detections
  tracks    = []

  while True:
    track     = []
    for framefaces in scenefaces:
      for face in framefaces:
        if track == []:
          track.append(face)
          framefaces.remove(face)
        elif face['frame'] - track[-1]['frame'] <= opt.num_failed_det:
          iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
          if iou > iouThres:
            track.append(face)
            framefaces.remove(face)
            continue
        else:
          break

    if track == []:
      break
    elif len(track) > opt.min_track:

      framenum    = np.array([ f['frame'] for f in track ])
      bboxes      = np.array([np.array(f['bbox']) for f in track])

      frame_i   = np.arange(framenum[0],framenum[-1]+1)

      bboxes_i    = []
      for ij in range(0,4):
        interpfn  = interp1d(framenum, bboxes[:,ij])
        bboxes_i.append(interpfn(frame_i))
      bboxes_i  = np.stack(bboxes_i, axis=1)

      if max(np.mean(bboxes_i[:,2]-bboxes_i[:,0]), np.mean(bboxes_i[:,3]-bboxes_i[:,1])) > opt.min_face_size:
        tracks.append({'scene_number': str(scene_number), 'frame':frame_i,'bbox':bboxes_i})

  return tracks

# ========== ========== ========== ==========
# # VIDEO CROP AND SAVE
# ========== ========== ========== ==========

def crop_video(opt,track,cropfile):

  flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
  flist.sort()

  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  vOut = cv2.VideoWriter(cropfile+'t.avi', fourcc, opt.frame_rate, (96,48))

  dets = {'x':[], 'y':[], 's':[]}

  for det in track['bbox']:

    dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2)
    dets['y'].append((det[1]+det[3])/2) # crop center x
    dets['x'].append((det[0]+det[2])/2) # crop center y

  # Smooth detections
  dets['s'] = signal.medfilt(dets['s'],kernel_size=13)
  dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
  dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

  for fidx, frame in enumerate(track['frame']):

    cs  = opt.crop_scale

    bs  = dets['s'][fidx]   # Detection box size
    bsi = int(bs*(1+2*cs))  # Pad videos by this amount

    image = cv2.imread(flist[frame])

    frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
    my  = dets['y'][fidx]+bsi  # BBox center Y
    mx  = dets['x'][fidx]+bsi  # BBox center X
    # cropping the lips
    diff = (int(mx+bs*(1+cs)) - int(mx-bs*(1+cs)))//5
    face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs))+diff:int(mx+bs*(1+cs))-diff]
    face = cv2.resize(face,(96,96))
    face = face[face.shape[0]//2:, :]

    vOut.write(face)

  audiotmp    = os.path.join(opt.tmp_dir,opt.reference,'audio.wav')
  audiostart  = (track['frame'][0])/opt.frame_rate
  audioend    = (track['frame'][-1]+1)/opt.frame_rate

  vOut.release()

  # ========== CROP AUDIO FILE ==========

  command = ("ffmpeg -y -i %s -ss %.3f -to %.3f %s" % (os.path.join(opt.avi_dir,opt.reference,'audio.wav'),audiostart,audioend,audiotmp))
  output = subprocess.call(command, shell=True, stdout=None)

  if output != 0:
    pdb.set_trace()

  sample_rate, audio = wavfile.read(audiotmp)

  # ========== COMBINE AUDIO AND VIDEO FILES ==========

  command = ("ffmpeg -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (cropfile,audiotmp,cropfile))
  output = subprocess.call(command, shell=True, stdout=None)

  if output != 0:
    pdb.set_trace()

  print('Written %s'%cropfile)

  os.remove(cropfile+'t.avi')

  print('Mean pos: x %.2f y %.2f s %.2f'%(np.mean(dets['x']),np.mean(dets['y']),np.mean(dets['s'])))

  return {'track':track, 'proc_track':dets}

# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========

def inference_video(opt):

  DET = S3FD(device='cuda')

  flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
  flist.sort()

  dets = []

  for fidx, fname in enumerate(flist):

    start_time = time.time()

    image = cv2.imread(fname)

    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])

    dets.append([]);
    for bbox in bboxes:
      dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})

    elapsed_time = time.time() - start_time

    print('%s-%05d; %d dets; %.2f Hz' % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),fidx,len(dets[-1]),(1/elapsed_time)))

  savepath = os.path.join(opt.work_dir,opt.reference,'faces.pckl')

  with open(savepath, 'wb') as fil:
    pickle.dump(dets, fil)

  return dets

# ========== ========== ========== ==========
# # SCENE DETECTION
# ========== ========== ========== ==========

def scene_detect(opt):

  video_manager = VideoManager([os.path.join(opt.avi_dir,opt.reference,'video.avi')])
  stats_manager = StatsManager()
  scene_manager = SceneManager(stats_manager)
  # Add ContentDetector algorithm (constructor takes detector options like threshold).
  scene_manager.add_detector(ContentDetector())
  base_timecode = video_manager.get_base_timecode()

  video_manager.set_downscale_factor()

  video_manager.start()

  scene_manager.detect_scenes(frame_source=video_manager)

  scene_list = scene_manager.get_scene_list(base_timecode)

  savepath = os.path.join(opt.work_dir,opt.reference,'scene.pckl')

  if scene_list == []:
    scene_list = [(video_manager.get_base_timecode(),video_manager.get_current_timecode())]

  with open(savepath, 'wb') as fil:
    pickle.dump(scene_list, fil)

  print('%s - scenes detected %d'%(os.path.join(opt.avi_dir,opt.reference,'video.avi'),len(scene_list)))

  return scene_list


# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description = "FaceTracker");
parser.add_argument('--data_dir',       type=str, default='./lip_out', help='Output direcotry')
parser.add_argument('--facedet_scale',  type=float, default=0.25, help='Scale factor for face detection')
parser.add_argument('--crop_scale',     type=float, default=0.08, help='Scale bounding box')
parser.add_argument('--min_track',      type=int, default=5,  help='Minimum facetrack duration')
parser.add_argument('--frame_rate',     type=int, default=25,   help='Frame rate')
parser.add_argument('--num_failed_det', type=int, default=25,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--min_face_size',  type=int, default=50,  help='Minimum face size in pixels')
parser.add_argument('--videofile',      type=str, default='',   help='Input video file')
parser.add_argument('--reference',      type=str, default='',   help='Video reference')
parser.add_argument('--file',      type=str, default='',   help='Input file with list of videos with their full path')
parser.add_argument('--folder', type=str, default='', help='Path to the folder where all the fide files to be processed exist')
parser.add_argument('--log_file', type=str, default='logs.txt', help='File where details of processing will be stored')
opt = parser.parse_args();

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))
setattr(opt,'frames_dir',os.path.join(opt.data_dir,'pyframes'))
opt.log_file = os.path.join(opt.data_dir,opt.log_file)

if len(opt.videofile) > 0 and len(opt.reference) > 0:
    videofiles = [opt.videofile]
elif len(opt.file) > 0:
    videofiles = []
    with open(opt.file, 'r') as f:
        data = f.readlines()
        for d in data:
            d = d.strip()
            videofiles.append(d)
elif len(opt.folder) > 0:
    videofiles = [os.path.join(opt.folder, f) for f in os.listdir(opt.folder)]

if not os.path.exists(opt.data_dir):
    os.makedirs(opt.data_dir)
f = open(opt.log_file, 'w')
    
for videofile in videofiles:

  opt.videofile = videofile
  if len(opt.reference) == 0:
    opt.reference = videofile.split('/')[-2] + '_' + videofile.split('/')[-1].split('.')[0] # name of the videofile

  print(opt.reference, opt.videofile)

  try:
    # ========== DELETE EXISTING DIRECTORIES ==========
    if os.path.exists(os.path.join(opt.work_dir,opt.reference)):
      rmtree(os.path.join(opt.work_dir,opt.reference))

    if os.path.exists(os.path.join(opt.crop_dir,opt.reference)):
      rmtree(os.path.join(opt.crop_dir,opt.reference))

    if os.path.exists(os.path.join(opt.avi_dir,opt.reference)):
      rmtree(os.path.join(opt.avi_dir,opt.reference))

    if os.path.exists(os.path.join(opt.frames_dir,opt.reference)):
      rmtree(os.path.join(opt.frames_dir,opt.reference))

    if os.path.exists(os.path.join(opt.tmp_dir,opt.reference)):
      rmtree(os.path.join(opt.tmp_dir,opt.reference))

# ========== MAKE NEW DIRECTORIES ==========

    os.makedirs(os.path.join(opt.work_dir,opt.reference))
    os.makedirs(os.path.join(opt.crop_dir,opt.reference))
    os.makedirs(os.path.join(opt.avi_dir,opt.reference))
    os.makedirs(os.path.join(opt.frames_dir,opt.reference))
    os.makedirs(os.path.join(opt.tmp_dir,opt.reference))

# ========== CONVERT VIDEO AND EXTRACT FRAMES ==========
    command = ("ffmpeg -y -i %s -qscale:v 2 -async 1 -r 25 %s" % (opt.videofile,os.path.join(opt.avi_dir,opt.reference,'video.avi')))
    output = subprocess.call(command, shell=True, stdout=None)

    command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),os.path.join(opt.frames_dir,opt.reference,'%06d.jpg')))
    output = subprocess.call(command, shell=True, stdout=None)

    command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),os.path.join(opt.avi_dir,opt.reference,'audio.wav')))
    output = subprocess.call(command, shell=True, stdout=None)

# ========== FACE DETECTION ==========

    faces = inference_video(opt)

# ========== SCENE DETECTION ==========

    scene = scene_detect(opt)

# ========== FACE TRACKING ==========

    alltracks = []
    vidtracks = []
    shot_number = 0


    for shot in scene:

      if shot[1].frame_num - shot[0].frame_num >= opt.min_track :
        shot_number += 1
        alltracks.extend(track_shot(opt,faces[shot[0].frame_num:shot[1].frame_num], shot_number))

# ========== FACE TRACK CROP ==========

    for ii, track in enumerate(alltracks):
      print(track['scene_number'])
      vidtracks.append(crop_video(opt,track,os.path.join(opt.crop_dir,opt.reference, track['scene_number']+'-'+'%05d'%ii)))

# ========== SAVE RESULTS ==========

    savepath = os.path.join(opt.work_dir,opt.reference,'tracks.pckl')

    with open(savepath, 'wb') as fil:
      pickle.dump(vidtracks, fil)

    num = len(os.listdir(os.path.join(opt.crop_dir,opt.reference)))
    f.write("{} {}\n".format(opt.videofile, num))

    rmtree(os.path.join(opt.tmp_dir,opt.reference))
    opt.reference = ''
  except Exception as e:
    print(e)
    f.write("{} {}\n".format(opt.videofile, "Processing failed"))
    opt.reference = '' 

f.close()
print("Finished Processing")
