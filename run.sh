bash download_model.sh

python  crop_from_timings.py --timings_file ../Merkel-Podcast-Corpus/timings.txt --data_path ../Merkel-Podcast-Corpus/corpus --save_dir temp/

find temp/ -type f -regex ".*\.mp4" > temp/mp4_list.txt

python run_process_faces.py --file temp/mp4_list.txt

python run_process_lips.py --file temp/mp4_list.txt

cd TalkNet_ASD

find ../out/ -type f -regex ".*pycrop.*\.avi" > avi_list.txt

mkdir ../TSout/

python run_TalkNet.py --videolist avi_list.txt --output_dir ../TSout/

cd ..

find out/ -type f -regex ".*pycrop.*\.avi" > avi_list.txt

python scene_scorer.py --videolist avi_list.txt --talknet_dir TSout/

python scene_extractor.py --scores_file_path files/facetalk_scores.csv

python scene_writer.py --lip_videos_dir lip_out/pycrop/ --face_videos_dir out/pycrop/ --wavs_dir '' --text_dir temp/ --out_dir ../Merkel_Single_Speaker

bash files/to_run.sh

python utils/metadata_writer.py --dataset_dir ../Merkel_Single_Speaker/

rm -rf out
rm -rf lip_out
rm -rf TSout
rm -rf temp
rm avi_list.txt
