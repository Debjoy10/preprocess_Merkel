bash download_model.sh

python  crop_from_timings.py --timings_file ../Merkel-Podcast-Corpus/timings.txt --data_path ../Merkel-Podcast-Corpus/corpus --save_dir temp/

find temp/ -type f -regex ".*\.mp4" > temp/mp4_list.txt

python run_process_faces.py --file temp/mp4_list.txt

python run_process_lips.py --file temp/mp4_list.txt

python scene_writer.py --lip_videos_dir lip_out/pycrop/ --face_videos_dir out/pycrop/ --wavs_dir '' --text_dir temp/ --out_dir ../Merkel_Single_Speaker

bash files/to_run.sh

python utils/metadata_writer.py --dataset_dir ../Merkel_Single_Speaker/

rm -rf out
rm -rf lip_out
rm -rf temp