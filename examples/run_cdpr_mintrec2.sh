#!/usr/bin bash

for dataset in 'MIntRec2.0'
do
    for seed in 0 1 2 3 4
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'cdpr' \
        --method 'CDPR' \
        --tune \
        --train \
        --save_results \
        --seed $seed \
        --gpu_id '0' \
        --video_feats_path 'swin_roi.pkl' \
        --audio_feats_path 'wavlm_feats.pkl' \
        --text_backbone 'bert-large-uncased' \
        --config_file_name 'cdpr_mintrec2' \
        --results_file_name 'results_cdpr_mintrec2.csv'
    done
done