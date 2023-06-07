OUTPUT_DIR=
python3 run_sum.py \
    --model_path tanay/efactsum-pegasus-xsum \
    --source ../outputs/xsum.test.source.txt \
    --output_dir $OUTPUT_DIR \
    --batch_size 2 \
    --max_length 512 \
    --gen_max_len 62 \
    --gen_min_len 11 \
    --num_beams 8 \
    --length_penalty 0.6