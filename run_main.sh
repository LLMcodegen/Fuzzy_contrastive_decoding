cd $dir

python3 main.py \
    --use_auth_token \
    --mean_num 0.0050 \
    --filter_num 0.666667 \
    --num_beams 1 \
    --src_weights 1.0 -0.7 \
    --model $model_path \
    --tasks $benchmark \
    --temperature 0.8 \
    --n_samples 15 \
    --top_p 0.95 \
    --allow_code_execution \
    --metric_output_path $metric_output_path \
    --save_generations_path $save_generations_path