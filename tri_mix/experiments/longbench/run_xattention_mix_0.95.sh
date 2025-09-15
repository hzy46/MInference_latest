# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

model_name=$1

python pred.py --model $1 --e \
    --attn_type "xattention_mix" \
    --attn_kwargs "{\"threshold\": 0.95, \"n_local\": 512, \"n_init\": 8, \"n_last\": 128, \"starting_layer\": 16}" \
    --method_name xattention_mix_0.95
