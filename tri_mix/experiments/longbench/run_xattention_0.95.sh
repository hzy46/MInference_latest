# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

model_name=$1

python pred.py --model $1 --e \
    --attn_type "xattention" \
    --attn_kwargs  "{\"threshold\": 0.95}" \
    --method_name xattention_0.95
