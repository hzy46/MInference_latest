# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]
mkdir -p $ZHIYUHE/250915_longbench

# bash run_xattention_mix_0.95.sh llama3.1-8b-instruct
# cp -r pred_e/* $ZHIYUHE/250915_longbench
bash run_xattention_mix_0.95.sh llama3-8b-instruct-262k
cp -r pred_e/* $ZHIYUHE/250915_longbench
