# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# 检查环境变量 ZHIYUHE 是否设置
if [ -z "$ZHIYUHE" ]; then
    echo "Error: Environment variable ZHIYUHE is not set."
    exit 1
fi

# llama3.1-8b-instruct
# llama3-8b-instruct-262k
# qwen2.5-7b-instruct
model_name=$1
TRI_LAYER_NUM=$2

# 根据 model_name 选择对应的列表
if [[ "$model_name" == *"llama3.1-8b-instruct"* ]]; then
    list=(31 30 29 24 25 28 22 23 21 26 27 20 18 17 19 16 3 0 15 14 12 1 4 6 9 5 2 10 7 11 13 8)
elif [[ "$model_name" == *"llama3-8b-instruct-262k"* ]]; then
    list=(31 30 29 28 23 24 22 25 21 26 18 20 16 27 17 19 12 3 0 15 1 14 6 5 9 4 13 7 2 10 11 8)
elif [[ "$model_name" == *"qwen2.5-7b-instruct"* ]]; then
    list=(27 26 25 24 23 22 6 21 10 3 1 7 18 19 9 15 20 17 13 4 2 8 12 16 14 11 5 0)
else
    echo "Error: Unsupported MODEL_NAME: $model_name"
    exit 1
fi


# 取前 TRI_LAYER_NUM 个
selected=("${list[@]:0:$TRI_LAYER_NUM}")

# 拼接成 "[xx,xx,xx]" 的字符串
tri_layer_idx_list_str="["
for i in "${!selected[@]}"; do
    if [ $i -gt 0 ]; then
        tri_layer_idx_list_str+=","
    fi
    tri_layer_idx_list_str+="${selected[$i]}"
done
tri_layer_idx_list_str+="]"

# 输出结果
echo "$tri_layer_idx_list_str"

python pred.py --model $1 --e \
    --attn_type "minference_mix_per_layer" \
    --attn_kwargs  "{\"gamma\": 0.95, \"tri_layer_idx_list\": ${tri_layer_idx_list_str}, \"n_local\": 512, \"n_init\": 8, \"last_n\": 512}" \
    --method_name minference_mix_0.95_by_gradient_tri_num_${TRI_LAYER_NUM}

mkdir -p $ZHIYUHE/250919_longbench_nlast512
cp -r pred_e/* $ZHIYUHE/250919_longbench_nlast512
