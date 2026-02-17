#!/bin/bash
# Copyright (c) 2024-2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# 检查环境变量 ZHIYUHE 是否设置
if [ -z "$ZHIYUHE" ]; then
    echo "Error: Environment variable ZHIYUHE is not set."
    exit 1
fi

REMOTE_SAVE_DIR=$ZHIYUHE/250916_ruler/flexprefill_mix_per_layer/
mkdir -p $REMOTE_SAVE_DIR
MODEL_FRAMEWORK=minference
ROOT_DIR=results_flexprefill_mix_per_layer


MODEL_NAME=$1
TRI_LAYER_NUM=$2
LENGTH_TYPE=$3

if [ "$3" = "long" ]; then
    SEQ_LENGTHS=(
        32768
        65536
        131072
    )
elif [ "$3" = "short" ]; then
    SEQ_LENGTHS=(
        4096
        8192
        16384
    )
elif [ "$3" = "all" ]; then
    SEQ_LENGTHS=(
        4096
        8192
        16384
        32768
        65536
        131072
    )
else
    echo "Error: third argument must be 'long', 'short', or 'all'"
    exit 1
fi


# 根据 model_name 选择对应的列表
if [[ "$MODEL_NAME" == *"Llama-3.1-8B-Instruct"* ]]; then
    list=(31 30 29 24 25 28 22 23 21 26 27 20 18 17 19 16 3 0 15 14 12 1 4 6 9 5 2 10 7 11 13 8)
elif [[ "$MODEL_NAME" == *"Llama-3-8B-Instruct-262k"* ]]; then
    list=(31 30 29 28 23 24 22 25 21 26 18 20 16 27 17 19 12 3 0 15 1 14 6 5 9 4 13 7 2 10 11 8)
elif [[ "$MODEL_NAME" == *"Qwen2.5-7B-Instruct"* ]]; then
    list=(27 26 25 24 23 22 6 21 10 3 1 7 18 19 9 15 20 17 13 4 2 8 12 16 14 11 5 0)
else
    echo "Error: Unsupported MODEL_NAME: $MODEL_NAME"
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


export TOKENIZERS_PARALLELISM=false
RULER_PATH=$(dirname $0)
python -c "import nltk; nltk.download('punkt')"

NUM_SAMPLES=100

TASKS=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)

# Experiment Setup
TEMPERATURE="0.0"
TOP_P="1.0"
TOP_K="32"

# The model
BENCHMARK="synthetic"
MODEL_TEMPLATE_TYPE="base"


# MInference
STARTING_LAYER=-1
KV_CACHE_CPU="false"
USE_SNAPKV="false"
TRUST_REMOTE_CODE="true"

if [ "${MODEL_FRAMEWORK}" == "minference" ]; then
    MINFERENCE_PARAMS="--starting_layer ${STARTING_LAYER}"

    if [ -n "${CONFIG_PATH}" ]; then
        MINFERENCE_PARAMS="${MINFERENCE_PARAMS} --config_path ${CONFIG_PATH}"
    fi

    if [ "${USE_SNAPKV}" == "true" ]; then
        MINFERENCE_PARAMS="${MINFERENCE_PARAMS} --use_snapkv"
    fi

    echo "MInference enabled with params: ${MINFERENCE_PARAMS}"
fi

if [ "${TRUST_REMOTE_CODE}" == "true" ]; then
    EXTRA_PARAMS="${EXTRA_PARAMS} --trust_remote_code"
fi

if [ "${KV_CACHE_CPU}" == "true" ]; then
    EXTRA_PARAMS="${EXTRA_PARAMS} --kv_cache_cpu --kv_cache_cpu_device cpu"
fi

# Gpu and output path
GPUS="1" # GPU size for tensor_parallel.

for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do

    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}_${MODEL_FRAMEWORK}_by_gradient_tri_num_${TRI_LAYER_NUM}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}

    for TASK in "${TASKS[@]}"; do
        python ${RULER_PATH}/data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${MODEL_NAME} \
            --tokenizer_type "hf" \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB}

        python ${RULER_PATH}/pred/call_api.py \
            --data_dir ${DATA_DIR} \
            --save_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --server_type ${MODEL_FRAMEWORK} \
            --attn_type flexprefill_mix_per_layer \
            --attn_kwargs "{\"tri_layer_idx_list\": $tri_layer_idx_list_str, \"gamma\": 0.95, \"n_local\": 512, \"n_init\": 8, \"last_n\": 128}" \
            --model_name_or_path ${MODEL_NAME} \
            --temperature ${TEMPERATURE} \
            --top_k ${TOP_K} \
            --top_p ${TOP_P} \
            ${MINFERENCE_PARAMS} \
            ${EXTRA_PARAMS} \
            ${STOP_WORDS}
    done

    python ${RULER_PATH}/eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK}
done


cp -r ${ROOT_DIR}/${MODEL_NAME}_${MODEL_FRAMEWORK}_by_gradient_tri_num_${TRI_LAYER_NUM} $REMOTE_SAVE_DIR
