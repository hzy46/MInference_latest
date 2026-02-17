#!/bin/bash
# Copyright (c) 2024-2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

export TOKENIZERS_PARALLELISM=false
RULER_PATH=$(dirname $0)
python -c "import nltk; nltk.download('punkt')"

MODEL_NAME=$1
THRESHOLD=$2 # 0.28, 0.5


# 检查环境变量 ZHIYUHE 是否设置
if [ -z "$ZHIYUHE" ]; then
    echo "Error: Environment variable ZHIYUHE is not set."
    exit 1
fi

REMOTE_SAVE_DIR=$ZHIYUHE/260217_ruler/
mkdir -p $REMOTE_SAVE_DIR


SEQ_LENGTHS=(
    # 4096
    # 8192
    # 16384
    32768
    65536
    131072
)

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
NUM_SAMPLES=100
TEMPERATURE="0.0"
TOP_P="1.0"
TOP_K="32"

# The model
BENCHMARK="synthetic"
MODEL_TEMPLATE_TYPE="base"
MODEL_FRAMEWORK=kvpress

# MInference
STARTING_LAYER=-1
KV_CACHE_CPU="false"
USE_SNAPKV="false"
TRUST_REMOTE_CODE="true"


# Gpu and output path
GPUS="1" # GPU size for tensor_parallel.
ROOT_DIR=results # the path that stores generated task samples and model predictions.

for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do

    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}_${MODEL_FRAMEWORK}_duo_{$THRESHOLD}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
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
            --press_kwargs "{\"head_compression_ratio\": ${THRESHOLD}}" \
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


cp -r ${ROOT_DIR}/${MODEL_NAME}_${MODEL_FRAMEWORK}_duo_{$THRESHOLD} $REMOTE_SAVE_DIR
