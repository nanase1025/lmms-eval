#!/usr/bin/env bash

ATTN_IMPLEMENTATION="flash_attention_2"
DOMINANT_RATIO="0.30"
CONTEXTUAL_RATIO="0.03"

CUDA_VISIBLE_DEVICES=6 python -m lmms_eval \
    --model qwen2_5_vl_visionzip \
    --model_args "pretrained=/home/deeplearning/data/data2/huggingface_model_share/Qwen2.5-VL-3B-Instruct,attn_implementation=${ATTN_IMPLEMENTATION},dominant_ratio=${DOMINANT_RATIO},contextual_ratio=${CONTEXTUAL_RATIO}" \
    --tasks pope \
    --batch_size 1 \
    --output_path ./logs/qwen2_5_vl_pope_vzip_save_33 \
    --log_samples
