#!/usr/bin/env bash

ATTN_IMPLEMENTATION="flash_attention_2"
DOMINANT_RATIO="0.30"
CONTEXTUAL_RATIO="0.03"

CUDA_VISIBLE_DEVICES=6 python -m lmms_eval \
    --model llava_hf_visionzip \
    --model_args "pretrained=/home/deeplearning/data/data2/wzc/VolInterp/methods/checkpoints/llava-1.5-7b-hf,attn_implementation=${ATTN_IMPLEMENTATION},dominant_ratio=${DOMINANT_RATIO},contextual_ratio=${CONTEXTUAL_RATIO},device_map=auto" \
    --tasks pope \
    --batch_size 1 \
    --output_path ./logs/llava_hf_pope_vzip \
    --log_samples \
    --force_simple
