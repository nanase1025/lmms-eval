#!/usr/bin/env bash

ATTN_IMPLEMENTATION="flash_attention_2"

CUDA_VISIBLE_DEVICES=7 python -m lmms_eval \
    --model llava_hf \
    --model_args "pretrained=/home/deeplearning/data/data2/wzc/VolInterp/methods/checkpoints/llava-1.5-7b-hf,attn_implementation=${ATTN_IMPLEMENTATION},device_map=auto" \
    --tasks pope \
    --batch_size 1 \
    --output_path ./logs/llava_hf_pope \
    --log_samples \
    --force_simple
