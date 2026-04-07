ATTN_IMPLEMENTATION="flash_attention_2"

CUDA_VISIBLE_DEVICES=7 python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args "pretrained=/home/deeplearning/data/data2/huggingface_model_share/Qwen2.5-VL-3B-Instruct,attn_implementation=${ATTN_IMPLEMENTATION}" \
    --tasks pope \
    --batch_size 1 \
    --force_simple
