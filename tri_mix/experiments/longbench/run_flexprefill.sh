# llama3.1-8b-instruct
# llama3-8b-instruct-262k
# qwen2.5-7b-instruct
model_name=$1

python pred.py --model $1 --e \
    --attn_type "flexprefill" \
    --attn_kwargs  "{\"gamma\": 0.95}" \
    --method_name flexprefill_0.95