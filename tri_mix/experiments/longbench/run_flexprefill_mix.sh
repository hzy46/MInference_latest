# llama3.1-8b-instruct
# llama3-8b-instruct-262k
# qwen2.5-7b-instruct
model_name=$1

if [[ "$model_name" == "llama3.1-8b-instruct" || "$model_name" == "llama3-8b-instruct-262k" ]]; then
  starting_layer=16
elif [[ "$model_name" == "qwen2.5-7b-instruct" ]]; then
  starting_layer=20
else
  echo "Error: unsupported model '$model_name'" >&2
  exit 1
fi

python pred.py --model $1 --e \
    --attn_type "flexprefill_mix" \
    --attn_kwargs  "{\"gamma\": 0.95, \"starting_layer\": ${starting_layer}, \"n_local\": 512, \"n_init\": 8, \"last_n\": 128}" \
    --method_name flexprefill_mix_0.95