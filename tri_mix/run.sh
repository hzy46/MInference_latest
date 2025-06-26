# python speed_test.py --model_name "meta-llama/Llama-3.1-8B-Instruct" --method dense
# python speed_test.py --model_name "meta-llama/Llama-3.1-8B-Instruct" --method tri_mix
# python speed_test.py --model_name "meta-llama/Llama-3.1-8B-Instruct" --method flexprefill
# python speed_test.py --model_name "meta-llama/Llama-3.1-8B-Instruct" --method minference
# python speed_test.py --model_name "meta-llama/Llama-3.1-8B-Instruct" --method minference_mix
# python speed_test.py --model_name "meta-llama/Llama-3.1-8B-Instruct" --method flexprefill_mix

# python speed_test.py --model_name "gradientai/Llama-3-8B-Instruct-262k" --method dense
# python speed_test.py --model_name "gradientai/Llama-3-8B-Instruct-262k" --method tri_mix
# python speed_test.py --model_name "gradientai/Llama-3-8B-Instruct-262k" --method flexprefill
# python speed_test.py --model_name "gradientai/Llama-3-8B-Instruct-262k" --method minference
# python speed_test.py --model_name "gradientai/Llama-3-8B-Instruct-262k" --method minference_mix
# python speed_test.py --model_name "gradientai/Llama-3-8B-Instruct-262k" --method flexprefill_mix

# python speed_test.py --model_name "Qwen/Qwen2.5-7B-Instruct" --method dense
# python speed_test.py --model_name "Qwen/Qwen2.5-7B-Instruct" --method tri_mix
# python speed_test.py --model_name "Qwen/Qwen2.5-7B-Instruct" --method flexprefill
# python speed_test.py --model_name "Qwen/Qwen2.5-7B-Instruct" --method minference
# python speed_test.py --model_name "Qwen/Qwen2.5-7B-Instruct" --method minference_mix
# python speed_test.py --model_name "Qwen/Qwen2.5-7B-Instruct" --method flexprefill_mix



python speed_test.py --model_name "meta-llama/Llama-3.1-8B-Instruct" --method dense
python speed_test.py --model_name "meta-llama/Llama-3.1-8B-Instruct" --method tri_mix
python speed_test.py --model_name "meta-llama/Llama-3.1-8B-Instruct" --method flexprefill --gamma 0.95
python speed_test.py --model_name "meta-llama/Llama-3.1-8B-Instruct" --method flexprefill --gamma 0.90
python speed_test.py --model_name "meta-llama/Llama-3.1-8B-Instruct" --method minference
python speed_test.py --model_name "meta-llama/Llama-3.1-8B-Instruct" --method minference_mix
python speed_test.py --model_name "meta-llama/Llama-3.1-8B-Instruct" --method flexprefill_mix --gamma 0.95
python speed_test.py --model_name "meta-llama/Llama-3.1-8B-Instruct" --method flexprefill_mix --gamma 0.90