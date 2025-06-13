# TriangleMix: A Lossless and Efficient Attention Pattern for Long Context Prefilling

We propose `TriangleMix`, a training-free static attention pattern. TriangleMix employs dense attention in shallow layers and switches to a triangle-shaped sparse pattern in deeper layers. Extensive experimental results show that TriangleMix achieves 1.4x to 2x speedup across input lengths ranging from 4K to 128K without sacrificing model accuracy. Furthermore, `TriangleMix` can be seamlessly combined with dynamic sparsity methods, resulting in additional acceleration and highlighting its potential for accelerating LLM inference.

## Quick Start

Make sure you have installed the latest `minference`:

```bash
conda create -n minference python=3.11
conda activate minference
pip install "transformers[torch]"
pip install flash-attn --no-build-isolation
git clone https://github.com/microsoft/MInference.git ~/MInference
cd ~/MInference
pip install -e .
```

Then you can use the `tri_mix` methods:

```diff
from transformers import AutoModelForCausalLM, AutoTokenizer
+ from minference import MInference

model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)

+minference_patch = MInference(
+    attn_type="tri_mix",
+    model_name=model_name,
+    attn_kwargs={"last_n": 128, "starting_layer": 16, "n_local": 512, "n_init": 8},
+)
+model = minference_patch(model)

prompt = "your prompt here"
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
output = model.generate(**inputs, do_sample=False, max_new_tokens=50)
```

## Reproduce Ruler Performance

First, set up the ruler test environment and change directory to `<minference>/experiments/ruler/`.

Run dense attention:

```bash
mkdir -p ./results/dense
bash run_dense.sh meta-llama/Llama-3.1-8B-Instruct minference ./results/dense
```

Run `TriangleMix`:

```bash
mkdir -p ./results/tri_mix
bash run_tri_mix.sh meta-llama/Llama-3.1-8B-Instruct minference ./results/tri_mix 16
```

Results on 128K context length (Minimal drop from 77.6 to 77.3):

| method      | AVG  | NI.SG1 | NI.SG2 | NI.SG3 | NI.MK1 | NI.MK2 | NI.MK3 | NI.MV  | NI.MQ  | VT   | CWE  | FWE   | QA1  | QA2  |
|-------------|------|--------|--------|--------|--------|--------|--------|--------|--------|------|------|-------|------|------|
| Dense       | 77.6 | 100.0  | 96.0   | 100.0  | 95.0   | 91.0   | 62.0   | 97.25  | 98.5   | 90.0 | 2.2  | 57.33 | 77.0 | 43.0 |
| TriangleMix | 77.3 | 100.0  | 96.0   | 100.0  | 95.0   | 91.0   | 63.0   | 97.75  | 98.0   | 93.2 | 0.0  | 50.67 | 77.0 | 43.0 |


## Reproduce Efficiency Metrics

We provide a speed test script `speed_test.py`. This script measures the TTFT (time-to-first-token).

```bash
# test
python speed_test.py --method dense --model_name meta-llama/Llama-3.1-8B-Instruct
python speed_test.py --method tri_mix --model_name meta-llama/Llama-3.1-8B-Instruct
```

TTFT on A100 80GB:

method,4K,8K,16K,32K,64K,128K
Dense,
TriangleMix

**pic here**

Note: `MInference` can be faster with latest kernel updates. Here the performance metrics are based on the old implementation.