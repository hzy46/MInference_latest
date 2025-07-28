# TriangleMix: A Lossless and Efficient Attention Pattern for Long Context Prefilling

We propose `TriangleMix`, a training-free static attention pattern for efficient long context prefilling. 



TriangleMix employs dense attention in shallow layers and switches to a triangle-shaped sparse pattern in deeper layers. 



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

First, setup ruler environments. See [setup_ruler.sh](./setup_ruler.sh) for details.

Then, change directory to `<minference>/tri_mix/ruler/`.

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

| Method      | AVG  | NI.SG1 | NI.SG2 | NI.SG3 | NI.MK1 | NI.MK2 | NI.MK3 | NI.MV  | NI.MQ  | VT   | CWE  | FWE   | QA1  | QA2  |
|-------------|------|--------|--------|--------|--------|--------|--------|--------|--------|------|------|-------|------|------|
| Dense       | 77.6 | 100.0  | 96.0   | 100.0  | 95.0   | 91.0   | 62.0   | 97.25  | 98.5   | 90.0 | 2.2  | 57.33 | 77.0 | 43.0 |
| TriangleMix | 77.3 | 100.0  | 96.0   | 100.0  | 95.0   | 91.0   | 63.0   | 97.75  | 98.0   | 93.2 | 0.0  | 50.67 | 77.0 | 43.0 |


## Reproduce Efficiency Metrics

We provide a speed test script `speed_test.py`. This script measures the TTFT (time-to-first-token).

```bash
python speed_test.py --method dense --model_name meta-llama/Llama-3.1-8B-Instruct
python speed_test.py --method tri_mix --model_name meta-llama/Llama-3.1-8B-Instruct
python speed_test.py --method tri_mix_minfernece --model_name meta-llama/Llama-3.1-8B-Instruct
```

TTFT in seconds on A100 80GB:

| Method             | 32K                 | 48K                 | 64K                 | 80K                 | 96K                 | 112K                | 128K                |
|--------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| Dense              | 4.1                 | 7.3                 | 11.2                | 15.9                | 21.3                | 27.5                | 34.5                |
| MInference         | 5.5 (<span style="color:red">+34%</span>)  | 7.8 (<span style="color:red">+7%</span>)   | 10.1 (<span style="color:green">-10%</span>) | 12.3 (<span style="color:green">-23%</span>) | 13.4 (<span style="color:green">-37%</span>) | 15.9 (<span style="color:green">-42%</span>) | 18.0 (<span style="color:green">-48%</span>) |
| TriangleMix        | 3.6 (<span style="color:green">-12%</span>) | 5.9 (<span style="color:green">-19%</span>) | 8.6 (<span style="color:green">-23%</span>)  | 11.7 (<span style="color:green">-26%</span>) | 15.2 (<span style="color:green">-29%</span>) | 19.1 (<span style="color:green">-31%</span>) | 23.4 (<span style="color:green">-32%</span>) |
| Ours + MInference  | 4.2 (<span style="color:red">+2%</span>)    | 6.0 (<span style="color:green">-18%</span>) | 7.7 (<span style="color:green">-31%</span>)  | 9.5 (<span style="color:green">-40%</span>)  | 10.9 (<span style="color:green">-49%</span>) | 12.7 (<span style="color:green">-54%</span>) | 14.5 (<span style="color:green">-58%</span>) |



Note: `MInference` can be faster with latest kernel updates. Here the performance metrics are based on the old implementation.