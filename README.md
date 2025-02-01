<div align="center">
  <img src="visuals/logos/OLMoE_logo.png" width="200" />
  <h1>s1: Simple test-time scaling</h1>
  <p>Minimal recipe for test-time scaling and strong reasoning performance matching o1-preview with just 1,000 examples & budget forcing
 </p>
</div>
<br>

![](visuals/scaling.png)

This repository provides an overview of all resources for the paper ["s1: Simple test-time scaling"](https://arxiv.org/abs/TODO).

- [Artifacts](#artifacts)
- [Structure](#structure)
- [Inference](#inference)
- [Training](#pretraining)
- [Evaluation](#evaluation)
- [Data](#data)
- [Visuals](#visuals)
- [Citation](#citation)

### Artifacts

- **Paper**: https://arxiv.org/abs/TODO
- **Model**: https://hf.co/simplescaling/s1-32B
- **Data**: https://hf.co/simplescaling/s1K

### Structure

- `eval/`: Evaluation scripts
- `data/`: Synthetic data creation scripts & co
- `train/`: Training scripts

### Inference

#### vLLM

TODO (Niklas): Add budget forcing

Install the `vllm` library and run:
```python
from vllm import LLM, SamplingParams
model = LLM(
    "simplescaling/s1-32B",
    tensor_parallel_size=2,
)
tok = AutoTokenizer.from_pretrained("simplescaling/s1-32B")

stop_token_ids = tok("<|im_end|>")["input_ids"]

sampling_params = SamplingParams(
    max_tokens=32768,
    min_tokens=0,
    stop_token_ids=stop_token_ids,
)

prompt = "How many r in raspberry"
prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"

o = model.generate(prompt, sampling_params=sampling_params)
print(o[0].outputs[0].text)
```

#### transformers

Install the `transformers` & `torch` libraries and run:

```python
from transformers import AutoModelCausalLM, AutoTokenizer
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "simplescaling/s1-32B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "How many r in raspberry"
messages = [
    {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### Training

To run training, you can find our script at `train/sft.py` which you can invoke via one of the `train/sft*sh` scripts which in turn you can launch via `train/launch.sh` if you are on a SLURM cluster (requires editing the file for your cluster setup).

### Evaluation

We cloned [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) at commit `4cec66e4e468d15789473d6d63c3a61a751fa524` and modified it. Setup:
```bash
cd eval/lm-evaluation-harness
pip install -e .[math,vllm]
```

All commands are in `eval/scripts/commands.sh`. For AIME24 we always pick the `aime24_nofigures` result, which uses a dataset that only contains the AIME24 figures if they are important for the task.

If you want to compute statistics (avg thinking tokens etc) for an evaluation run you can use 
`python eval/scripts/compute_sample_stats.py path_to_samples_file.jsonl`

To run REBASE
... @Weijia ...

### Data

To recreate our data:
1. Run `data/collect_data.py` followed by `data/fix_gpqa.py` & `data/add_aime.py` to collect the questions; Make sure to change the hub path in the respective files to one of your own
2. ... @Zitong ...

### Visuals

All figures and tables are created via [this colab](https://colab.research.google.com/drive/1GAfwbJs2Y1dgGGsxrQyQg2G7CRH5NgN3?usp=sharing) equivalent to `visuals/visuals.ipynb`.

### Known Issues

- vLLM throws `ValueError: Token id XXXXX is out of vocabulary`
  - This can happen with budget forcing, especially when running with temperature 1, where the model will sometimes do crazy stuff and predict a vocab id that is larger than its max token id but still within its embedding size i.e. anyting <152064, >151664; When we refeed the model's previous outputs to it which is done when setting e.g. max_thinking_tokens in the evaluation then this will cause the error cuz vLLM does this check even though it would only be an issue for IDs >152064. To fix it you can just uncomment the vLLM ValueError (It is the line `if max_input_id > tokenizer.max_token_id:` in `vllm/engine/llm_engine.py`)

### Citation

```bibtex
TODO
```
