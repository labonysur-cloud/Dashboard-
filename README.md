🚀 I'm excited and a bit nervous to share my first-ever AI project – Linkieee: an AI-powered LinkedIn content generator.

This project is still a work-in-progress and there's a lot of improvement needed, but it's been a huge learning journey for me.

🛠 What I’ve done so far:

- Collected and cleaned a custom dataset of 500+ samples
- Fine-tuned two models:
  - TinyLlama-1.1B-Chat-v1.0 (for English)
  - ai-forever/mGPT (for Bengali)
- Built an interactive frontend with Gradio
- Deployed on Google Colab + Hugging Face Spaces

🌍 My goal

My goal is to help students and professionals (especially in Bangladesh 🇧🇩) write better LinkedIn posts — in both English and Bengali.

🔗 Try it here (feedback is welcome!):

👉 https://lnkd.in/eTHPJrBn

📂 Source Code:

👉 https://lnkd.in/ej27VFii

---

# Dashboard- — AI LinkedIn Post Generator (Linkieee)

One-sentence summary
This repository contains a reproducible Jupyter notebook that demonstrates how to fine-tune an efficient causal language model with LoRA (PEFT) and bitsandbytes quantization to generate LinkedIn posts (English and Bengali) and expose the model via a Gradio UI.

Overview
Dashboard- (Linkieee) is a research/demo project for ML engineers, NLP practitioners, students, and professionals who want to experiment with memory-efficient fine-tuning of LLMs for domain-specific text generation. The notebook walks through dataset preparation, LoRA-based fine-tuning using trl.SFTTrainer, 4-bit quantized model loading (bitsandbytes), inference with Hugging Face pipelines, and a Gradio-based interactive demo.

Key capabilities demonstrated
- Convert CSV rows into instruction/response training prompts and build a Hugging Face Dataset
- Configure and perform LoRA (PEFT) fine-tuning using trl.SFTTrainer
- Load base models with bitsandbytes 4-bit quantization (bnb) for memory-efficient training/inference
- Fine-tune separate models (English, Bengali) and save them to distinct output directories
- Run inference with transformers.pipeline and serve a simple Gradio UI for interactive generation

Repository layout
```
README.md                Project README (this file)
Linkieee.ipynb           Jupyter notebook: dataset prep, model config, training, inference, and Gradio UI
```

Stack
- Language: Python (>= 3.8)
- Notebook runtime: Jupyter / Google Colab
- Notable libraries:
  - transformers (Hugging Face)
  - datasets (Hugging Face)
  - peft (LoRA)
  - bitsandbytes (4-bit quantization)
  - trl (SFTTrainer)
  - gradio (lightweight UI)
  - huggingface_hub

How it fits together (runtime shape)
The notebook (Linkieee.ipynb) performs these steps in sequence:
1. Load CSV training data and convert each row into an instruction-style prompt (Instruction / Response).
2. Build a Hugging Face Dataset object from the formatted text column.
3. Load a base causal LM and tokenizer (example IDs in notebook), configure BitsAndBytesConfig for 4-bit quantization.
4. Apply a LoRA configuration (PEFT) and run fine-tuning with trl.SFTTrainer using TrainingArguments.
5. Save the fine-tuned model(s) to disk.
6. Load the saved model via transformers.pipeline for text-generation and wrap it with a Gradio interface for interactive generation.

How to run it

Quickstart — Google Colab (recommended for GPU):
1. Open the notebook in Colab:
   https://colab.research.google.com/github/labonysur-cloud/Dashboard-/blob/main/Linkieee.ipynb
2. Install required packages (run in a cell):
```bash
!pip install -q -U transformers datasets accelerate peft bitsandbytes trl huggingface_hub gradio pandas
```
3. Upload or mount your CSV dataset (the notebook expects a file like `just for linkedin - just for linkedin (1).csv`).
4. Run the notebook cells top-to-bottom to prepare data, fine-tune, save models, and launch the Gradio demo.

Local environment (alternative):
1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.\\.venv\\Scripts\\activate     # Windows
pip install --upgrade pip
```
2. Install dependencies:
```bash
pip install transformers datasets accelerate peft bitsandbytes trl huggingface_hub gradio pandas
```
3. Run the notebook with Jupyter, or convert the notebook to a script and run relevant steps manually.

Dataset format (required columns)
The notebook expects a CSV with at least these columns:
- Topic/Industry
- Tone
- Goal/Objective
- Generated Content

The notebook formats each row into a training example like:
```
### Instruction:
Generate a LinkedIn post with the following properties:
- Topic: {Topic/Industry}
- Tone: {Tone}
- Goal: {Goal/Objective}

### Response:
{Generated Content}
```
Ensure `Generated Content` contains the desired output text for supervised fine-tuning.

Training (notebook examples)
- Example LoRA configuration:
  - r = 8
  - target_modules = ["q_proj","o_proj","k_proj","v_proj","gate_proj","up_proj","down_proj"] (or "c_attn" for some GPT-style models)
  - task_type = "CAUSAL_LM"
- Example TrainingArguments:
  - output_dir = "linkedin_content_generator"
  - num_train_epochs = 3
  - per_device_train_batch_size = 4
  - learning_rate = 2e-4
  - logging_steps = 10–20
- Trainer: trl.SFTTrainer (model, train_dataset, args, peft_config)

Inference and demo
- After training, load the saved model:
```python
from transformers import pipeline
pipe = pipeline("text-generation", model="linkedin_content_generator_final", tokenizer=tokenizer, max_new_tokens=256)
```
- The notebook demonstrates extracting only the generated response by splitting on `### Response:` and provides a Gradio interface.

Model choices & recommendations
- Example base models used in the notebook:
  - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — lightweight example
  - `ai-forever/mGPT` — better Bengali support (example)
- Use bitsandbytes 4-bit (bnb) quantization to reduce VRAM. If bf16 is not supported on your GPU, try fp16 or adjust device_map and compute dtype.
- For larger datasets or production, consider:
  - Larger models + more epochs and distributed training
  - Hugging Face Accelerate or multi-GPU training
  - Validation split and evaluation metrics (BLEU, ROUGE, or human eval)

Hardware & runtime notes
- A GPU is strongly recommended (Colab T4 used in notebook metadata).
- bfloat16/bf16 settings require GPU/driver support. If unsupported, fall back to float16 or float32.
- bitsandbytes may require matching CUDA versions — consult their installation docs if you hit build/runtime errors.

Security, data & ethical considerations
- Inspect and clean training data to avoid toxic, private, or leaked content.
- Fine-tuned models inherit biases from training data; validate outputs carefully before public use.
- When deploying a public demo, consider rate-limiting, content filters, and usage monitoring.

Recommended requirements file (example)
```
transformers
datasets
accelerate
peft
bitsandbytes
trl
huggingface-hub
gradio
pandas
```

Troubleshooting tips
- bitsandbytes install fails: check CUDA and driver versions and use prebuilt wheels where available.
- Out of memory: decrease per_device_train_batch_size, use gradient_accumulation, or use smaller base models.
- Tokenizer/model load error: confirm model_id is correct and you have access (use a Hugging Face token for private models).

Next improvements (suggested)
- Add a requirements.txt or environment.yml
- Add a reproducible training script (Python script + CLI) instead of or alongside the notebook
- Add validation/eval cells and a small test split to measure generation quality
- Add a LICENSE file (MIT/Apache-2.0 recommended) and a CONTRIBUTING guide
- Optionally push fine-tuned models to the Hugging Face Hub with proper metadata

Contributing
This repository is a demo. Contributions welcome — open an issue or submit a PR with fixes, reproducible scripts, or improved dataset handling.

License
No license is set. Add a LICENSE file if you want to open-source the work (MIT / Apache-2.0 suggested).

Contact
Repository owner: https://github.com/labonysur-cloud
