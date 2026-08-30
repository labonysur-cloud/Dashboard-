# Dashboard-

AI LinkedIn Post Generator

Overview

Dashboard- is a research / demo repository that contains an end-to-end Jupyter Notebook demonstrating how to fine-tune a causal language model (LLM) to generate LinkedIn posts. The notebook shows dataset preparation, LoRA-based fine-tuning (PEFT), 4-bit quantized model loading using bitsandbytes, inference with the Hugging Face pipeline, and a simple Gradio web UI for interactive generation. This project is intended for ML engineers, NLP researchers, and anyone experimenting with efficient fine-tuning of LLMs for domain-specific text generation (including multilingual content such as Bengali).

Key capabilities demonstrated

- Load and preprocess CSV training data into a Hugging Face Dataset
- Train a causal LM with PEFT / LoRA and trl.SFTTrainer
- Use bitsandbytes 4-bit quantization (bnb) for memory-efficient model loading
- Fine-tune both English and Bengali-capable models and maintain separate output directories
- Run inference using transformers pipeline and expose the model via a Gradio app

Repository layout

```
README.md                Project README (this file)
Linkieee.ipynb           Jupyter notebook: dataset prep, training, inference, and Gradio UI
```

How it fits together

The single notebook (Linkieee.ipynb) contains the full flow: it loads a CSV file of example LinkedIn posts, formats each row into an instruction/response prompt, builds a Hugging Face Dataset, configures a base model and tokenizer (examples: TinyLlama/TinyLlama-1.1B-Chat-v1.0 and ai-forever/mGPT), applies LoRA (PEFT) settings, runs training with trl.SFTTrainer, saves the fine-tuned model to a directory, and demonstrates inference and a Gradio UI to generate posts interactively.

Technical stack

- Language: Python (>=3.8)
- Notebook environment: Jupyter / Google Colab
- Transformers: Hugging Face Transformers
- PEFT: peft (LoRA)
- Trainer: trl.SFTTrainer
- Quantization: bitsandbytes (4-bit, nf4)
- Dataset: datasets (Hugging Face)
- Inference / serving: transformers.pipeline
- Lightweight web UI: Gradio

Notable libraries referenced in the notebook

- transformers
- datasets
- accelerate
- peft
- bitsandbytes
- trl
- huggingface_hub
- gradio

Hardware and runtime notes

- The notebook configures BitsAndBytesConfig to load models in 4-bit (bnb) which significantly reduces GPU memory usage. This makes it possible to run large-ish models on single GPUs with limited memory (e.g., T4, A10, A100 depending on model size).
- A GPU is strongly recommended for training and faster inference. Colab GPU (T4) is used in the notebook's metadata badge, and the notebook sets device_map to use the GPU.
- When using bfloat16 or bf16 compute dtype, ensure your CUDA/cuDNN setup and GPU support the chosen dtype. If not available, fallback to float16 or float32.

Quickstart — run in Google Colab (recommended)

1. Open the notebook in Colab:
   - https://colab.research.google.com/github/labonysur-cloud/Dashboard-/blob/main/Linkieee.ipynb
2. Install requirements in a notebook cell:

```bash
!pip install -q -U transformers datasets accelerate peft bitsandbytes trl huggingface_hub gradio
```

3. Upload or mount your CSV dataset. The notebook expects a CSV file named like:

```
just for linkedin - just for linkedin (1).csv
```

This CSV should include columns referenced in the notebook: Topic/Industry, Tone, Goal/Objective, Generated Content.

4. Run the notebook cells in order to prepare the Dataset, load a base model, configure LoRA, train, save, and run inference.

Local environment (alternative)

1. Create and activate a Python virtual environment (Python 3.8+)

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.\.venv\Scripts\activate   # Windows
pip install --upgrade pip
```

2. Install dependencies

```bash
pip install transformers datasets accelerate peft bitsandbytes trl huggingface_hub gradio
```

3. Run the notebook with jupyter or convert to a script. You can also run inference by loading the saved model directories (`linkedin_content_generator_final` or `bengali_content_generator_final`) with transformers.pipeline.

Preparing your dataset

- The notebook shows how to convert CSV rows into instruction-style prompts. Each row is formatted into a single text field with a structure like:

```
### Instruction:
Generate a LinkedIn post with the following properties:
- Topic: {Topic/Industry}
- Tone: {Tone}
- Goal: {Goal/Objective}

### Response:
{Generated Content}
```

- Ensure your CSV has `Generated Content` column with the expected response text and columns used in the prompt formation.

Training

- The notebook demonstrates LoRA configuration and uses trl.SFTTrainer with TrainingArguments. Example settings include:
  - num_train_epochs=3
  - per_device_train_batch_size=4
  - learning_rate=2e-4
  - output_dir set to `linkedin_content_generator` or `bengali_content_generator`
- Training with LoRA (PEFT) reduces the number of parameters that need to be updated, making fine-tuning feasible on limited hardware.

Inference

- After training, the notebook loads the saved model with transformers.pipeline("text-generation") and uses the same prompt structure for generating content.
- The notebook includes extraction of the generated text by splitting on the `### Response:` marker.
- A Gradio UI is provided that wraps the pipeline and allows interactive generation given Topic, Tone, and Goal.

Model choices and recommendations

- The notebook uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` as a lightweight public example and `ai-forever/mGPT` for stronger Bengali support. These are examples — pick the base model that fits your target language and compute budget.
- For production or larger datasets, consider using larger base models and distributed training or Hugging Face Accelerate.
- Validate and filter your training data to avoid low-quality or harmful outputs. Fine-tuned models can amplify biases present in the training data.

Environment variables and credentials

- If you use private Hugging Face models or want to push models to the Hub, set HF_TOKEN (Hugging Face access token) and call `huggingface_hub.login(token)` or use `notebook_login()` in Colab.

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

Caveats and troubleshooting

- bitsandbytes installation can fail on some platforms; refer to its installation docs and ensure CUDA toolkit compatibility.
- If you run out of GPU memory, lower the batch size, switch to gradient checkpointing, or use smaller base models.
- When the tokenizer or model cannot be loaded, verify the `model_id` strings and that the models are publicly available or that you have access tokens for private models.

Contributing

This repository is a personal project / example. Contributions are welcome — open an issue or submit a pull request with improvements, bug fixes, or more reproducible training scripts.

License

No license is set in this repository. If you want to open-source the work, add a LICENSE file (MIT, Apache-2.0, etc.).

Contact

If you need help or want to collaborate, open an issue or contact the repository owner: https://github.com/labonysur-cloud
