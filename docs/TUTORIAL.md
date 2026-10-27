# Tutorial: LoRA Fine-Tuning a Language Model (Step by Step)

This project fine-tunes a small language model using **LoRA (Low-Rank
Adaptation)** — the technique that makes fine-tuning possible without a data
center. It's the most advanced project in the portfolio and connects directly to
applied-LLM work.

---

## ⚠️ Where to run this

Fine-tuning needs a GPU and the PyTorch + Hugging Face stack, which won't run on
an Intel Mac. **Use Google Colab (free GPU).** This is also how LoRA fine-tuning
is done in practice, so it's the realistic environment, not a workaround.

### Colab recipe
1. https://colab.research.google.com → New notebook → Runtime → **GPU**.
2. Install the stack:
   ```python
   !pip -q install torch transformers peft datasets accelerate
   ```
3. Paste the contents of `src/finetune_lora.py`, then run:
   ```python
   train()          # fine-tunes and saves the LoRA adapter
   generate()       # loads base + adapter and generates text
   ```
   With the tiny demo dataset this finishes in a minute or two on a Colab GPU.

---

## The big idea: why LoRA exists

Fine-tuning normally means updating **all** of a model's weights. For modern
models that's billions of parameters — far too much memory for a single GPU.

**LoRA freezes the original model** and injects small trainable "adapter"
matrices into the attention layers. You train only those adapters — often **less
than 1%** of the parameters. Results are close to full fine-tuning, but it fits
on a free GPU, and the saved adapter is only a few megabytes (the giant base
model stays untouched and shared).

`model.print_trainable_parameters()` in the code prints exactly how tiny the
trainable fraction is — run it and see.

---

## The workflow (map each to `src/finetune_lora.py`)

### 1. Load the base model + tokenizer
`AutoModelForCausalLM.from_pretrained("distilgpt2")`. We use a small model so it
trains fast on free hardware; the technique is identical for larger models.

### 2. Attach LoRA adapters
`LoraConfig(r=8, ...)` then `get_peft_model(...)`. `r` is the adapter rank
(bigger = more capacity, more to train). `target_modules=["c_attn"]` injects
adapters into GPT-2's attention projection. The base weights are now frozen.

### 3. Prepare the dataset
The demo uses a handful of `### Instruction / ### Response` examples. **This is
where your real work goes** — a meaningful fine-tune needs hundreds to thousands
of examples in your target style/domain. The format teaches the model a
consistent instruction→response behavior.

### 4. Train
A Hugging Face `Trainer` runs the loop (same forward→loss→backward→step idea as
the other projects, just wrapped). Only the LoRA adapters update.

### 5. Save just the adapter
`model.save_pretrained("./lora-adapter")` saves only the small adapter weights,
not the whole base model.

### 6. Inference
`generate()` loads the base model, applies the adapter with `PeftModel`, and
generates text in the fine-tuned style.

---

## Honest expectations
- `distilgpt2` is small, and the demo dataset is tiny, so the *output quality*
  is modest — the point is to **learn and demonstrate the LoRA workflow**, not
  to produce a polished model.
- To get real results: use a stronger base model (e.g. a small Llama/Qwen),
  a real dataset of hundreds+ examples, and more training steps.

---

## How this connects to the portfolio
- **neural-net-from-scratch** → training mechanics by hand.
- **ml-training-pipeline** (sklearn) → classical ML workflow.
- **pytorch-image-classifier** → deep-learning training loop.
- **llm-lora-finetune** (this) → modern, parameter-efficient LLM fine-tuning.
- **rag-from-scratch** → the *other* main way to adapt an LLM to your data
  (retrieval instead of fine-tuning). Together, RAG + LoRA cover the two primary
  approaches to customizing LLM behavior.

---

## Exercises
1. Add 20+ of your own instruction/response examples and re-train. Does the
   style transfer?
2. Change `r` (8 → 16 → 32). More rank = more trainable params; watch the % and
   the results.
3. Compare: when would you use **RAG** vs **fine-tuning**? (Hint: facts/knowledge
   vs style/behavior.)
