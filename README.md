# LLM LoRA Fine-Tuning

Fine-tune a small language model using **LoRA (Low-Rank Adaptation)** — the
parameter-efficient technique that makes fine-tuning possible on a single free
GPU by training tiny adapter matrices instead of the whole model.

## ⚠️ Where to run
Fine-tuning needs a GPU and the PyTorch + Hugging Face stack, so run this on
**Google Colab (free GPU)** — which is also how LoRA fine-tuning is done in
practice. A full Colab recipe is in the tutorial.

## What it demonstrates
- **Why LoRA:** freeze the base model, train only small adapters (<1% of params)
- Attaching LoRA adapters to attention layers with `peft`
- Formatting an instruction dataset and running a Hugging Face `Trainer`
- Saving just the adapter (a few MB) and running inference with base + adapter
- How fine-tuning (style/behavior) complements **RAG** (facts/knowledge)

## Quick start (Google Colab)
```python
!pip -q install torch transformers peft datasets accelerate
# paste src/finetune_lora.py, then:
train()
generate()
```

## Learn it step by step
See **[docs/TUTORIAL.md](docs/TUTORIAL.md)** — explains LoRA, the full workflow,
honest expectations, and when to choose RAG vs fine-tuning.

## Project layout
```
src/finetune_lora.py   # LoRA fine-tuning script (heavily commented)
docs/TUTORIAL.md       # step-by-step tutorial + Colab recipe
```

## Stack
Python · PyTorch · Hugging Face Transformers · PEFT (LoRA) · distilgpt2
