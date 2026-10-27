"""
LoRA Fine-Tuning of a Small Language Model.

   NOTE ON WHERE THIS RUNS
   -----------------------
   Fine-tuning a language model needs a GPU and the PyTorch + Hugging Face stack,
   which will NOT run on an Intel Mac. Run this on GOOGLE COLAB (free GPU) — see
   docs/TUTORIAL.md for the exact notebook recipe. This file is the training
   script you paste into Colab.

What it teaches (parameter-efficient fine-tuning):
   - WHY LoRA: full fine-tuning updates ALL of a model's (often billions of)
     weights, which needs huge memory. LoRA (Low-Rank Adaptation) FREEZES the
     base model and trains only tiny "adapter" matrices injected into attention
     layers — typically <1% of the parameters. This makes fine-tuning possible
     on a single free GPU.
   - The workflow: load base model -> attach LoRA adapters -> format a dataset
     -> train -> save the (tiny) adapter -> run inference with base + adapter.

We fine-tune a small model (distilgpt2) to adopt a consistent style/behavior
from a tiny instruction dataset — small enough to finish on a free Colab GPU.
"""

from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model

BASE_MODEL = "distilgpt2"   # small enough for a free Colab GPU


# ---------------------------------------------------------------------------
# A tiny demonstration dataset. In a real project this would be hundreds or
# thousands of examples in your target style/domain.
# ---------------------------------------------------------------------------
TRAIN_EXAMPLES = [
    "### Instruction: Describe a cat.\n### Response: A cat is a small, independent feline companion known for curiosity and agility.",
    "### Instruction: Describe a dog.\n### Response: A dog is a loyal, social animal valued for companionship and trainability.",
    "### Instruction: Describe the sun.\n### Response: The sun is the star at the center of our solar system, providing light and heat.",
    "### Instruction: Describe water.\n### Response: Water is a clear, essential liquid made of hydrogen and oxygen that sustains life.",
    # ... add many more real examples for a meaningful fine-tune ...
]


def build():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # The LoRA config: inject low-rank adapters into the attention projections.
    # r = rank (adapter size); only these adapter weights are trained.
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn"],   # GPT-2 attention projection
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()   # shows the tiny % of params being trained
    return model, tokenizer


def make_dataset(tokenizer):
    ds = Dataset.from_dict({"text": TRAIN_EXAMPLES})
    def tok(batch):
        out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)
        return out
    return ds.map(tok, batched=True, remove_columns=["text"])


def train():
    model, tokenizer = build()
    ds = make_dataset(tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="./lora-out",
        per_device_train_batch_size=2,
        num_train_epochs=10,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()

    # Save ONLY the LoRA adapter — a few MB, not the whole base model.
    model.save_pretrained("./lora-adapter")
    tokenizer.save_pretrained("./lora-adapter")
    print("Saved LoRA adapter -> ./lora-adapter (small; base model stays separate)")


def generate(prompt="### Instruction: Describe a bird.\n### Response:"):
    """Load base model + LoRA adapter and generate text."""
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained("./lora-adapter")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, "./lora-adapter")
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        generate()
    else:
        train()
