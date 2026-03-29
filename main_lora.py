import os
import random
import logging
import time
import inspect
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE_DIR = Path(__file__).resolve().parent

# Windows + huggingface_hub: tránh đường tải XET dễ bị đứng và tắt cảnh báo symlink.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Dùng model có safetensors trên nhánh chính để tránh lỗi torch.load (CVE-2025-32434).
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "main"
DATA_PATH = str((BASE_DIR / "../dataset/mental_health_vi_final.json").resolve())
OUTPUT_DIR = str((BASE_DIR / "./finetuned-qwen2.5-1.5b-lora-fast").resolve())

# Fast preset cho RTX 4070 SUPER 12GB: tăng tốc train, chấp nhận giảm nhẹ chất lượng.
MAX_LEN = 512
EPOCHS = 2
PER_DEVICE_BATCH = 4
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
SEED = 42
VALIDATION_SPLIT = 0.05
SAVE_TOTAL_LIMIT = 2
OPTIMIZER = "paged_adamw_8bit"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def supports_bf16():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def get_device():
    """Lấy device (GPU hoặc CPU)."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def log_device_info():
    """In ra thông tin về device."""
    device = get_device()
    logger.info(f"Device being used: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        logger.info(f"Số GPU khả dụng: {torch.cuda.device_count()}")


def get_target_modules(model_name):
    lower = model_name.lower()
    if "falcon" in lower:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    if any(k in lower for k in ["qwen", "llama", "mistral", "gemma"]):
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    return ["q_proj", "v_proj"]


def build_quantization_config():
    compute_dtype = torch.bfloat16 if supports_bf16() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def extract_last_assistant_reply(text):
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("Trợ lý:"):
            prompt = "\n".join(lines[:i]) + "\nTrợ lý:"
            response = lines[i].replace("Trợ lý:", "").strip()
            return prompt, response
    return "Trợ lý:", text.strip()


def build_text_from_example(example):
    if "instruction" in example and "output" in example:
        instr = example.get("instruction", "").strip()
        out = example.get("output", "").strip()
        if not out:
            return None, None

        if "Người dùng:" in instr or "Trợ lý:" in instr:
            prompt = instr
            if not prompt.endswith("Trợ lý:"):
                prompt = prompt + "\nTrợ lý:"
            return prompt, out

        return f"Người dùng: {instr}\nTrợ lý:", out

    return extract_last_assistant_reply(example.get("text", "").strip())


def tokenize_and_build(example, tokenizer):
    prompt, response = build_text_from_example(example)
    if not prompt or not response:
        return None

    full_text = prompt + " " + response + tokenizer.eos_token
    # Dynamic padding is handled in the data collator to avoid wasting compute on PAD tokens.
    enc = tokenizer(full_text, truncation=True, max_length=MAX_LEN)
    labels = enc["input_ids"].copy()
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100
    enc["labels"] = labels
    return enc


def process_dataset(ds, tokenizer, desc="Tokenizing"):
    if ds is None:
        return None

    processed = []
    for ex in tqdm(ds, total=len(ds), desc=desc):
        encoded = tokenize_and_build(ex, tokenizer)
        if encoded is not None:
            processed.append(encoded)

    return Dataset.from_list(processed)


class TimeCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.time()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        t = time.time() - self.epoch_start
        logger.info(f"Epoch {int(state.epoch)} time: {t/60:.2f} phút")

    def on_train_end(self, args, state, control, **kwargs):
        t = time.time() - self.train_start
        logger.info(f"Tổng thời gian train: {t/60:.2f} phút")


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    using_ddp = world_size > 1

    if torch.cuda.is_available():
        if local_rank >= 0:
            torch.cuda.set_device(local_rank)
            device_map = {"": local_rank}
        else:
            torch.cuda.set_device(0)  # Chắc chắn set GPU 0
            device_map = {"": 0}
    else:
        device_map = None

    logger.info(
        "Runtime: cuda=%s, gpus=%s, ddp=%s, bf16=%s",
        torch.cuda.is_available(),
        torch.cuda.device_count(),
        using_ddp,
        supports_bf16(),
    )
    log_device_info()  # In thông tin GPU

    logger.info("Loading dataset from: %s", DATA_PATH)
    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    logger.info("Loaded dataset with %d samples", len(ds))
    if VALIDATION_SPLIT > 0:
        ds = ds.train_test_split(test_size=VALIDATION_SPLIT, seed=SEED)
        train_ds, eval_ds = ds["train"], ds["test"]
        logger.info(
            "Split dataset: train=%d, eval=%d", len(train_ds), len(eval_ds)
        )
    else:
        train_ds, eval_ds = ds, None
        logger.info("Using full dataset for training: train=%d", len(train_ds))

    logger.info(
        "Preparing tokenizer/model from %s @ %s", MODEL_NAME, MODEL_REVISION
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.padding_side = "right"

    bnb_config = (
        build_quantization_config() if torch.cuda.is_available() else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        quantization_config=bnb_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    logger.info("Model load completed.")

    # Kiểm tra model ở đâu
    logger.info(f"Model device sau load: {next(model.parameters()).device}")

    # Nếu GPU có sẵn nhưng model không trên GPU, move nó lên GPU
    if (
        torch.cuda.is_available()
        and next(model.parameters()).device.type != "cuda"
    ):
        logger.warning("Model không ở trên GPU, đang move lên GPU...")
        gpu_id = device_map.get("", 0) if device_map else 0
        model = model.to(f"cuda:{gpu_id}")

    # Keep checkpointing disabled to avoid torch checkpoint warnings and improve throughput on 12GB VRAM.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=False
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=get_target_modules(MODEL_NAME),
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)

    logger.info("Tokenizing train dataset...")
    train_tok = process_dataset(train_ds, tokenizer, desc="Tokenizing train")
    logger.info("Tokenized train dataset: %d samples", len(train_tok))

    if eval_ds:
        logger.info("Tokenizing eval dataset...")
    eval_tok = (
        process_dataset(eval_ds, tokenizer, desc="Tokenizing eval")
        if eval_ds
        else None
    )
    if eval_tok:
        logger.info("Tokenized eval dataset: %d samples", len(eval_tok))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    dataloader_workers = 0 if os.name == "nt" else min(8, os.cpu_count() or 2)
    logger.info("Using dataloader_num_workers=%d", dataloader_workers)

    training_args_kwargs = {
        "output_dir": OUTPUT_DIR,
        "per_device_train_batch_size": PER_DEVICE_BATCH,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "num_train_epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": 100,
        "weight_decay": 0.05,
        "bf16": supports_bf16(),
        "fp16": (torch.cuda.is_available() and not supports_bf16()),
        "logging_steps": 50,
        "save_strategy": "epoch",
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "remove_unused_columns": False,
        "dataloader_num_workers": dataloader_workers,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "report_to": "none",
        "optim": OPTIMIZER,
        "gradient_checkpointing": False,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "ddp_find_unused_parameters": False if using_ddp else None,
        "max_grad_norm": 0.3,
        "disable_tqdm": False,
    }

    # transformers version compatibility: map and filter kwargs by real signature
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_value = "epoch" if eval_tok else "no"
    if "evaluation_strategy" in ta_params:
        training_args_kwargs["evaluation_strategy"] = eval_value
    elif "eval_strategy" in ta_params:
        training_args_kwargs["eval_strategy"] = eval_value
    else:
        logger.warning(
            "Neither evaluation_strategy nor eval_strategy is supported by this transformers version."
        )

    filtered_training_args_kwargs = {}
    for key, value in training_args_kwargs.items():
        if key not in ta_params:
            logger.warning(
                "Skipping unsupported TrainingArguments arg: %s", key
            )
            continue
        if value is None:
            continue
        filtered_training_args_kwargs[key] = value

    training_args = TrainingArguments(**filtered_training_args_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_tok,
        "eval_dataset": eval_tok,
        "data_collator": data_collator,
        "callbacks": [TimeCallback()],
    }

    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        logger.warning(
            "Neither tokenizer nor processing_class is supported by this Trainer version."
        )

    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
