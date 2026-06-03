#!/usr/bin/env python3
"""
LoRA Fine-Tuning Example - Nexus-LLM
======================================
Demonstrates how to fine-tune a model using LoRA (Low-Rank Adaptation).
"""

from nexus_llm import InferenceEngine
from nexus_llm.training import LoRAConfig, Trainer, DatasetLoader


def main():
    # Load and prepare the dataset
    dataset_loader = DatasetLoader()
    train_dataset = dataset_loader.load_jsonl(
        path="data/train.jsonl",
        split="train",
        validation_split=0.1,
    )
    val_dataset = dataset_loader.load_jsonl(
        path="data/train.jsonl",
        split="validation",
        validation_split=0.1,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Configure LoRA parameters
    lora_config = LoRAConfig(
        r=16,                        # LoRA rank
        lora_alpha=32,               # LoRA scaling factor
        target_modules=[             # Modules to apply LoRA to
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,           # Dropout probability
        bias="none",                 # Bias handling
        task_type="CAUSAL_LM",       # Task type
    )

    # Initialize the trainer
    trainer = Trainer(
        base_model="nexus-7b-chat",
        lora_config=lora_config,
        output_dir="./checkpoints/lora-finetune",
        training_args={
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "eval_strategy": "steps",
            "eval_steps": 100,
            "save_strategy": "steps",
            "save_steps": 100,
            "save_total_limit": 3,
            "fp16": True,
            "gradient_checkpointing": True,
            "report_to": "tensorboard",
        },
    )

    # Start training
    print("Starting LoRA fine-tuning...")
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Save the LoRA adapter
    adapter_path = "./checkpoints/lora-finetune/final_adapter"
    trainer.save_adapter(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}")

    # Test the fine-tuned model
    print("\n--- Testing fine-tuned model ---")
    engine = InferenceEngine(
        model_name="nexus-7b-chat",
        adapter_path=adapter_path,
        device="auto",
    )

    response = engine.chat("Explain the key principles of our product's return policy.")
    print(f"Fine-tuned response: {response.text}")

    # Merge LoRA weights into the base model (optional)
    print("\nMerging LoRA weights into base model...")
    merged_path = "./models/nexus-7b-finetuned-merged"
    trainer.merge_and_save(merged_path)
    print(f"Merged model saved to {merged_path}")


if __name__ == "__main__":
    main()
