# ============================================
# train_qwen_unsloth.py - With Unsloth + 4-bit Quantization
# ============================================

#!/usr/bin/env python3
"""
Training Qwen2.5-Coder-1.5B with Unsloth for 2-5x faster training
Run: python train_qwen_unsloth.py
"""

import os
import torch
import argparse
import pandas as pd
import re
from tqdm import tqdm
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import torch.utils._pytree
try:
    torch.utils._pytree.register_constant = lambda x: x
except AttributeError:
    pass

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

class Config:
    # Model - Unsloth optimized
    model_name = "unsloth/Qwen2.5-Coder-1.5B-Instruct"  # Unsloth's optimized version[citation:8]
    
    # Dataset
    dataset_name = "yikun-li/TitanVul"
    
    # Training - Optimized for Unsloth
    max_seq_length = 2048  # Up to 128K supported, but 2K is efficient[citation:8]
    batch_size = 2
    gradient_accumulation = 8  # Effective batch = 16
    learning_rate = 2e-4
    num_epochs = 3
    
    # LoRA config (Unsloth optimized)
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                           "gate_proj", "up_proj", "down_proj"]
    
    # 4-bit quantization (QLoRA)
    load_in_4bit = True  # ~75% memory reduction[citation:10]
    bnb_4bit_quant_type = "nf4"
    bnb_4bit_compute_dtype = torch.bfloat16
    bnb_4bit_use_double_quant = True
    
    # System prompt for vulnerability detection
    system_prompt = """You are a code security expert. Analyze the provided code and determine if it contains a security vulnerability.

Classification rules:
- If the code contains a vulnerability (buffer overflow, SQL injection, etc.), respond with "VULNERABLE"
- If the code is secure, respond with "SAFE"
- For vulnerable code, also identify the CWE type

Respond in the following format:
VERDICT: [VULNERABLE/SAFE]
CWE: [CWE-ID if vulnerable, else NONE]"""

# ============================================
# DATA PROCESSING (Same as before)
# ============================================

def extract_cwe_number(cwe_string):
    if pd.isna(cwe_string) or not isinstance(cwe_string, str):
        return None
    if cwe_string == 'NVD-CWE-noinfo':
        return None
    match = re.search(r'(\d+)', cwe_string)
    if match:
        return int(match.group(1))
    return None

def prepare_training_data(config):
    """Load TitanVul and format for Unsloth training"""
    print("\n" + "="*50)
    print("LOADING TITANVUL DATASET")
    print("="*50)
    
    dataset = load_dataset(config.dataset_name)
    df = dataset['train'].to_pandas()
    print(f"Loaded {len(df)} samples")
    
    # Process vulnerable and safe samples
    vulnerable = []
    safe = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        vuln_code = row['func_before']
        fixed_code = row['func_after']
        cwe_str = row['cwe_id']
        
        cwe_num = extract_cwe_number(cwe_str)
        if cwe_num is None:
            continue
        
        if not isinstance(vuln_code, str) or len(vuln_code.strip()) < 20:
            continue
        
        vulnerable.append({
            'code': vuln_code,
            'label': 1,
            'cwe_string': cwe_str,
        })
        
        if isinstance(fixed_code, str) and len(fixed_code.strip()) > 20:
            if fixed_code.strip() != vuln_code.strip():
                safe.append({
                    'code': fixed_code,
                    'label': 0,
                    'cwe_string': None,
                })
    
    # Balance dataset
    min_samples = min(len(vulnerable), len(safe))
    vulnerable = vulnerable[:min_samples]
    safe = safe[:min_samples]
    
    result_df = pd.DataFrame(vulnerable + safe)
    print(f"\nFinal dataset: {len(result_df)} samples")
    print(f"  Vulnerable: {result_df['label'].sum()}")
    print(f"  Safe: {len(result_df) - result_df['label'].sum()}")
    
    # Format for instruction tuning
    formatted_data = []
    for _, row in result_df.iterrows():
        if row['label'] == 1:
            response = f"VERDICT: VULNERABLE\nCWE: {row['cwe_string']}"
        else:
            response = "VERDICT: SAFE\nCWE: NONE"
        
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": f"Analyze this code for vulnerabilities:\n\n```\n{row['code']}\n```"},
            {"role": "assistant", "content": response}
        ]
        
        formatted_data.append({
            'messages': messages,
            'label': row['label']
        })
    
    return formatted_data

# ============================================
# UNSLOTH MODEL LOADING (KEY DIFFERENCE)
# ============================================

def load_unsloth_model(config):
    """
    Load Qwen2.5-Coder with Unsloth's optimized kernels
    This is 2-5x faster than standard transformers![citation:8]
    """
    print("\n" + "="*50)
    print("LOADING MODEL WITH UNSLOTH")
    print("="*50)
    
    # Load with Unsloth's FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.bnb_4bit_compute_dtype,  # bfloat16
        load_in_4bit=config.load_in_4bit,      # 4-bit quantization
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded: {config.model_name}")
    print(f"Max sequence length: {config.max_seq_length}")
    print(f"4-bit quantization: {config.load_in_4bit}")
    
    return model, tokenizer

# ============================================
# LORA SETUP WITH UNSLOTH
# ============================================

def setup_unsloth_lora(model, config):
    """
    Add LoRA adapters using Unsloth's optimized implementation
    Unsloth's LoRA is faster and uses less memory than PEFT[citation:3][citation:6]
    """
    print("\n" + "="*50)
    print("SETTING UP LORA (UNSLOTH OPTIMIZED)")
    print("="*50)
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total parameters: {total_params:,}")
    
    return model

# ============================================
# FORMAT DATA FOR UNSLOTH
# ============================================

def format_for_unsloth(samples, tokenizer, config):
    """Format messages using the tokenizer's chat template"""
    texts = []
    for sample in samples:
        text = tokenizer.apply_chat_template(
            sample['messages'],
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return texts

# ============================================
# TRAINING WITH TRL + UNSLOTH
# ============================================

def train_with_unsloth(model, tokenizer, train_data, val_data, config):
    """
    Train using SFTTrainer (optimized for Unsloth)
    Unsloth recommends TRL's SFTTrainer for best performance[citation:8]
    """
    print("\n" + "="*50)
    print("STARTING UNSLOTH TRAINING")
    print("="*50)
    
    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Format datasets
    def format_func(examples):
        texts = []
        for i in range(len(examples['messages'])):
            text = tokenizer.apply_chat_template(
                examples['messages'][i],
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}
    
    train_dataset = train_dataset.map(format_func, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(format_func, batched=True, remove_columns=val_dataset.column_names)
    
    # Training arguments
    training_args = SFTConfig(
        output_dir="./qwen-unsloth-titanvul",
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        bf16=is_bfloat16_supported(),
        dataloader_num_workers=4,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataset_text_field="text",
        max_length=config.max_seq_length,
        eos_token=tokenizer.eos_token,
    )
    
    # SFTTrainer (optimized for Unsloth)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )
    
    # Print training info
    effective_batch = config.batch_size * config.gradient_accumulation
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation}")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max sequence length: {config.max_seq_length}")
    print("DIAGNOSTIC: Type of training_args:", type(training_args))
    print("DIAGNOSTIC: training_args.eos_token:", getattr(training_args, "eos_token", None))
    from trl import SFTConfig as TRL_SFTConfig
    print("DIAGNOSTIC: isinstance(training_args, SFTConfig):", isinstance(training_args, TRL_SFTConfig))
    print("="*50)
    
    # Train
    trainer.train()
    
    return trainer

# ============================================
# INFERENCE WITH TRAINED MODEL
# ============================================

def predict_vulnerability(code_snippet, model, tokenizer, config):
    """Run inference using Unsloth's FastLanguageModel"""
    FastLanguageModel.for_inference(model)
    
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": f"Analyze this code for vulnerabilities:\n\n```\n{code_snippet}\n```"}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=128,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    is_vulnerable = "VULNERABLE" in response.upper()
    cwe_match = re.search(r'CWE[:\s]+(\w+)', response, re.IGNORECASE)
    cwe = cwe_match.group(1) if cwe_match else None
    
    return {
        'is_vulnerable': is_vulnerable,
        'response': response.strip(),
        'cwe': cwe
    }

# ============================================
# EVALUATION
# ============================================

def evaluate_model(model, tokenizer, test_data, config):
    """Evaluate on test set"""
    print("\n" + "="*50)
    print("EVALUATING MODEL")
    print("="*50)
    
    predictions = []
    labels = []
    
    for sample in tqdm(test_data, desc="Evaluating"):
        result = predict_vulnerability(sample['messages'][1]['content'].split("```\n")[1].split("\n```")[0], 
                                       model, tokenizer, config)
        predictions.append(1 if result['is_vulnerable'] else 0)
        labels.append(sample['label'])
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# ============================================
# SAVE AND EXPORT
# ============================================

def save_model(model, tokenizer, output_dir):
    """Save model in multiple formats"""
    print("\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)
    
    # Save LoRA adapter
    model.save_pretrained(f"{output_dir}/lora_adapter")
    tokenizer.save_pretrained(f"{output_dir}/lora_adapter")
    
    # Option 1: Save merged model in 16-bit
    print("Saving merged 16-bit model...")
    model.save_pretrained_merged(f"{output_dir}/merged_16bit", tokenizer, save_method="merged_16bit")
    
    # Option 2: Save as GGUF for ollama/llama.cpp (4-bit quantized)
    print("Saving GGUF 4-bit model for local inference...")
    model.save_pretrained_gguf(f"{output_dir}/gguf", tokenizer, quantization_method="q4_k_m")
    
    print(f"\nModels saved to {output_dir}")
    print("  - lora_adapter/ : LoRA weights only")
    print("  - merged_16bit/ : Full model in 16-bit")
    print("  - gguf/ : 4-bit quantized GGUF (for ollama/llama.cpp)")

# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()
    
    config = Config()
    
    if args.mode == "train":
        print("\n" + "="*60)
        print("QWEN2.5-CODER VULNERABILITY DETECTION WITH UNSLOTH")
        print("="*60)
        
        # Load and prepare data
        formatted_data = prepare_training_data(config)
        
        # Split data
        train_data, temp_data = train_test_split(
            formatted_data, test_size=0.3, random_state=42,
            stratify=[d['label'] for d in formatted_data]
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, random_state=42,
            stratify=[d['label'] for d in temp_data]
        )
        
        print(f"\nData split:")
        print(f"  Train: {len(train_data)}")
        print(f"  Validation: {len(val_data)}")
        print(f"  Test: {len(test_data)}")
        
        # Load Unsloth model (with 4-bit quantization)
        model, tokenizer = load_unsloth_model(config)
        model = setup_unsloth_lora(model, config)
        
        # Train
        trainer = train_with_unsloth(model, tokenizer, train_data, val_data, config)
        
        # Evaluate
        evaluate_model(model, tokenizer, test_data, config)
        
        # Save
        save_model(model, tokenizer, "./qwen-unsloth-titanvul")
        
        # Test inference
        print("\n" + "="*50)
        print("TESTING INFERENCE")
        print("="*50)
        
        test_code = """
        void vulnerable(char *input) {
            char buffer[10];
            strcpy(buffer, input);
        }
        """
        
        result = predict_vulnerability(test_code, model, tokenizer, config)
        print(f"Test input: Buffer overflow example")
        print(f"Prediction: {'VULNERABLE' if result['is_vulnerable'] else 'SAFE'}")
        print(f"Response: {result['response']}")
        
        print("\n✅ Training complete!")
        
    elif args.mode == "eval":
        # Load saved model
        print("Loading saved model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            args.model_path or "./qwen-unsloth-titanvul/merged_16bit",
            max_seq_length=Config.max_seq_length,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        
        # Interactive testing
        print("\nEnter code to analyze (type 'quit' to exit):")
        while True:
            print("\n" + "-"*40)
            code = input("Paste code snippet (or 'quit'): ")
            if code.lower() == 'quit':
                break
            if code.strip():
                result = predict_vulnerability(code, model, tokenizer, Config())
                print(f"\nResult: {'VULNERABLE' if result['is_vulnerable'] else 'SAFE'}")
                if result['cwe']:
                    print(f"CWE: {result['cwe']}")
                print(f"Full response:\n{result['response']}")

if __name__ == "__main__":
    main()
