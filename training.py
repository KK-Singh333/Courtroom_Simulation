from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling
import pandas as pd
import torch
import os

# -------- Prepare Dataset  ----------- #
def prepare_clm_dataset(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['text'])
    return Dataset.from_pandas(df[['text']])

def prepare_qna_dataset(file_path):
    df = pd.read_json(file_path, lines=True)
    return Dataset.from_pandas(df)

# -------- Tokenization -------- #
def tokenize_clm_data(dataset, tokenizer):
    return dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=1024), batched=True)

def format_messages(messages):
    user_message = next(m['content'] for m in messages if m['role'] == 'user')
    assistant_message = next(m['content'] for m in messages if m['role'] == 'assistant')

    return f"### Instruction:\n{user_message}\n\n### Response:\n{assistant_message}"

def tokenize_qna_data(dataset, tokenizer):
    def preprocess(example):
        text = format_messages(example['messages'])
        return tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512
        )

    return dataset.map(preprocess)

# -------- LoRA Configuration -------- #
def get_peft_model_from_base(base_model):
    lora_config = LoraConfig(
        r=1,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = prepare_model_for_kbit_training(base_model)
    return get_peft_model(model, lora_config)

# -------- Causal LM Training Function -------- #
def train_with_qlora(model_name, dataset, output_dir, epochs, use_sft=False, qna=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set up BitsAndBytesConfig for quantization and enable offloading
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,llm_int8_enable_fp32_cpu_offload=True)

    # Device Configuration
    device_map = 'auto'
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,  # Use custom device map for offloading
        torch_dtype=torch.float16 # Enable offloading to CPU with 32-bit precision
    )

    model = get_peft_model_from_base(base_model)

    # Tokenize dataset for QnA vs CLM
    if qna:
        tokenized_data = tokenize_qna_data(dataset, tokenizer)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM does not use MLM
            return_tensors='pt'
        )
    else:
        tokenized_data = tokenize_clm_data(dataset, tokenizer)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM does not use MLM
            return_tensors='pt'
        )

    # Modifying dataset for causal language modeling
    def add_labels(example):
        example["labels"] = example["input_ids"].copy()  # Shift the input_ids to create labels
        return example

    tokenized_data = tokenized_data.map(add_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        fp16=True,
        report_to="none",
        logging_dir=f"{output_dir}/logs",
        logging_first_step=True,
        logging_strategy="steps",
        disable_tqdm=False
    )

    # Trainer setup
    if use_sft:
        from trl import SFTTrainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=tokenized_data,
            data_collator=data_collator,
            args=training_args
        )
    else:
        trainer = Trainer(
            model=model,
            train_dataset=tokenized_data,
            data_collator=data_collator,
            args=training_args
        )

    trainer.train()

    #  LoRA Technique
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Merge LoRA into base model and save full model
    merged_model = model.merge_and_unload()
    full_model_dir = os.path.join(output_dir, "merged")
    merged_model.save_pretrained(full_model_dir)
    tokenizer.save_pretrained(full_model_dir)

    return output_dir


def main():
    model_name = "microsoft/Phi-3-mini-4k-instruct"

    Step 1 - Courtroom CLM fine-tuning
    clm_dataset = prepare_clm_dataset("/kaggle/input/my-data1234/data (1).csv")
    step1_model_path = train_with_qlora(model_name, clm_dataset, "/kaggle/working/phi3-legal-clm-qlora", epochs=4, use_sft=False, qna=False)

    step1_model_path='/kaggle/working/phi3-legal-clm-qlora'
    qna_dataset = prepare_qna_dataset("/kaggle/input/my-data1234/QnA.jsonl")
    train_with_qlora(step1_model_path, qna_dataset, "/kaggle/working/phi3-legal-qna-qlora", epochs=1, use_sft=True, qna=True)


main()
