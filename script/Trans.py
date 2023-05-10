# %%
import torch
from datasets import load_dataset, load_metric
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, TrainingArguments, Trainer
import wandb
import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.model_selection import train_test_split


# %%
# Load the XSum dataset
dataset = load_dataset("xsum")

# Reduce dataset size
train_indices, test_data = train_test_split(range(len(dataset['train'])), test_size=0.5, random_state=42)

# %%
wandb.init(project="text_summarization", name="transformer_summarization")

# %%
# Set the model name and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)

# %%
def preprocess_batch(batch):
    input_texts = ["summarize: " + doc for doc in batch["document"]]
    target_texts = batch["summary"]

    source = tokenizer(input_texts, max_length=512, truncation=True, padding='max_length', return_tensors="pt")
    target = tokenizer(target_texts, max_length=150, truncation=True, padding='max_length', return_tensors="pt")

    return {
        "input_ids": source["input_ids"].tolist(),
        "attention_mask": source["attention_mask"].tolist(),
        "labels": target["input_ids"].tolist(),
    }

# Replace the original train dataset with the reduced dataset
reduced_dataset = {}
reduced_dataset["train"] = dataset["train"].select(train_indices)
reduced_dataset["validation"] = dataset["validation"]
reduced_dataset["test"] = dataset["test"]

# Preprocess the data
tokenized_dataset = {}
for split in reduced_dataset:
    tokenized_dataset[split] = reduced_dataset[split].map(preprocess_batch, remove_columns=["document", "summary"], batched=True, batch_size=16)

train_dataset = tokenized_dataset["train"]

# %%
# Load the T5 model
config = T5Config.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)


# %%
# Set up training and evaluation arguments
num_train_examples = len(train_dataset)
train_batch_size = 8
steps_per_epoch = len(reduced_dataset["train"]) // train_batch_size
save_steps = int(steps_per_epoch * 0.1)

# %%
training_args = TrainingArguments(
    output_dir="./t5_xsum",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=1275,  # Change the eval_steps to be a multiple of save_steps
    logging_dir="./logs",
    weight_decay=0.01,
    save_steps=save_steps,
    save_total_limit=3,  # Limit the number of saved checkpoints
    load_best_model_at_end=True,
    report_to="wandb",
)

# %%
# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)


# %%
import os

checkpoint_dir = "./t5_xsum"

# Get all checkpoint filenames
checkpoint_filenames = os.listdir(checkpoint_dir)

# Filter filenames to get only checkpoint files
checkpoint_filenames = [filename for filename in checkpoint_filenames if "checkpoint" in filename]

# Sort filenames by epoch number
checkpoint_filenames = sorted(checkpoint_filenames, key=lambda x: float(x.split("-")[-1]))

# Print the filename with the largest epoch number
print(checkpoint_filenames[-1])


# %%
# Checkpoint load
checkpoint_path = "./t5_xsum/checkpoint-3825"

# Check if the checkpoint is loadable
try:
    model_test = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
    print("Checkpoint is loadable.")
except Exception as e:
    print("Error loading checkpoint:", e)

# %%
# Fine-tune the model
trainer.train("./t5_xsum/checkpoint-36975")

# %%
print(trainer.state)


# %%
# Save the fine-tuned model
trainer.save_model("./t5_xsum_finetuned")

# %%
from tqdm import tqdm

# Reduce the test dataset to 0.02
_, reduced_test_indices = train_test_split(test_data, test_size=0.02, random_state=42)
reduced_test_data = [dataset["train"][i] for i in reduced_test_indices]

def generate_summary_transformer(article):
    model = T5ForConditionalGeneration.from_pretrained('./t5_xsum_finetuned')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    inputs = tokenizer.encode("summarize: " + article, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, num_return_sequences=1, max_length=150, no_repeat_ngram_size=2, min_length=30, early_stopping=True)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def evaluate_model(generate_summary_function, reduced_test_data):
    rouge = Rouge()
    bleu_score = 0
    predictions = []
    references = []

    for example in reduced_test_data:
        article = example["document"]
        summary = generate_summary_function(article)
        predictions.append(summary)
        references.append(example["summary"])

    rouge_scores = rouge.get_scores(predictions, references, avg=True)

    # Compute BLEU score
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        bleu_score += sentence_bleu([ref_tokens], pred_tokens, smoothing_function=SmoothingFunction().method1)

    bleu_score = bleu_score / len(predictions)

    return rouge_scores, bleu_score


# Evaluate the Transformer-based model
transformer_rouge_scores, transformer_bleu_score = evaluate_model(generate_summary_transformer, reduced_test_data)
print("Transformer Rouge Scores:", transformer_rouge_scores)
print("Transformer BLEU Score:", transformer_bleu_score)

# Log Rouge and BLEU scores to WandB
wandb.log({"rouge1": transformer_rouge_scores['rouge-1']['f'],
           "rouge2": transformer_rouge_scores['rouge-2']['f'],
           "rougeL": transformer_rouge_scores['rouge-l']['f'],
           "avg_bleu": transformer_bleu_score})

# Finish the run
wandb.finish()


