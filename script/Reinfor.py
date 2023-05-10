# %%
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.data import DataLoader, Dataset
import wandb
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import math


# %%
# Load the XSum dataset
dataset = load_dataset("xsum")

# Reduce dataset size
train_indices, _ = train_test_split(range(len(dataset['train'])), test_size=0.5, random_state=42)

# Initialize wandb
wandb.init(project="text_summarization", name="reinforce_summarization")

# %%
# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# %%
# Preprocess the dataset
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
# Hyperparameters
alpha = 0.9
gamma = 0.99
n_epochs = 3
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

rouge = Rouge()

# %%
# Calculate the Rouge-L score
def compute_rouge_l(pred_summary, ref_summary):
    scores = rouge.get_scores(pred_summary, ref_summary)
    return scores[0]["rouge-l"]["f"]

# %%
class CustomDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.tokenized_dataset["input_ids"][idx]),
            "attention_mask": torch.tensor(self.tokenized_dataset["attention_mask"][idx]),
            "labels": torch.tensor(self.tokenized_dataset["labels"][idx]),
        }
        return item

# %%
train_data = CustomDataset(tokenized_dataset["train"])
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Log hyperparameters
config = wandb.config
config.learning_rate = 1e-5
config.batch_size = batch_size
config.n_epochs = n_epochs

# %%
# Define a custom collate function
def collate_fn(batch):
    keys = batch[0].keys()
    batch_dict = {key: torch.stack([item[key] for item in batch]) for key in keys}
    return batch_dict

# Update the DataLoader with the custom collate function
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# %%
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# %%
print("Train dataset size:", len(reduced_dataset["train"]))
print("Validation dataset size:", len(reduced_dataset["validation"]))
print("Test dataset size:", len(reduced_dataset["test"]))


# %%
print("Tokenized train dataset size:", len(tokenized_dataset["train"]))
print("Tokenized validation dataset size:", len(tokenized_dataset["validation"]))
print("Tokenized test dataset size:", len(tokenized_dataset["test"]))


# %%
half_epoch_batches = math.ceil(len(train_dataloader) / 2)

for epoch in range(n_epochs):
    epoch_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}")
    for i, batch in enumerate(progress_bar):
        model.train()
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Generate summaries
        with torch.no_grad():
            summary_ids = model.generate(input_ids, attention_mask=attention_mask, num_beams=4, max_length=150, early_stopping=True)
        pred_summaries = [tokenizer.decode(s, skip_special_tokens=True) for s in summary_ids]
        ref_summaries = [tokenizer.decode(s, skip_special_tokens=True) for s in labels]

        # Calculate rewards
        rewards = []
        for pred_summary, ref_summary in zip(pred_summaries, ref_summaries):
            reward = compute_rouge_l(pred_summary, ref_summary)
            rewards.append(reward)

        rewards = torch.tensor(rewards).to(device)

        # Pad summary_ids to match the sequence length of logits
        logits = model(input_ids, attention_mask=attention_mask, decoder_input_ids=summary_ids).logits
        max_seq_len = logits.size(1)
        padded_summary_ids = torch.zeros(batch_size, max_seq_len).long().to(device)
        padded_summary_ids[:, :summary_ids.size(1)] = summary_ids

        # Compute the policy gradient loss
        log_probs = torch.gather(logits.view(-1, logits.size(-1)), 1, padded_summary_ids.view(-1, 1)).view(batch_size, -1)
        loss = -(rewards.view(-1, 1) * log_probs).sum() / batch_size

        # Backpropagate the loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"batch_loss": loss.item()})

         # Save checkpoint every 0.5 epoch
        if (i + 1) == half_epoch_batches or (i + 1) == len(train_dataloader):
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}_batch_{i + 1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at: {checkpoint_path}")

        # Update the progress bar
        progress_bar.update(1)

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{n_epochs} | Loss: {avg_epoch_loss}")

    # Log the average loss for the current epoch
    wandb.log({"epoch": epoch + 1, "loss": avg_epoch_loss})

# %%
# Evaluate the model
def generate_summary(text):
    input_text = "summarize: " + text
    inputs = tokenizer(input_text, max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        summary_ids = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], num_beams=4, max_length=150, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

rouge = Rouge()
test_dataset = tokenized_dataset["test"]
predictions = []
references = []

for example in test_dataset:
    article = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
    pred_summary = generate_summary(article)
    ref_summary = tokenizer.decode(example["labels"], skip_special_tokens=True)

    predictions.append(pred_summary)
    references.append(ref_summary)

rouge_scores = rouge.compute(predictions=predictions, references=references, rouge_types=["rouge1", "rouge2", "rougeL"])

print("Rouge Scores:", rouge_scores)

# Calculate BLEU scores
smooth = SmoothingFunction().method1
bleu_scores = [sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth) for ref, pred in zip(references, predictions)]
avg_bleu = np.mean(bleu_scores)

print("Average BLEU Score:", avg_bleu)

# Log Rouge and BLEU scores to WandB
wandb.log({"rouge1": rouge_scores['rouge1'].mid.fmeasure,
           "rouge2": rouge_scores['rouge2'].mid.fmeasure,
           "rougeL": rouge_scores['rougeL'].mid.fmeasure,
           "avg_bleu": avg_bleu})


# Close the wandb run
wandb.finish()


