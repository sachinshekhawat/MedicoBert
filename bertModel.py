import pandas as pd
import torch
import numpy as np
import random
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from evaluate import load  # ✅ Corrected import

# Set a random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ✅ Load datasets
train_csv = "/Users/home/Desktop/Dataset_Trial/train_dataset.csv"
val_csv = "/Users/home/Desktop/Dataset_Trial/val_dataset.csv"
test_csv = "/Users/home/Desktop/Dataset_Trial/test_dataset.csv"

df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)
df_test = pd.read_csv(test_csv)

# ✅ Ensure labels are mapped correctly
label_map = {"PRESCRIPTION": 0, "OBSERVATION": 1, "OTHER": 2}
df_train["label"] = df_train["LABEL"].map(label_map)
df_val["label"] = df_val["LABEL"].map(label_map)
df_test["label"] = df_test["LABEL"].map(label_map)

# ✅ Drop NaNs in labels (if any unmapped labels exist)
df_train = df_train.dropna(subset=["label"]).reset_index(drop=True)
df_val = df_val.dropna(subset=["label"]).reset_index(drop=True)
df_test = df_test.dropna(subset=["label"]).reset_index(drop=True)

df_train["label"] = df_train["label"].astype(int)
df_val["label"] = df_val["label"].astype(int)
df_test["label"] = df_test["label"].astype(int)

# ✅ Load tokenizer
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# ✅ Define Custom Dataset Class
class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# ✅ Create dataset objects
train_dataset = MedicalDataset(df_train["TEXT"], df_train["label"], tokenizer)
val_dataset = MedicalDataset(df_val["TEXT"], df_val["label"], tokenizer)
test_dataset = MedicalDataset(df_test["TEXT"], df_test["label"], tokenizer)

# ✅ Load BioBERT model
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# ✅ Define Accuracy Metric
metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# ✅ Training Arguments
training_args = TrainingArguments(
    output_dir="./bert_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6,  # ✅ Increased for better training
    logging_dir="./logs",
    logging_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # ✅ Corrected metric issue
    greater_is_better=True,
    seed=SEED
)

# ✅ Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,  # ✅ Fixed evaluation issue
)

# ✅ Train the model
trainer.train()

# ✅ Save trained model & tokenizer
model.save_pretrained("./bert_model")
tokenizer.save_pretrained("./bert_model")

print("\n🎉 Training complete! Model saved in './bert_model' 🚀")

