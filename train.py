import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Tokenize text
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation = True,
        padding = "max_length",
        max_length = 128,
    )

# Memuat data
df = pd.read_csv("data.csv")
# Rename label
df = df.rename(columns={"Teks": "text", "label": "label"})
# Menghapus data kosong
df = df.dropna(subset=["text"])

# Prepare HF Dataset format
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df),
})

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Build IndoBERT classifier
num_labels = 3

model = AutoModelForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", num_labels=num_labels)

# Train
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

training_args = TrainingArguments(
    output_dir="indoBERT-sms",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size= 8,
    per_device_eval_batch_size = 8,
    learning_rate = 2e-5,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized["train"],
    eval_dataset = tokenized["test"],
    compute_metrics = compute_metrics,
)

trainer.train()

# Prediction
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation = True, padding = "max_length", max_length = 128)
    outputs = model(**inputs)
    pred = outputs.logits.argmax(dim = 1).item()
    return pred


trainer.save_model("indoBERT-sms")
tokenizer.save_pretrained("indoBERT-sms")