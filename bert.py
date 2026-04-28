from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import DataCollatorWithPadding
from pathlib import Path

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="C-L-V/PsyDefDetect_bert-base-uncased_merged_lr-6")
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("C-L-V/PsyDefDetect_bert-base-uncased_merged_lr-6")
model = AutoModelForSequenceClassification.from_pretrained("C-L-V/PsyDefDetect_bert-base-uncased_merged_lr-6")

for name, param in model.base_model.named_parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "classifier" in name:
        param.requires_grad = True

def preprocess_function(examples):
    conversations = []
    for dialogue, current_text in zip(examples["dialogue"], examples["current_text"]):
        conversation = "\n".join(
            f"{turn['speaker']}: {turn['text']}"
            for turn in dialogue
        )
        conversations.append(f"{conversation}\nTarget utterance:\n{current_text}")
    tokenized = tokenizer(conversations, truncation=True)
    return tokenized

# Load dataset from JSON file
dataset_path = Path(__file__).parent / "test_label.json"
dataset = load_dataset("json", data_files=str(dataset_path))
# Rename 'label' column to 'labels' for compatibility with transformers
dataset = dataset.rename_column("label", "labels")
# Split into train/validation sets (80/20 split)
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy_metric = evaluate.load("accuracy")
auc_score_metric = evaluate.load("roc_auc")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    try:
        auc = auc_score_metric.compute(prediction_scores=logits, references=labels, multi_class="ovr")["roc_auc"]
    except Exception:
        auc = 0.0
    return {"accuracy": acc["accuracy"], "roc_auc": auc}
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
    logging_dir="./logs",
    logging_strategy="epoch",
)