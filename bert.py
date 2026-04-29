from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import os
import matplotlib.pyplot as plt
from transformers import DataCollatorWithPadding
from pathlib import Path

# DMRS defense mechanism labels 0-8
NUM_LABELS = 9
LABEL_NAMES = [
    "No Defense",
    "Action",
    "Major Image-Distorting",
    "Disavowal",
    "Minor Image-Distorting",
    "Neurotic",
    "Obsessional",
    "High-Adaptive",
    "Needs More Info",
]
MODEL_NAME = "bert-base-uncased"

HERE = Path(__file__).parent
PSYDEFCONV_TRAIN = "https://huggingface.co/datasets/AIMH/PsyDefConv/resolve/main/Splits/train.json"
PSYDEFCONV_TEST  = "https://huggingface.co/datasets/AIMH/PsyDefConv/resolve/main/Splits/test.json"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True,
)
model.config.problem_type = "single_label_classification"

# All layers trainable — training from scratch
for param in model.parameters():
    param.requires_grad = True


def preprocess_function(examples):
    conversations = []
    for dialogue, current_text in zip(examples["dialogue"], examples["current_text"]):
        conversation = "\n".join(
            f"{turn['speaker']}: {turn['text']}"
            for turn in dialogue
        )
        conversations.append(f"{conversation}\nTarget utterance:\n{current_text}")
    return tokenizer(conversations, truncation=True)

# Load PsyDefConv train and test splits directly from HuggingFace
train_raw = load_dataset("json", data_files=PSYDEFCONV_TRAIN)["train"]
val_raw   = load_dataset("json", data_files=PSYDEFCONV_TEST)["train"]
train_raw = train_raw.rename_column("label", "labels")
val_raw   = val_raw.rename_column("label", "labels")

train_tokenized = train_raw.map(preprocess_function, batched=True)
val_tokenized   = val_raw.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
os.environ.setdefault("TENSORBOARD_LOGGING_DIR", "./logs")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(
        predictions=predictions,
        references=labels,
        average="macro",
    )
    recall = recall_metric.compute(
        predictions=predictions,
        references=labels,
        average="macro",
    )
    f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="macro",
    )
    return {
        "accuracy": acc["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }

lr = 2e-4
batch_size = 8
num_epochs = 5
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=lr,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
    dataloader_pin_memory=False,
    logging_strategy="epoch",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()

# Evaluation on validation split
output = trainer.predict(val_tokenized)
logits = output.predictions
y_pred = np.argmax(logits, axis=-1)
y_true = np.array(val_tokenized["labels"])

acc = accuracy_metric.compute(predictions=y_pred, references=y_true)
precision = precision_metric.compute(
    predictions=y_pred,
    references=y_true,
    average="macro",
)
recall = recall_metric.compute(
    predictions=y_pred,
    references=y_true,
    average="macro",
)
f1 = f1_metric.compute(
    predictions=y_pred,
    references=y_true,
    average="macro",
)

print("\nValidation Metrics:")
print(f"Accuracy : {acc['accuracy']:.4f}")
print(f"Precision: {precision['precision']:.4f}")
print(f"Recall   : {recall['recall']:.4f}")
print(f"F1-score : {f1['f1']:.4f}")

cm = np.zeros((NUM_LABELS, NUM_LABELS), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t][p] += 1

print("\nConfusion Matrix (rows=true, cols=predicted):")
print(cm)

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm, cmap="Blues")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_xticks(np.arange(NUM_LABELS))
ax.set_yticks(np.arange(NUM_LABELS))
ax.set_xticklabels(LABEL_NAMES, rotation=35, ha="right", fontsize=8)
ax.set_yticklabels(LABEL_NAMES, fontsize=8)
for i in range(NUM_LABELS):
    for j in range(NUM_LABELS):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
plt.title("Confusion Matrix – BERT fine-tuned on PsyDefConv")
plt.colorbar(im)
plt.tight_layout()
plot_path = HERE / "confusion_matrix.png"
plt.savefig(plot_path, dpi=150)
print(f"Confusion matrix saved to {plot_path}")
plt.show()
