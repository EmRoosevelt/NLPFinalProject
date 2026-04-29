from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_recall_fscore_support
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
DISPLAY_LABEL_NAMES = [
    "0 No\nDefense",
    "1 Action",
    "2 Major\nImage-Distorting",
    "3 Disavowal",
    "4 Minor\nImage-Distorting",
    "5 Neurotic",
    "6 Obsessional",
    "7 High-\nAdaptive",
    "8 Needs More\nInfo",
]
MODEL_NAME = "bert-base-uncased"

HERE = Path(__file__).parent
PSYDEFCONV_TRAIN = "https://huggingface.co/datasets/AIMH/PsyDefConv/resolve/main/Splits/train.json"
PSYDEFCONV_TEST  = "https://huggingface.co/datasets/AIMH/PsyDefConv/resolve/main/Splits/test.json"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True,
)
model.config.problem_type = "single_label_classification"
model.config.pad_token_id = tokenizer.pad_token_id

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


# Inverse-frequency class weights help prevent majority-class collapse.
train_labels = np.array(train_raw["labels"])
label_counts = np.bincount(train_labels, minlength=NUM_LABELS)
class_weights = label_counts.sum() / (NUM_LABELS * np.maximum(label_counts, 1))
class_weights = torch.tensor(class_weights, dtype=torch.float)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


lr = 2e-5
batch_size = 8
num_epochs = 10

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=lr,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=1,
    dataloader_pin_memory=False,
    logging_strategy="epoch",
)
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    class_weights=class_weights,
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

per_prec, per_rec, per_f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=np.arange(NUM_LABELS),
    zero_division=0,
)
print("\nPer-class metrics:")
for idx, name in enumerate(LABEL_NAMES):
    print(
        f"{idx:>2} {name:<24} | P={per_prec[idx]:.3f} R={per_rec[idx]:.3f} F1={per_f1[idx]:.3f} N={support[idx]}"
    )

cm = np.zeros((NUM_LABELS, NUM_LABELS), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t][p] += 1

print("\nConfusion Matrix (rows=true, cols=predicted):")
print(cm)

fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(cm, cmap="Blues")
ax.set_aspect("equal")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_xticks(np.arange(NUM_LABELS))
ax.set_yticks(np.arange(NUM_LABELS))
ax.set_xticklabels(DISPLAY_LABEL_NAMES, rotation=0, ha="center", fontsize=9)
ax.set_yticklabels(DISPLAY_LABEL_NAMES, fontsize=9)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.tick_params(axis="x", pad=10)
ax.set_xticks(np.arange(-0.5, NUM_LABELS, 1), minor=True)
ax.set_yticks(np.arange(-0.5, NUM_LABELS, 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)
for i in range(NUM_LABELS):
    for j in range(NUM_LABELS):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
plt.title("Confusion Matrix – RoBERTa fine-tuned on PsyDefConv")
plt.colorbar(im)
plt.tight_layout()
plot_path = HERE / "confusion_matrix.png"
plt.savefig(plot_path, dpi=150)
print(f"Confusion matrix saved to {plot_path}")
plt.show()
