import pandas as pd

TRAIN_URL = "https://huggingface.co/datasets/AIMH/PsyDefConv/resolve/main/Splits/train.json"
TEST_URL = "https://huggingface.co/datasets/AIMH/PsyDefConv/resolve/main/Splits/test.json"

train_df = pd.read_json(TRAIN_URL)
test_df = pd.read_json(TEST_URL)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("\nTrain columns:")
print(train_df.columns.tolist())

print("\nFirst row:")
print(train_df.iloc[0])

print("\nLabel counts:")
print(train_df["label"].value_counts().sort_index())

train_df[["current_text", "label"]].to_csv("psydefconv_train.csv", index=False)
test_df[["current_text", "label"]].to_csv("psydefconv_test.csv", index=False)

print("\nSaved psydefconv_train.csv and psydefconv_test.csv")