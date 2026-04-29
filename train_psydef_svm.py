import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

train_df = pd.read_csv("psydefconv_train.csv")
test_df = pd.read_csv("psydefconv_test.csv")

X_train = train_df["current_text"]
y_train = train_df["label"]

X_test = test_df["current_text"]
y_test = test_df["label"]

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LinearSVC()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("PSYDEF SVM RESULTS\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))