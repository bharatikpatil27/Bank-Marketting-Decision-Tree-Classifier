import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Create output folder
os.makedirs("decision_tree_charts", exist_ok=True)

# Load dataset
df = pd.read_csv("bank-full.csv", sep=";")

print("DATASET LOADED SUCCESSFULLY!")
print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# Data cleaning
df.drop_duplicates(inplace=True)

# Encode categorical columns
df_encoded = pd.get_dummies(
    df,
    columns=df.select_dtypes(include="object").columns.drop("y"),
    drop_first=True
)

# Separate features and target
X = df_encoded.drop("y", axis=1)
y = df_encoded["y"].map({"yes": 1, "no": 0})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build Decision Tree Classifier
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:")
print(f"{accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("\nCONFUSION MATRIX:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save trained model
with open("decision_tree_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save predictions
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
results.to_csv("prediction_results.csv", index=False)

# Confusion matrix chart
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No", "Yes"],
    yticklabels=["No", "Yes"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("decision_tree_charts/confusion_matrix.png")
plt.close()

# Feature importance chart
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False).head(10)

plt.figure(figsize=(9, 5))
importance.sort_values().plot(kind="barh")
plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("decision_tree_charts/feature_importance.png")
plt.close()

# Decision tree visualization
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=7
)
plt.title("Bank Marketing Decision Tree")
plt.savefig("decision_tree_charts/decision_tree.png")
plt.close()

# Save model information report
with open("decision_tree_report.txt", "w") as file:
    file.write("BANK MARKETING - DECISION TREE CLASSIFICATION\n")
    file.write("=" * 50 + "\n\n")
    file.write(f"Total records: {len(df)}\n")
    file.write(f"Training records: {len(X_train)}\n")
    file.write(f"Testing records: {len(X_test)}\n")
    file.write(f"Model accuracy: {accuracy * 100:.2f}%\n\n")
    file.write("Classification Report:\n")
    file.write(classification_report(y_test, y_pred))
    file.write("\nConfusion Matrix:\n")
    file.write(str(cm))
    file.write("\n\nConclusion:\n")
    file.write(
        "A Decision Tree Classifier was trained to predict "
        "whether a bank customer would subscribe to a term deposit."
    )

print("\nDECISION TREE CLASSIFICATION COMPLETED SUCCESSFULLY!")
print("Model saved successfully.")
print("Prediction results saved.")
print("Charts generated successfully.")
print("Report generated successfully.")