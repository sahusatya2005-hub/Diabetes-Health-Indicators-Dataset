import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------
path = "/mnt/data/diabetes_dataset.csv"   # your uploaded file
df = pd.read_csv(path)

print("Shape:", df.shape)
print(df.head())

# ----------------------------------------------------------
# 2. BASIC EXPLORATION
# ----------------------------------------------------------
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

# ----------------------------------------------------------
# 3. SIMPLE VISUALIZATION
# ----------------------------------------------------------
plt.figure(figsize=(6,4))
df['Outcome'].value_counts().plot(kind='bar')
plt.title("Diabetes Outcome Distribution (0=No, 1=Yes)")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.show()

# ----------------------------------------------------------
# 4. TRAIN A MACHINE LEARNING MODEL
# ----------------------------------------------------------

# Assuming the dataset uses 'Outcome' as target column
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=150)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
