import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


df = pd.read_csv('streamlit_model\diabetes.csv')
print(df.head(5))
print("\nMissing Values:\n", df.isnull().sum())
print("\nStats:\n", df.describe())
print("\nDuplicated values:\n",df.duplicated().sum())
l=df.columns
for i in l:
    print("valuecounts of",df[i].value_counts())

f = df.columns.tolist()
f.remove("Outcome")
print("Features:",f)

plt.figure(figsize=(15, 10))
df[f].boxplot()
plt.title("Boxplot of All Features", fontsize=16)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(16, 20))

for i, col in enumerate(f, 1):
    plt.subplot(4, 2, i)
    sns.boxplot(data=df, x="Outcome", y=col)
    plt.title(f"{col} vs Outcome", fontsize=12)

plt.tight_layout()
plt.show()


X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#since all are numeric values we are gouing to use standscaler and also there are no missing values so we are going with standardscaler instead of minmax scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


k_range = range(1, 40)
accuracies = []

for k in k_range:
    knn = KNeighborsClassifier(
        n_neighbors=k, 
        metric='minkowski', 
        p=1               # Manhattan distance
    )
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, pred))

best_k = accuracies.index(max(accuracies)) + 1
print("Best K =", best_k)
print("Best Accuracy with Manhattan =", max(accuracies))

knn = KNeighborsClassifier(
    n_neighbors=best_k, 
    metric='minkowski', 
    p=1,                # Manhattan distance
    weights='distance'  # Improve accuracy further
)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Final Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")