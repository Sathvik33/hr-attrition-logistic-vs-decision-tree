import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os


base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "Data", "employee_attrition.csv")

df = pd.read_csv(data_path)

# Dataset Basic Info
print("\n Dataset Overview:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values Check:")
print(df.isnull().sum())
print("\n Target Variable Count (Attrition):")
print(df['Attrition'].value_counts())

# Feature Selection & Preprocessing
features = ['MonthlyIncome', 'Age', 'JobSatisfaction', 'YearsAtCompany', 'OverTime']
X = df[features].copy()
X['OverTime'] = X['OverTime'].map({'Yes': 1, 'No': 0})

y = df['Attrition'].map({'Yes': 1, 'No': 0})

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\nLogistic Regression Results")
print(classification_report(y_test, y_pred_lr))
lr_acc = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", lr_acc)

# Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\nDecision Tree Results")
print(classification_report(y_test, y_pred_dt))
dt_acc = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", dt_acc)

# Comparison Visualization
models = ['Logistic Regression', 'Decision Tree']
accuracies = [lr_acc, dt_acc]

plt.figure(figsize=(7, 5))
plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f"{acc:.2f}", ha='center', fontsize=12, fontweight='bold')
plt.show()
