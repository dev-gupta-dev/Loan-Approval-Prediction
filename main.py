import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample loan dataset
data = {
    "Income": [4000, 3000, 5000, 2000, 7000, 10000, 1500, 8000, 2500, 6000],
    "Credit_Score": [700, 650, 800, 580, 750, 900, 550, 780, 600, 720],
    "Loan_Amount": [100, 80, 200, 60, 250, 300, 50, 275, 90, 220],
    "Approved": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]   # 1 = Approved, 0 = Not Approved
}

df = pd.DataFrame(data)
print("📂 Loan Dataset:\n", df)
X = df[["Income", "Credit_Score", "Loan_Amount"]]
y = df["Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Example applicant: Income=5000, Credit Score=720, Loan Amount=180
new_applicant = np.array([[5000, 720, 180]])
prediction = model.predict(new_applicant)
probability = model.predict_proba(new_applicant)

print("\n🔍 Prediction:", "Approved ✅" if prediction[0] == 1 else "Not Approved ❌")
print("📌 Approval Probability:", probability[0][1])
