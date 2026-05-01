import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load data
df = pd.read_csv("data.csv")

# Encode categorical (if your CSV uses text)
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
if df["contract"].dtype == "object":
    df["contract"] = df["contract"].map(contract_map)

# Features/target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model (tuned a bit)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate (console)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model + columns (important for app)
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(list(X.columns), open("columns.pkl", "wb"))

print("Model + columns saved!")