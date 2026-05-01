import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load model + columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction (Advanced)")
st.markdown("Predict churn and view insights in real-time.")

# Sidebar inputs
st.sidebar.header("Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}

# Build input dataframe (important: same order as training)
input_dict = {
    "tenure": tenure,
    "monthly_charges": monthly,
    "contract": contract_map[contract]
}

input_df = pd.DataFrame([input_dict])

# Ensure column order
input_df = input_df.reindex(columns=columns, fill_value=0)

# Predict
if st.button("Predict Churn"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    st.subheader("🔍 Prediction Result")

    if pred == 1:
        st.error(f"⚠️ Likely to churn ({prob[1]*100:.2f}%)")
        st.write("💡 Suggest: Offer discount / improve service")
    else:
        st.success(f"✅ Will stay ({prob[0]*100:.2f}%)")
        st.write("💡 Suggest: Maintain engagement")

# --- Feature Importance ---
st.markdown("---")
st.subheader("📊 Feature Importance")

importance = model.feature_importances_

fig, ax = plt.subplots()
ax.barh(columns, importance)
ax.set_title("Feature Importance")
st.pyplot(fig)

# --- Demo Confusion Matrix (dummy demo for UI) ---
st.markdown("---")
st.subheader("📉 Confusion Matrix (Demo)")

# (For real use, you’d compute from test data)
cm = np.array([[8, 2], [1, 9]])

fig2, ax2 = plt.subplots()
ax2.imshow(cm)
ax2.set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax2.text(j, i, cm[i, j], ha="center", va="center")
st.pyplot(fig2)

# Footer
st.markdown("---")
st.write("Built with Python + Streamlit")