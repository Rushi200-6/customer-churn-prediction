# 📊 Customer Churn Prediction

## 🔗 Live Demo

👉 [https://your-app.streamlit.app](https://customer-churn-prediction-6y3rwkwrrzrjqcuvnvpcwv.streamlit.app/)

---

## 📌 Overview

Customer churn prediction is a machine learning project that identifies whether a customer is likely to leave a service. This helps companies take proactive actions to retain customers and reduce revenue loss.

---

## 🧠 Problem Statement

In competitive industries, retaining customers is more important than acquiring new ones.
The goal of this project is to build a machine learning model that predicts customer churn based on features like tenure, monthly charges, and contract type.

---

## ⚙️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit

---

## 📂 Project Structure

```
customer-churn-prediction/
│
├── app.py                # Streamlit app
├── train_model.py        # Model training script
├── eda.py                # Data analysis
├── data.csv              # Dataset
├── model.pkl             # Trained model
├── columns.pkl           # Feature columns
├── requirements.txt      # Dependencies
└── README.md
```

---

## 📊 Exploratory Data Analysis (EDA)

* Analyzed customer behavior using visualizations
* Identified patterns such as:

  * Customers with low tenure are more likely to churn
  * Higher monthly charges increase churn probability
  * Contract type significantly impacts churn

---

## 🤖 Model Development

* Implemented multiple models:

  * Logistic Regression
  * Decision Tree
  * Random Forest (selected as final model)

* Random Forest performed best due to:

  * High accuracy
  * Ability to capture complex patterns
  * Feature importance analysis

---

## 📈 Model Evaluation

* Accuracy
* Precision
* Recall
* F1-score

👉 Recall is prioritized to minimize loss from missed churn cases.

---

## 🔍 Feature Importance

Key factors affecting churn:

* Contract type (most important)
* Monthly charges
* Tenure

---

## 💻 Application

A Streamlit web application was developed to:

* Take user input
* Predict churn in real-time
* Display prediction probability
* Provide business recommendations

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

---

## 💡 Business Impact

* Helps identify high-risk customers
* Enables targeted retention strategies
* Reduces revenue loss

---

## 🚀 Future Improvements

* Use a larger real-world dataset (Telco dataset)
* Hyperparameter tuning
* Deploy with database integration
* Improve UI/UX

---

## 👨‍💻 Author

Rushikesh Patil

```
```
