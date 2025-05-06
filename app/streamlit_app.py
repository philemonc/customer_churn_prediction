import altair as alt
import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

st.set_page_config(layout="centered")
"""
# Customer Churn Prediction Model!

"""
st.write("You are working for ABC Multistate bank and the Data Science Division has asked you to come up with a machine learning model to predict customer churn.")
st.write("You are given the following dataset with the fields below:")

st.markdown("1. customer_id - unique id")
st.markdown("2. credit_score")
st.markdown("3. country")
st.markdown("4. gender")
st.markdown("5. age")
st.markdown("6. tenure")
st.markdown("7. balance")
st.markdown("8. products_number")
st.markdown("9. credit_card")
st.markdown("10. active_member")
st.markdown("11. estimated_salary")
st.markdown("12. churn - 0 for non-churn, 1 for churn")
st.write("Dataset is from https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset")

# 1. show dataframe
st.subheader("1. Load Data")
st.markdown("Let's load the dataset using pandas")

df = pd.read_csv("bank_customer_churn.csv")
#st.dataframe(df.head())

st.subheader("2. Perform Exploratory Data Analysis (EDA)")

st.markdown("Let's perform EDA for the numerical fields: credit score, age, tenure, balance, estimated salary and product number")

# select numerical columns
numerical_columns = ['credit_score', 'age', 'tenure', 'balance','estimated_salary','products_number']

st.subheader("2.1 Visualisations for Numerical Columns")

# Loop through the columns in 2 rows with 3 columns each
for col_name in numerical_columns:
    st.subheader(f"Interactive Histogram for {col_name.replace('_', ' ').title()}")

    # Add slider for number of bins
    bins = st.slider(f"Select number of bins for {col_name.replace('_', ' ').title()}", min_value=10, max_value=100, value=30)

    # Create the Matplotlib histogram
    fig, ax = plt.subplots()
    ax.hist(df[col_name], bins=bins, color='skyblue', edgecolor='black')
    ax.set_title(f"Distribution of {col_name.replace('_', ' ').title()}")
    ax.set_xlabel(col_name.replace('_', ' ').title())
    ax.set_ylabel('Count')

    # Display the Matplotlib chart in Streamlit
    st.pyplot(fig)

st.subheader("2.2 Visualisations for Categorical Columns")
categorical_columns = ['gender', 'product_type', 'country']

# Loop through categorical columns and generate bar charts
categorical_columns = ['country','gender', 'credit_card', 'active_member']
for col in categorical_columns:
    st.subheader(f"Bar chart for {col.replace('_', ' ').title()}")

    # Count the occurrences of each category
    category_counts = df[col].value_counts()

    # Create a bar chart using Streamlit
    st.bar_chart(category_counts)

st.subheader("3. Run Machine Learning Model (XGBoost")

st.write("Train the XGBoost model!")
numerical_columns = ['credit_score', 'age', 'tenure', 'balance','estimated_salary','products_number']
categorical_columns = ['country','gender', 'credit_card', 'active_member']
target = ['churn']
selected_cols = numerical_columns + categorical_columns

X = df[selected_cols]
y = df[target]

label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X.loc[:, col] = le.fit_transform(X[col])

X['country'] = X['country'].astype(str).astype('int64')
X['gender'] = X['gender'].astype(str).astype('int64')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if st.button("Train the Machine Learning Model! (XGBoost)"):
    model = xgb.XGBClassifier(eval_metric='logloss', enable_categorical=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    st.success(f"Model trained! Accuracy: {acc:.4f}")

    # --- ROC Curve ---
    st.subheader("4. Model Evaluation - ROC")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax_roc.plot([0, 1], [0, 1], "k--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Receiver Operating Characteristic")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # --- Precision-Recall Curve ---
    st.subheader("5. Precision Recall - PR Curve")
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(recall, precision, label=f"Average Precision = {ap_score:.2f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend(loc="upper right")
    st.pyplot(fig_pr)


    # --- Feature Importance ---
    st.subheader("6. Feature importances")
    st.write("These are the top features that affected the model prediction:")
    importances_fig, importances_ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model, ax=importances_ax, height=0.5, importance_type='gain', show_values=False)
    importances_ax.set_title("XGBoost Feature Importance (by Gain)")
    st.pyplot(importances_fig)
   