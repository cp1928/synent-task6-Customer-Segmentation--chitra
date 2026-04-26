import streamlit as st
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🧠 Customer Segmentation App")
st.write("Group customers based on income and spending behavior")

st.markdown("""
### 📌 About This Project
This application uses **Machine Learning (K-Means Clustering)** to segment customers  
based on their **income and spending behavior**.

📈 Businesses can use these insights to:
- Identify high-value customers  
- Design targeted marketing strategies  
- Improve customer retention  
""")

# ---------------- LOAD DATA (FIXED PATH) ----------------
file_path = os.path.join("data", "Mall_Customers.csv")
df = pd.read_csv(file_path)

# ---------------- FEATURES ----------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- MODEL ----------------
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ---------------- CLUSTER LABELS ----------------
cluster_names = {
    0: "Average Customers",
    1: "Premium Customers 💎",
    2: "Impulse Buyers 🛍️",
    3: "Careful Customers 🤔",
    4: "Low Value Customers 💸"
}

df['Segment'] = df['Cluster'].map(cluster_names)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("Enter Customer Details")

with st.sidebar.form("input_form"):
    income = st.number_input("Annual Income (k$)", 0, 150, 50)
    spend_score = st.number_input("Spending Score (1-100)", 0, 100, 50)
    submit = st.form_submit_button("Predict")

# ---------------- PREDICTION ----------------
if submit:
    input_data = scaler.transform([[income, spend_score]])
    cluster = kmeans.predict(input_data)[0]

    st.subheader("📊 Prediction Result")
    st.success(f"🎯 This customer is classified as: {cluster_names[cluster]}")

    insights = {
        0: "Average customers. Maintain engagement with regular offers.",
        1: "High-value customers. Target VIP deals and luxury products.",
        2: "Impulse buyers. Use discounts and emotional marketing.",
        3: "Careful customers. Focus on value-based messaging.",
        4: "Low-value customers. Low marketing priority."
    }

    st.info(f"💡 Insight: {insights[cluster]}")

# ---------------- SUMMARY ----------------
st.write("")
st.subheader("📊 Cluster Summary")

summary = df.groupby('Segment')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
st.dataframe(summary)

# ---------------- VISUALIZATION (FIXED) ----------------
st.write("")
st.subheader("📊 Customer Segments Visualization")

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster'],
    cmap='tab10'
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Customer Segments")

# FIXED LEGEND (NO ERRORS)
for cluster_id, name in cluster_names.items():
    ax.scatter([], [], label=name)

ax.legend(title="Customer Segments")

st.pyplot(fig)

# ---------------- EVALUATION ----------------
st.write("")
st.subheader("📊 Model Evaluation")

sil_score = silhouette_score(X_scaled, df['Cluster'])
st.write(f"Silhouette Score: {sil_score:.2f}")

# ---------------- DATA VIEW ----------------
if st.checkbox("Show Dataset"):
    st.dataframe(df.head())

st.markdown("---")
st.write("Built with ❤️ using Streamlit & Machine Learning")