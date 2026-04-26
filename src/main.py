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
based on their income and spending behavior.

📈 Businesses can use these insights to:
- Identify high-value customers  
- Design targeted marketing strategies  
- Improve customer retention  
""")

# ---------------- LOAD DATA ----------------
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
    1: "Premium Customers",
    2: "Impulse Buyers",
    3: "Careful Customers",
    4: "Low Value Customers"
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
    input_data = np.array([[income, spend_score]])
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]

    st.subheader("📊 Prediction Result")
    st.success(f"🎯 Customer Segment: {cluster_names[cluster]}")

    insights = {
        0: "Average customers. Maintain engagement.",
        1: "High-value customers. Target premium offers.",
        2: "Impulse buyers. Use discounts.",
        3: "Careful customers. Focus on value.",
        4: "Low-value customers. Low priority."
    }

    st.info(f"💡 Insight: {insights[cluster]}")

# ---------------- SUMMARY ----------------
st.write("")
st.subheader("📊 Cluster Summary")

summary = df.groupby('Segment')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
st.dataframe(summary)

# ---------------- VISUALIZATION ----------------
st.write("")
st.subheader("📊 Customer Segments Visualization")

fig, ax = plt.subplots(figsize=(8, 5))

# Existing data points (clusters)
ax.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=df['Cluster'],
    cmap='tab10',
    alpha=0.6
)

# Labels
ax.set_xlabel("Income (scaled)")
ax.set_ylabel("Spending Score (scaled)")
ax.set_title("Customer Segments")

# Plot cluster legends
for i, name in cluster_names.items():
    ax.scatter([], [], label=name)

# ---------------- IMPORTANT PART ----------------
# Plot user input point (ONLY when user clicks predict)
if submit:
    input_scaled = scaler.transform(np.array([[income, spend_score]]))

    ax.scatter(
        input_scaled[0][0],
        input_scaled[0][1],
        color='black',
        s=200,
        marker='X',
        label='Your Input'
    )

ax.legend()

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