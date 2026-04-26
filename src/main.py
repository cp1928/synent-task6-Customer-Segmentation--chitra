import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


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

df = pd.read_csv("..../data/Mall_Customers.csv")   
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_names = {
    0: "Average Customers",
    1: "Premium Customers 💎",
    2: "Impulse Buyers 🛍️",
    3: "Careful Customers 🤔",
    4: "Low Value Customers 💸"
}

df['Segment'] = df['Cluster'].map(cluster_names)

st.sidebar.header("Enter Customer Details")

with st.sidebar.form("input_form"):
    income = st.number_input("Annual Income (k$)", 0, 150, 50)
    spend_score = st.number_input("Spending Score", 0, 100, 50)

    submit = st.form_submit_button("Predict")

if submit:
    input_data = scaler.transform([[income, spend_score]])
    cluster = kmeans.predict(input_data)[0]

    st.write("")
    st.subheader("📊 Prediction Result")
    st.success(f"🎯 This customer is classified as: {cluster_names[cluster]}")

    insights = {
        0: "These customers are average. Maintain engagement with regular offers.",
        1: "High-value premium customers. Target with VIP deals and luxury products.",
        2: "Impulse buyers. Use discounts and emotional marketing strategies.",
        3: "Careful customers. Convince them with quality and value-based messaging.",
        4: "Low-value customers. Focus less marketing budget here."
    }

    st.info(f"💡 Insight: {insights[cluster]}")

st.write("")
st.subheader("📊 Cluster Summary")

summary = df.groupby('Segment')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
st.dataframe(summary)

st.write("")
st.subheader("📊 Customer Segments Visualization")

fig, ax = plt.subplots(figsize=(8, 5))

scatter = ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster'],
    cmap='tab10'
)

handles, _ = scatter.legend_elements()
labels = list(cluster_names.values())
ax.legend(handles, labels, title="Customer Segments")

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segments")

st.pyplot(fig)

st.write("")
st.subheader("📊 Model Evaluation")

sil_score = silhouette_score(X_scaled, df['Cluster'])
st.write(f"Silhouette Score: {sil_score:.2f}")

if st.checkbox("Show Dataset"):
    st.dataframe(df.head())

st.markdown("---")
st.write("Built with ❤️ using Streamlit & Machine Learning")