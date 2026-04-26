# 🧠 Customer Segmentation using Machine Learning

## 📌 Problem Statement
Businesses often struggle to understand different types of customers and their purchasing behavior.  
This project segments customers into distinct groups based on their **income** and **spending patterns** using machine learning, enabling more targeted and effective marketing strategies.

---

## 📊 Dataset Details
- Source: Mall Customers Dataset (Kaggle)  
- Link: https://www.kaggle.com/datasets/shwetabh123/mall-customers  

- Features used:
  - Annual Income (k$)
  - Spending Score (1–100)

- Total Records: ~200 customers  


---

## ⚙️ Approach

1. **Data Preprocessing**
   - Selected relevant features (Income & Spending Score)
   - Applied feature scaling using StandardScaler

2. **Clustering Model**
   - Implemented **K-Means Clustering**
   - Used Elbow Method to determine optimal number of clusters (k = 5)

3. **Evaluation**
   - Used **Silhouette Score** to evaluate clustering performance

4. **Deployment**
   - Built an interactive **Streamlit web app**
   - Users input values → model assigns customer segment in real time

---

## 📈 Results

The model segmented customers into 5 distinct groups:

| Segment | Description |
|--------|------------|
| 💎 Premium Customers | High income, high spending |
| 🛍️ Impulse Buyers | Low income, high spending |
| 🤔 Careful Customers | High income, low spending |
| 💸 Low Value Customers | Low income, low spending |
| 🟡 Average Customers | Moderate income & spending |

---

## 💡 Key Insights
- Premium customers should be targeted with **VIP offers and loyalty programs**
- Impulse buyers respond well to **discounts and promotional campaigns**
- Careful customers require **value-based and trust-driven marketing**
- Low-value customers need **minimal marketing investment**

---

## 🖥️ Application Features
- Interactive user input (income & spending score)  
- Real-time customer segmentation  
- Visual representation of clusters  
- Actionable business insights  

---

## 🚀 How to Run

```bash
git clone <your-repo-link>
cd customer-segmentation
pip install -r requirements.txt
streamlit run src/main.py