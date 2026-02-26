# Import libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("Mall_Customers.csv")

# Select features for clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# ==============================
# 2. Scale Features
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# 3. Apply K-Means
# ==============================

kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# ==============================
# 4. Calculate Silhouette Score
# ==============================

score = silhouette_score(X_scaled, labels)

print("Silhouette Score:", score)