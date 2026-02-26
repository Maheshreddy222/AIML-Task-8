# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("Mall_Customers.csv")

# ==============================
# 2. Select Features for Clustering
# (Commonly Income & Spending Score)
# ==============================

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# ==============================
# 3. Scale Features
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# 4. Fit K-Means
# ==============================

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# ==============================
# 5. Assign Cluster Labels
# ==============================

df["Cluster"] = kmeans.labels_

print("First 5 rows with Cluster Labels:")
print(df.head())

# ==============================
# 6. Visualize Clusters
# ==============================

plt.figure(figsize=(6,5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df["Cluster"])
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.title("K-Means Clustering")
plt.show()