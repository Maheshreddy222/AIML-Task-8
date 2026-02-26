# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("Mall_Customers.csv")

# Select features (Income & Spending Score)
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# ==============================
# 2. Scale Features
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# 3. Elbow Method
# ==============================

inertia_values = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# ==============================
# 4. Plot Elbow Graph
# ==============================

plt.figure(figsize=(6,5))
plt.plot(K_range, inertia_values)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method for Optimal K")
plt.show()