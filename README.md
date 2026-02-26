📂 Dataset Information

Dataset Name: Mall Customers Dataset

Total Samples: 200

Features:

CustomerID

Gender

Age

Annual Income (k$)

Spending Score (1–100)

For clustering, we used:

Annual Income (k$)

Spending Score (1–100)

1️⃣ Load and Visualize Dataset
Objective

Understand distribution of customers.

Visualization

Plotted:

Annual Income vs Spending Score

Observation:

Clear grouping patterns visible:

High income – high spending

High income – low spending

Low income – high spending

Low income – low spending

Average customers

Optional PCA was applied for 2D visualization.

2️⃣ Fit K-Means and Assign Cluster Labels
Objective

Group customers into clusters.

K-Means was applied with:

n_clusters = 5

Cluster labels (0–4) were assigned to each customer.

Each customer now belongs to a specific segment.

3️⃣ Elbow Method to Find Optimal K
Objective

Determine best number of clusters.

Method:

Calculated inertia (WCSS) for K = 1 to 10

Plotted K vs Inertia

The “elbow point” appeared at:

K ≈ 5

This indicates 5 clusters is optimal.

4️⃣ Visualize Clusters with Color-Coding

Clusters were plotted using different colors.

Each color represents a customer segment.

Typical segments:

High income – high spending

High income – low spending

Low income – high spending

Low income – low spending

Moderate income – moderate spending

This helps businesses identify customer behavior patterns.

5️⃣ Evaluate Using Silhouette Score

Silhouette Score measures clustering quality.

Range:

+1 → Well-separated clusters

0 → Overlapping clusters

-1 → Incorrect clustering

Result:

Silhouette Score ≈ 0.55

Interpretation:

Good cluster separation

Reasonably strong grouping
