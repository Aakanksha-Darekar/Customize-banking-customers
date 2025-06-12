import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer

# Read dataset
file_path = 'C:/Users/admin/Downloads/Bank Customer Churn Prediction.csv'
data = pd.read_csv(file_path)

# Drop irrelevant columns
data = data.drop(columns=['customer_id'])

# Define features for clustering (excluding churn to use other attributes for segmentation)
X = data.drop(columns=['churn'])

# Identify categorical and numerical columns
categorical_features = ['country', 'gender']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Preprocessing: One-hot encode categorical features and scale numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Transform the dataset
X_processed = preprocessor.fit_transform(X)

# Apply clustering models
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_processed)

hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(X_processed)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_processed)

# Calculate silhouette scores
kmeans_score = silhouette_score(X_processed, kmeans_labels)
hierarchical_score = silhouette_score(X_processed, hierarchical_labels)
dbscan_score = silhouette_score(X_processed, dbscan_labels)

# Print scores
print(f'K-Means Silhouette Score: {kmeans_score:.3f}')
print(f'Hierarchical Clustering Silhouette Score: {hierarchical_score:.3f}')
print(f'DBSCAN Silhouette Score: {dbscan_score:.3f}')

# Compare clustering models
models = ['K-Means', 'Hierarchical', 'DBSCAN']
scores = [kmeans_score, hierarchical_score, dbscan_score]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=scores, palette='viridis')
plt.title('Clustering Model Comparison')
plt.ylabel('Silhouette Score')
plt.show()

# Define thresholds for different clusters
high_income_threshold = data['estimated_salary'].quantile(0.75)  # Top 25% earners
low_risk_threshold = data['credit_score'].quantile(0.75)  # Top 25% credit scores

low_balance_threshold = data['balance'].quantile(0.25)  # Bottom 25% balance
high_risk_threshold = data['credit_score'].quantile(0.25)  # Bottom 25% credit scores

# Filter for high-income and low-risk customers
high_income_low_risk = data[(data['estimated_salary'] >= high_income_threshold) & 
                            (data['credit_score'] >= low_risk_threshold)]

# Filter for low-balance and high-risk customers
low_balance_high_risk = data[(data['balance'] <= low_balance_threshold) & 
                             (data['credit_score'] <= high_risk_threshold)]

# Filter for moderate earners with average financial activity
moderate_earners = data[(data['estimated_salary'].between(data['estimated_salary'].quantile(0.4), 
                                                          data['estimated_salary'].quantile(0.6))) & 
                         (data['credit_score'].between(data['credit_score'].quantile(0.4), 
                                                       data['credit_score'].quantile(0.6)))]

# Plot clusters separately based on estimated_salary and balance
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

if not high_income_low_risk.empty:
    axes[0].scatter(high_income_low_risk['balance'], high_income_low_risk['estimated_salary'], alpha=0.5)
    axes[0].set_title('High Income & Low Risk')
    axes[0].set_xlabel('Balance')
    axes[0].set_ylabel('Estimated Salary')
    axes[0].grid(True)

if not low_balance_high_risk.empty:
    axes[1].scatter(low_balance_high_risk['balance'], low_balance_high_risk['estimated_salary'], alpha=0.5)
    axes[1].set_title('Low Balance & High Risk')
    axes[1].set_xlabel('Balance')
    axes[1].set_ylabel('Estimated Salary')
    axes[1].grid(True)

if not moderate_earners.empty:
    axes[2].scatter(moderate_earners['balance'], moderate_earners['estimated_salary'], alpha=0.5)
    axes[2].set_title('Moderate Earners & Avg Financial Activity')
    axes[2].set_xlabel('Balance')
    axes[2].set_ylabel('Estimated Salary')
    axes[2].grid(True)

plt.tight_layout()
plt.show()

# Visualizing all clusters together for each model
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X_processed[:, 0], X_processed[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.5)
axes[0].set_title('K-Means Clustering')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

axes[1].scatter(X_processed[:, 0], X_processed[:, 1], c=hierarchical_labels, cmap='viridis', alpha=0.5)
axes[1].set_title('Hierarchical Clustering')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')

axes[2].scatter(X_processed[:, 0], X_processed[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.5)
axes[2].set_title('DBSCAN Clustering')
axes[2].set_xlabel('Feature 1')
axes[2].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
