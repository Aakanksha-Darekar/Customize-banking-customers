# 📊 Customer Churn Prediction and Segmentation Using Supervised & Unsupervised Learning

## 📌 Project Overview

This project explores both **supervised** and **unsupervised** machine learning techniques to analyze and model customer churn behavior and financial segmentation from a retail bank dataset. The goal is to determine the best model(s) for:

* Predicting customer churn
* Segmenting customers based on financial attributes for targeted business strategies

---

## 📂 Dataset Description

The dataset used is titled **Bank Customer Churn Prediction**, which includes the following key columns:

* **customer\_id** (dropped during preprocessing)
* **churn** *(target)* — Whether the customer has churned or not
* **credit\_score, age, tenure, balance, products\_number, estimated\_salary**
* **country, gender, credit\_card, active\_member**
  Access link : https://docs.google.com/spreadsheets/d/1Dg5xEfo-xy4uZpYlPTiqvS38kYMVzwgst-Fyj0mnv3A/edit?usp=sharing 

---

## 🔍 Supervised Learning Models

Supervised learning was applied to predict customer churn using the `churn` column as the target. Three neural architectures were explored:

### 1. Deep Neural Network (DNN)

* Dense layers with ReLU activation
* Dropout for regularization
* Sigmoid output for binary classification

### 2. Wide & Deep Model

* Combines linear and deep feature interactions
* Suitable for both memorization and generalization

### 3. TabTransformer

* Embedding + Transformer encoder for categorical data
* Captures feature interaction patterns in tabular format

#### 📈 Evaluation Metrics:

* Accuracy
* AUC (Area Under the ROC Curve)
* Classification report (Precision, Recall, F1-score)

**Result:**

> The **TabTransformer model** showed superior performance with the highest AUC and more balanced precision/recall across classes.

---

## 🔍 Unsupervised Learning Models

Unsupervised learning was used to segment customers based on financial and demographic features (excluding churn). Three clustering models were tested:

### 1. K-Means Clustering

* Simple centroid-based algorithm
* Best when clusters are spherical and evenly sized

### 2. Hierarchical Clustering

* Agglomerative strategy
* Provides a tree-based view (dendrogram)

### 3. DBSCAN

* Density-based clustering
* Handles noise and outliers

#### 📈 Evaluation Metric:

* **Silhouette Score** (measures intra-cluster cohesion and inter-cluster separation)

| Model        | Silhouette Score         |
| ------------ | ------------------------ |
| K-Means      | High (\~0.4–0.6)         |
| Hierarchical | Slightly lower           |
| DBSCAN       | Poor (many noise points) |

**Result:**

> **K-Means** performed best for customer segmentation with well-defined clusters and meaningful interpretation.

#### 🧠 Segment Definitions:

* **High Income & Low Risk:** Customers with high salary and credit scores
* **Low Balance & High Risk:** At-risk segment with low savings and poor credit
* **Moderate Earners:** Balanced financial behavior

---

## 🏆 Best Models to Use

| Task                      | Recommended Model  | Reason                                                           |
| ------------------------- | ------------------ | ---------------------------------------------------------------- |
| **Churn Prediction**      | TabTransformer     | Captures both categorical and numerical interactions effectively |
| **Customer Segmentation** | K-Means Clustering | Best silhouette score and interpretable segment clusters         |

---

## ⚙️ Technologies Used

* Python, TensorFlow, Keras
* Scikit-learn, Pandas, NumPy, Seaborn, Matplotlib
* Jupyter Notebook / Colab

---

## 📌 How to Run

1. Clone the repository or download the scripts.
2. Make sure the dataset is in the same directory or update the file path.
3. Run supervised model training using `supervised_models.py` or in notebook cells.
4. Run unsupervised analysis using `unsupervised_models.py`.
5. Visualizations and outputs are saved or plotted inline.
