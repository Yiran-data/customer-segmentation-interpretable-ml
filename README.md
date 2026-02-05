# Customer-segmentation-interpretable-ml
An end-to-end customer segmentation project using K-means, decision tree surrogate models, A/B testing simulation, and random forest feature importance for interpretability. 

# Customer Segmentation with Interpretable Machine Learning

This project demonstrates an end-to-end customer segmentation workflow, with a strong focus on **interpretability** and **decision-oriented analysis** rather than pure predictive performance.

Using a small, well-known customer dataset, the project showcases how unsupervised clustering results can be explained, validated, and operationalized using interpretable machine learning models and experimental thinking.

---

## Project Overview

The project follows four main stages:

1. **Exploratory Data Analysis (EDA)**  
2. **Customer Segmentation using K-Means**  
3. **Interpretability via Surrogate Models (Decision Tree & Random Forest)**  
4. **Segment-Aware A/B Testing (Simulated)**  

Although the dataset is limited in size, the goal is **methodological**:  
to demonstrate how clustering results can support **business understanding and decision-making**.

---

## Repository Structure

```text
customer-segmentation-interpretable-ml/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_kmeans_segmentation.ipynb
│   ├── 03_models_dt_rf.ipynb
│   └── 04_ab_test.ipynb
├── Mall_Customers.csv
└── README.md
```

## Dataset

- **Source**: Mall Customers dataset (public Kaggle dataset)  
- **Number of customers**: 200  

### Features used for segmentation
- Age  
- Annual Income (k$)  
- Spending Score (1–100)  

The dataset contains no missing values and all features are numerical, making it suitable for distance-based clustering methods.

## Methodology

### 1️⃣ Exploratory Data Analysis (EDA)

Basic descriptive statistics are used to understand feature distributions and value ranges.  
Given the small sample size, EDA is intentionally concise and focuses on identifying overall patterns rather than exhaustive visualization.

### 2️⃣ K-Means Customer Segmentation

K-means clustering is applied to segment customers based on spending behavior and income-related features.  
The number of clusters is selected using standard evaluation techniques such as the elbow method and silhouette analysis.

The resulting clusters represent distinct customer profiles with different consumption patterns.

### 3️⃣ Model Interpretability via Surrogate Models

To improve interpretability of the unsupervised clustering results, a decision tree surrogate model is trained to approximate the K-means cluster assignments.

This approach provides human-readable rules that explain how customers are assigned to different segments, making the results more accessible for non-technical stakeholders.

### 4️⃣ Feature Importance Analysis

A random forest classifier is trained using cluster labels as pseudo-targets to estimate feature importance.  
This step helps quantify the relative contribution of each feature to the segmentation outcome and validates whether the clustering aligns with intuitive business reasoning.

### 5️⃣ A/B Testing Simulation

A simplified A/B testing simulation is conducted to demonstrate how customer segments could be used in downstream decision-making scenarios, such as targeted marketing strategies.

This section illustrates how segmentation results may translate into measurable business impact.

## Key Insights

- Annual income is the primary driver of customer segmentation.
- Spending score further differentiates behavior within income groups.
- Age acts as a secondary refinement factor.
- Interpretable models align closely with visual cluster structures.
- Segment-level experimentation highlights how clustering can support targeted decisions.

## Notes & Limitations

- The dataset is small and used for demonstration purposes only.
- The A/B test is simulated and does not represent real experimental data.
- Results should be interpreted as methodological examples rather than business conclusions.

## Tools & Libraries

- Python  
- pandas, numpy  
- scikit-learn  
- matplotlib, seaborn  
- scipy

## About This Project

This project was created as a portfolio example to demonstrate:

- customer segmentation logic,
- model interpretability techniques,
- and experiment-driven analytical thinking.

The focus is on building an end-to-end, interpretable workflow rather than optimizing predictive performance.

