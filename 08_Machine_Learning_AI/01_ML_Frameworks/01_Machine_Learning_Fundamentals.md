# Machine Learning Fundamentals

*Duration: 2-3 weeks*

## Basic Concepts

### Supervised vs. Unsupervised Learning

#### Supervised Learning
In supervised learning, we train models using labeled data - input-output pairs where we know the correct answer.

**Key Characteristics:**
- Has labeled training data (X, y pairs)
- Goal: Learn a mapping function f: X → y
- Performance can be measured against known correct answers
- Examples: Email spam detection, image classification, stock price prediction

**Types of Supervised Learning:**

**1. Classification** - Predicting discrete categories/classes
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Generate sample classification data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                         n_informative=2, n_clusters_per_class=1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy:.3f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# Visualize decision boundary
def plot_decision_boundary(X, y, model, title):
    plt.figure(figsize=(10, 8))
    
    # Create a mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Make predictions on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

plot_decision_boundary(X_test, y_test, classifier, 'Classification Decision Boundary')
```

**2. Regression** - Predicting continuous numerical values
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Generate sample regression data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Train polynomial regression (degree 3)
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Make predictions
y_pred_linear = linear_reg.predict(X_test)
y_pred_poly = poly_reg.predict(X_test_poly)

# Evaluate models
mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_linear = r2_score(y_test, y_pred_linear)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Linear Regression - MSE: {mse_linear:.3f}, R²: {r2_linear:.3f}")
print(f"Polynomial Regression - MSE: {mse_poly:.3f}, R²: {r2_poly:.3f}")

# Visualize results
plt.figure(figsize=(15, 5))

# Plot linear regression
plt.subplot(1, 3, 1)
plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual')
plt.scatter(X_test, y_pred_linear, color='red', alpha=0.6, label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression')
plt.legend()

# Plot polynomial regression
plt.subplot(1, 3, 2)
plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual')
plt.scatter(X_test, y_pred_poly, color='red', alpha=0.6, label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Polynomial Regression')
plt.legend()

# Plot residuals
plt.subplot(1, 3, 3)
plt.scatter(y_pred_linear, y_test - y_pred_linear, alpha=0.6, label='Linear')
plt.scatter(y_pred_poly, y_test - y_pred_poly, alpha=0.6, label='Polynomial')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.legend()

plt.tight_layout()
plt.show()
```

#### Unsupervised Learning
In unsupervised learning, we work with data that has no labeled outcomes. The goal is to discover hidden patterns or structures.

**Key Characteristics:**
- No labeled data - only input features X
- Goal: Find hidden patterns, structures, or representations
- More exploratory in nature
- Harder to evaluate since there's no "ground truth"

**Types of Unsupervised Learning:**

**1. Clustering** - Grouping similar data points
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Generate sample clustering data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Apply different clustering algorithms
kmeans = KMeans(n_clusters=4, random_state=42)
dbscan = DBSCAN(eps=0.5, min_samples=5)
hierarchical = AgglomerativeClustering(n_clusters=4)

# Fit and predict
y_kmeans = kmeans.fit_predict(X)
y_dbscan = dbscan.fit_predict(X)
y_hierarchical = hierarchical.fit_predict(X)

# Evaluate clustering quality
def evaluate_clustering(X, y_pred, y_true, algorithm_name):
    ari = adjusted_rand_score(y_true, y_pred)
    silhouette = silhouette_score(X, y_pred)
    print(f"{algorithm_name}:")
    print(f"  Adjusted Rand Index: {ari:.3f}")
    print(f"  Silhouette Score: {silhouette:.3f}")
    return ari, silhouette

print("Clustering Evaluation:")
evaluate_clustering(X, y_kmeans, y_true, "K-Means")
evaluate_clustering(X, y_dbscan, y_true, "DBSCAN")
evaluate_clustering(X, y_hierarchical, y_true, "Hierarchical")

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Original data
axes[0, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
axes[0, 0].set_title('Original Data (True Clusters)')

# K-Means
axes[0, 1].scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6)
axes[0, 1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                  marker='x', s=200, c='red', linewidth=3)
axes[0, 1].set_title('K-Means Clustering')

# DBSCAN
axes[1, 0].scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='viridis', alpha=0.6)
axes[1, 0].set_title('DBSCAN Clustering')

# Hierarchical
axes[1, 1].scatter(X[:, 0], X[:, 1], c=y_hierarchical, cmap='viridis', alpha=0.6)
axes[1, 1].set_title('Hierarchical Clustering')

plt.tight_layout()
plt.show()
```

**2. Dimensionality Reduction** - Reducing the number of features while preserving information
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# Load high-dimensional data (digits dataset: 64 features)
digits = load_digits()
X, y = digits.data, digits.target

print(f"Original data shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original data (first two features)
scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', alpha=0.6)
axes[0].set_title('Original Data (First 2 Features)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# PCA
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6)
axes[1].set_title(f'PCA (Explained Variance: {pca.explained_variance_ratio_.sum():.3f})')
axes[1].set_xlabel('First Principal Component')
axes[1].set_ylabel('Second Principal Component')

# t-SNE
scatter3 = axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6)
axes[2].set_title('t-SNE')
axes[2].set_xlabel('t-SNE Component 1')
axes[2].set_ylabel('t-SNE Component 2')

# Add colorbar
plt.colorbar(scatter1, ax=axes[0])
plt.colorbar(scatter2, ax=axes[1])
plt.colorbar(scatter3, ax=axes[2])

plt.tight_layout()
plt.show()

# Show explained variance ratio for PCA
pca_full = PCA()
pca_full.fit(X)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, 11), pca_full.explained_variance_ratio_[:10], 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Individual Explained Variance')

plt.subplot(1, 2, 2)
plt.plot(range(1, 11), np.cumsum(pca_full.explained_variance_ratio_[:10]), 'ro-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')

plt.tight_layout()
plt.show()
```

### Training, Validation, and Testing

Understanding the data splitting strategy is crucial for building reliable ML models that generalize well to unseen data.

#### The Three-Way Split

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)

# Three-way split: 60% train, 20% validation, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning using validation set
alphas = np.logspace(-4, 4, 50)
train_scores = []
val_scores = []

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    
    train_scores.append(train_mse)
    val_scores.append(val_mse)

# Find best hyperparameter
best_alpha_idx = np.argmin(val_scores)
best_alpha = alphas[best_alpha_idx]

print(f"Best alpha: {best_alpha:.6f}")
print(f"Best validation MSE: {val_scores[best_alpha_idx]:.3f}")

# Train final model with best hyperparameter
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train_scaled, y_train)

# Evaluate on test set (only once!)
test_pred = final_model.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, test_pred)

print(f"Final test MSE: {test_mse:.3f}")

# Visualize validation curve
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.semilogx(alphas, train_scores, 'b-', label='Training MSE', alpha=0.8)
plt.semilogx(alphas, val_scores, 'r-', label='Validation MSE', alpha=0.8)
plt.axvline(best_alpha, color='g', linestyle='--', label=f'Best α={best_alpha:.6f}')
plt.xlabel('Regularization Parameter (α)')
plt.ylabel('Mean Squared Error')
plt.title('Validation Curve')
plt.legend()
plt.grid(True)

# Learning curve
train_sizes, train_scores_lc, val_scores_lc = learning_curve(
    Ridge(alpha=best_alpha), X_train_scaled, y_train, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='neg_mean_squared_error'
)

plt.subplot(1, 3, 2)
plt.plot(train_sizes, -train_scores_lc.mean(axis=1), 'b-', label='Training MSE')
plt.fill_between(train_sizes, -train_scores_lc.mean(axis=1) - train_scores_lc.std(axis=1),
                 -train_scores_lc.mean(axis=1) + train_scores_lc.std(axis=1), alpha=0.2, color='b')
plt.plot(train_sizes, -val_scores_lc.mean(axis=1), 'r-', label='Validation MSE')
plt.fill_between(train_sizes, -val_scores_lc.mean(axis=1) - val_scores_lc.std(axis=1),
                 -val_scores_lc.mean(axis=1) + val_scores_lc.std(axis=1), alpha=0.2, color='r')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)

# Residual plot for final model
plt.subplot(1, 3, 3)
plt.scatter(test_pred, y_test - test_pred, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot (Test Set)')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Overfitting and Regularization

Overfitting occurs when a model learns the training data too well, including its noise and peculiarities, leading to poor generalization.

#### Understanding Overfitting

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Generate noisy dataset
np.random.seed(42)
n_samples = 100
X = np.linspace(0, 1, n_samples).reshape(-1, 1)
y = 1.5 * X.ravel() + np.sin(X.ravel() * 10) + np.random.normal(0, 0.3, n_samples)

# Split data
X_train, X_test = X[:70], X[70:]
y_train, y_test = y[:70], y[70:]

# Test different polynomial degrees
degrees = [1, 3, 9, 15]
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, degree in enumerate(degrees):
    # Fit polynomial model
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    poly_model.fit(X_train, y_train)
    
    # Generate smooth curve for plotting
    X_plot = np.linspace(0, 1, 200).reshape(-1, 1)
    y_plot = poly_model.predict(X_plot)
    
    # Make predictions
    y_train_pred = poly_model.predict(X_train)
    y_test_pred = poly_model.predict(X_test)
    
    # Calculate errors
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    # Plot
    axes[i].scatter(X_train, y_train, alpha=0.6, label='Training Data')
    axes[i].scatter(X_test, y_test, alpha=0.6, color='red', label='Test Data')
    axes[i].plot(X_plot, y_plot, color='green', linewidth=2, label=f'Degree {degree} Fit')
    axes[i].set_title(f'Degree {degree}\nTrain MSE: {train_mse:.3f}, Test MSE: {test_mse:.3f}')
    axes[i].legend()
    axes[i].grid(True)
    
    if test_mse > train_mse * 2:
        axes[i].text(0.5, 0.95, 'OVERFITTING!', transform=axes[i].transAxes, 
                    fontsize=12, color='red', weight='bold', ha='center')

plt.tight_layout()
plt.show()
```

#### Regularization Techniques

**1. Ridge Regression (L2 Regularization)**
```python
# Compare different regularization strengths
alphas = [0, 0.1, 1, 10, 100]
fig, axes = plt.subplots(1, len(alphas), figsize=(20, 4))

for i, alpha in enumerate(alphas):
    # Ridge regression with polynomial features
    ridge_model = Pipeline([
        ('poly', PolynomialFeatures(degree=9)),
        ('ridge', Ridge(alpha=alpha))
    ])
    ridge_model.fit(X_train, y_train)
    
    # Predictions
    X_plot = np.linspace(0, 1, 200).reshape(-1, 1)
    y_plot = ridge_model.predict(X_plot)
    y_train_pred = ridge_model.predict(X_train)
    y_test_pred = ridge_model.predict(X_test)
    
    # Errors
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    # Plot
    axes[i].scatter(X_train, y_train, alpha=0.6, label='Training')
    axes[i].scatter(X_test, y_test, alpha=0.6, color='red', label='Test')
    axes[i].plot(X_plot, y_plot, color='green', linewidth=2)
    axes[i].set_title(f'Ridge α={alpha}\nTrain: {train_mse:.3f}, Test: {test_mse:.3f}')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()
```

**2. Lasso Regression (L1 Regularization)**
```python
# Lasso for feature selection
from sklearn.datasets import make_regression

# Generate dataset with many features
X_many, y_many = make_regression(n_samples=100, n_features=20, n_informative=5, 
                                noise=10, random_state=42)

# Split data
X_train_many, X_test_many, y_train_many, y_test_many = train_test_split(
    X_many, y_many, test_size=0.3, random_state=42)

# Compare Ridge vs Lasso
alphas = np.logspace(-2, 2, 20)
ridge_coefs = []
lasso_coefs = []

for alpha in alphas:
    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_many, y_train_many)
    ridge_coefs.append(ridge.coef_)
    
    # Lasso
    lasso = Lasso(alpha=alpha, max_iter=2000)
    lasso.fit(X_train_many, y_train_many)
    lasso_coefs.append(lasso.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

# Plot coefficient paths
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Ridge coefficients
for i in range(ridge_coefs.shape[1]):
    ax1.semilogx(alphas, ridge_coefs[:, i], label=f'Feature {i+1}')
ax1.set_xlabel('Regularization Parameter (α)')
ax1.set_ylabel('Coefficient Value')
ax1.set_title('Ridge Regression Coefficient Path')
ax1.grid(True)

# Lasso coefficients
for i in range(lasso_coefs.shape[1]):
    ax2.semilogx(alphas, lasso_coefs[:, i], label=f'Feature {i+1}')
ax2.set_xlabel('Regularization Parameter (α)')
ax2.set_ylabel('Coefficient Value')
ax2.set_title('Lasso Regression Coefficient Path')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Show feature selection effect
lasso_final = Lasso(alpha=1.0)
lasso_final.fit(X_train_many, y_train_many)
selected_features = np.where(np.abs(lasso_final.coef_) > 0.01)[0]

print(f"Original number of features: {X_many.shape[1]}")
print(f"Features selected by Lasso: {len(selected_features)}")
print(f"Selected feature indices: {selected_features}")
```

### Feature Engineering

Feature engineering is the process of creating new features or transforming existing ones to improve model performance.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor

# Create sample dataset
np.random.seed(42)
n_samples = 1000

# Generate raw features
data = {
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.normal(50000, 20000, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'experience_years': np.random.randint(0, 40, n_samples),
    'date_joined': pd.date_range('2020-01-01', periods=n_samples, freq='D')
}

df = pd.DataFrame(data)

# Target variable (synthetic relationship)
df['salary'] = (df['income'] * 0.5 + df['age'] * 1000 + df['experience_years'] * 2000 + 
                np.random.normal(0, 5000, n_samples))

print("Original Dataset:")
print(df.head())
print(f"Shape: {df.shape}")

# Feature Engineering Techniques

# 1. Numerical Feature Transformations
print("\n1. Numerical Feature Transformations:")

# Log transformation for skewed data
df['log_income'] = np.log1p(df['income'])

# Polynomial features
df['age_squared'] = df['age'] ** 2
df['income_experience_interaction'] = df['income'] * df['experience_years']

# Binning continuous variables
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 65, 100], 
                        labels=['Young', 'Middle', 'Senior', 'Elder'])

print("Added: log_income, age_squared, income_experience_interaction, age_group")

# 2. Categorical Feature Encoding
print("\n2. Categorical Feature Encoding:")

# Label Encoding
le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])

# One-Hot Encoding
education_dummies = pd.get_dummies(df['education'], prefix='education')
df = pd.concat([df, education_dummies], axis=1)

print("Added: education_encoded, education_* dummy variables")

# 3. Date/Time Feature Engineering
print("\n3. Date/Time Feature Engineering:")

df['join_year'] = df['date_joined'].dt.year
df['join_month'] = df['date_joined'].dt.month
df['join_day_of_week'] = df['date_joined'].dt.dayofweek
df['join_quarter'] = df['date_joined'].dt.quarter

# Days since joining
df['days_since_joining'] = (pd.Timestamp.now() - df['date_joined']).dt.days

print("Added: join_year, join_month, join_day_of_week, join_quarter, days_since_joining")

# 4. Feature Scaling
print("\n4. Feature Scaling:")

numerical_features = ['age', 'income', 'experience_years', 'log_income', 'age_squared', 
                     'income_experience_interaction', 'days_since_joining']

# Standard Scaling (z-score normalization)
scaler_standard = StandardScaler()
df_scaled_standard = df.copy()
df_scaled_standard[numerical_features] = scaler_standard.fit_transform(df[numerical_features])

# Min-Max Scaling
scaler_minmax = MinMaxScaler()
df_scaled_minmax = df.copy()
df_scaled_minmax[numerical_features] = scaler_minmax.fit_transform(df[numerical_features])

print("Applied StandardScaler and MinMaxScaler to numerical features")

# 5. Feature Selection
print("\n5. Feature Selection:")

# Prepare features for selection
feature_columns = numerical_features + list(education_dummies.columns) + ['education_encoded']
X = df_scaled_standard[feature_columns]
y = df['salary']

# Univariate Feature Selection
selector_univariate = SelectKBest(score_func=f_regression, k=10)
X_selected_univariate = selector_univariate.fit_transform(X, y)
selected_features_univariate = X.columns[selector_univariate.get_support()].tolist()

print(f"Univariate selection chose: {selected_features_univariate}")

# Recursive Feature Elimination
rf = RandomForestRegressor(n_estimators=50, random_state=42)
selector_rfe = RFE(estimator=rf, n_features_to_select=8)
X_selected_rfe = selector_rfe.fit_transform(X, y)
selected_features_rfe = X.columns[selector_rfe.get_support()].tolist()

print(f"RFE selection chose: {selected_features_rfe}")

# Feature Importance from Random Forest
rf.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualize feature engineering results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Original vs transformed distributions
axes[0, 0].hist(df['income'], bins=30, alpha=0.7, label='Original Income')
axes[0, 0].set_title('Original Income Distribution')
axes[0, 0].set_xlabel('Income')

axes[0, 1].hist(df['log_income'], bins=30, alpha=0.7, label='Log Income', color='orange')
axes[0, 1].set_title('Log-Transformed Income Distribution')
axes[0, 1].set_xlabel('Log(Income + 1)')

# Feature correlation with target
correlation_with_target = X.corrwith(y).abs().sort_values(ascending=False)
axes[0, 2].barh(range(len(correlation_with_target[:10])), correlation_with_target[:10])
axes[0, 2].set_yticks(range(len(correlation_with_target[:10])))
axes[0, 2].set_yticklabels(correlation_with_target[:10].index)
axes[0, 2].set_title('Feature Correlation with Target')
axes[0, 2].set_xlabel('Absolute Correlation')

# Feature importance
axes[1, 0].barh(range(len(feature_importance[:10])), feature_importance['importance'][:10])
axes[1, 0].set_yticks(range(len(feature_importance[:10])))
axes[1, 0].set_yticklabels(feature_importance['feature'][:10])
axes[1, 0].set_title('Random Forest Feature Importance')
axes[1, 0].set_xlabel('Importance')

# Scaling comparison
axes[1, 1].boxplot([df['income'], df_scaled_standard['income'], df_scaled_minmax['income']], 
                   labels=['Original', 'StandardScaled', 'MinMaxScaled'])
axes[1, 1].set_title('Effect of Different Scaling Methods')
axes[1, 1].set_ylabel('Values')

# Age group distribution
age_group_counts = df['age_group'].value_counts()
axes[1, 2].pie(age_group_counts.values, labels=age_group_counts.index, autopct='%1.1f%%')
axes[1, 2].set_title('Age Group Distribution (Binning Result)')

plt.tight_layout()
plt.show()

print(f"\nFinal dataset shape after feature engineering: {df.shape}")
print(f"Number of features available for modeling: {len(feature_columns)}")
```

## Neural Network Basics

### Neurons and Activation Functions

#### Understanding the Artificial Neuron

An artificial neuron (perceptron) mimics the basic functionality of biological neurons. It receives inputs, processes them, and produces an output.

**Mathematical Model:**
```
output = activation_function(Σ(weights * inputs) + bias)
```

```python
import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    """Simple implementation of an artificial neuron"""
    
    def __init__(self, n_inputs):
        # Initialize weights randomly
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = np.random.randn() * 0.1
        
    def forward(self, inputs, activation='sigmoid'):
        """Forward pass through the neuron"""
        # Linear combination
        z = np.dot(inputs, self.weights) + self.bias
        
        # Apply activation function
        if activation == 'sigmoid':
            return self.sigmoid(z)
        elif activation == 'relu':
            return self.relu(z)
        elif activation == 'tanh':
            return self.tanh(z)
        else:
            return z  # Linear activation
    
    @staticmethod
    def sigmoid(z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def relu(z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    @staticmethod
    def tanh(z):
        """Hyperbolic tangent activation function"""
        return np.tanh(z)

# Demonstrate different activation functions
x = np.linspace(-5, 5, 1000)

# Create sample neuron
neuron = Neuron(1)
neuron.weights = np.array([1.0])  # Set weight to 1 for clarity
neuron.bias = 0.0  # Set bias to 0 for clarity

# Calculate outputs for different activations
sigmoid_output = [neuron.forward(np.array([xi]), 'sigmoid') for xi in x]
relu_output = [neuron.forward(np.array([xi]), 'relu') for xi in x]
tanh_output = [neuron.forward(np.array([xi]), 'tanh') for xi in x]
linear_output = [neuron.forward(np.array([xi]), 'linear') for xi in x]

# Plot activation functions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(x, sigmoid_output, 'b-', linewidth=2)
axes[0, 0].set_title('Sigmoid Activation\nσ(x) = 1/(1+e^(-x))')
axes[0, 0].grid(True)
axes[0, 0].set_ylabel('Output')

axes[0, 1].plot(x, relu_output, 'r-', linewidth=2)
axes[0, 1].set_title('ReLU Activation\nReLU(x) = max(0, x)')
axes[0, 1].grid(True)

axes[1, 0].plot(x, tanh_output, 'g-', linewidth=2)
axes[1, 0].set_title('Tanh Activation\ntanh(x) = (e^x - e^(-x))/(e^x + e^(-x))')
axes[1, 0].grid(True)
axes[1, 0].set_xlabel('Input')
axes[1, 0].set_ylabel('Output')

axes[1, 1].plot(x, linear_output, 'm-', linewidth=2)
axes[1, 1].set_title('Linear Activation\nf(x) = x')
axes[1, 1].grid(True)
axes[1, 1].set_xlabel('Input')

plt.tight_layout()
plt.show()

# Properties of activation functions
print("Activation Function Properties:")
print("1. Sigmoid: Output range (0,1), smooth, prone to vanishing gradient")
print("2. ReLU: Output range [0,∞), not smooth at 0, solves vanishing gradient")
print("3. Tanh: Output range (-1,1), smooth, zero-centered")
print("4. Linear: Output range (-∞,∞), no non-linearity")
```

#### Advanced Activation Functions

```python
class AdvancedActivations:
    """Collection of advanced activation functions"""
    
    @staticmethod
    def leaky_relu(z, alpha=0.01):
        """Leaky ReLU to prevent dying neurons"""
        return np.where(z > 0, z, alpha * z)
    
    @staticmethod
    def elu(z, alpha=1.0):
        """Exponential Linear Unit"""
        return np.where(z > 0, z, alpha * (np.exp(z) - 1))
    
    @staticmethod
    def swish(z):
        """Swish activation function"""
        return z * (1 / (1 + np.exp(-z)))
    
    @staticmethod
    def softmax(z):
        """Softmax for multi-class classification"""
        exp_z = np.exp(z - np.max(z))  # Numerical stability
        return exp_z / np.sum(exp_z)

# Demonstrate advanced activations
activations = AdvancedActivations()

leaky_relu_output = [activations.leaky_relu(xi) for xi in x]
elu_output = [activations.elu(xi) for xi in x]
swish_output = [activations.swish(xi) for xi in x]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x, leaky_relu_output, 'purple', linewidth=2, label='Leaky ReLU')
plt.plot(x, relu_output, 'r--', alpha=0.7, label='ReLU')
plt.title('Leaky ReLU vs ReLU')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, elu_output, 'orange', linewidth=2)
plt.title('ELU Activation')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, swish_output, 'brown', linewidth=2)
plt.title('Swish Activation')
plt.grid(True)

plt.tight_layout()
plt.show()

# Demonstrate softmax for classification
class_scores = np.array([2.0, 1.0, 0.1])
softmax_probs = activations.softmax(class_scores)
print(f"Class scores: {class_scores}")
print(f"Softmax probabilities: {softmax_probs}")
print(f"Sum of probabilities: {np.sum(softmax_probs):.6f}")
```

### Feedforward Networks

#### Building a Multi-Layer Perceptron from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NeuralNetwork:
    """Multi-layer perceptron implementation from scratch"""
    
    def __init__(self, layers):
        """
        Initialize neural network
        layers: list of integers representing number of neurons in each layer
        """
        self.layers = layers
        self.num_layers = len(layers)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # Xavier initialization
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid function"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU function"""
        return (z > 0).astype(float)
    
    def forward(self, X):
        """Forward propagation"""
        activations = [X]
        z_values = []
        
        for i in range(self.num_layers - 1):
            z = np.dot(activations[i], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Use ReLU for hidden layers, sigmoid for output
            if i < self.num_layers - 2:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)
            
            activations.append(a)
        
        return activations, z_values
    
    def compute_cost(self, y_true, y_pred):
        """Binary cross-entropy loss"""
        m = y_true.shape[0]
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def backward(self, X, y, activations, z_values):
        """Backpropagation"""
        m = X.shape[0]
        gradients_w = []
        gradients_b = []
        
        # Output layer error
        delta = activations[-1] - y
        
        # Backpropagate through layers
        for i in range(self.num_layers - 2, -1, -1):
            # Compute gradients
            dW = np.dot(activations[i].T, delta) / m
            db = np.mean(delta, axis=0, keepdims=True)
            
            gradients_w.insert(0, dW)
            gradients_b.insert(0, db)
            
            # Compute delta for previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                # Apply derivative of activation function
                if i > 0:  # Hidden layers use ReLU
                    delta *= self.relu_derivative(z_values[i-1])
        
        return gradients_w, gradients_b
    
    def update_parameters(self, gradients_w, gradients_b, learning_rate):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients_w[i]
            self.biases[i] -= learning_rate * gradients_b[i]
    
    def train(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):
        """Train the neural network"""
        costs = []
        
        for epoch in range(epochs):
            # Forward propagation
            activations, z_values = self.forward(X)
            y_pred = activations[-1]
            
            # Compute cost
            cost = self.compute_cost(y, y_pred)
            costs.append(cost)
            
            # Backward propagation
            gradients_w, gradients_b = self.backward(X, y, activations, z_values)
            
            # Update parameters
            self.update_parameters(gradients_w, gradients_b, learning_rate)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Cost = {cost:.6f}")
        
        return costs
    
    def predict(self, X):
        """Make predictions"""
        activations, _ = self.forward(X)
        predictions = (activations[-1] > 0.5).astype(int)
        return predictions, activations[-1]

# Generate sample datasets
print("Generating sample datasets...")

# Dataset 1: Linearly separable
X1, y1 = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                           n_informative=2, n_clusters_per_class=1, random_state=42)

# Dataset 2: Non-linearly separable (circles)
X2, y2 = make_circles(n_samples=1000, factor=0.3, noise=0.1, random_state=42)

datasets = [
    (X1, y1, "Linearly Separable"),
    (X2, y2, "Non-linearly Separable (Circles)")
]

# Train networks on different datasets
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for dataset_idx, (X, y, title) in enumerate(datasets):
    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape y for neural network
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # Create and train neural network
    print(f"\nTraining neural network on {title} data...")
    nn = NeuralNetwork([2, 8, 4, 1])  # 2 inputs, 8 hidden, 4 hidden, 1 output
    costs = nn.train(X_train_scaled, y_train, epochs=1000, learning_rate=0.1, verbose=False)
    
    # Make predictions
    train_predictions, train_probs = nn.predict(X_train_scaled)
    test_predictions, test_probs = nn.predict(X_test_scaled)
    
    # Calculate accuracy
    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)
    
    print(f"Training Accuracy: {train_accuracy:.3f}")
    print(f"Testing Accuracy: {test_accuracy:.3f}")
    
    # Plot results
    row = dataset_idx
    
    # Original data
    axes[row, 0].scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='coolwarm', alpha=0.7)
    axes[row, 0].set_title(f'{title}\nOriginal Data')
    
    # Training curve
    axes[row, 1].plot(costs)
    axes[row, 1].set_title('Training Loss Curve')
    axes[row, 1].set_xlabel('Epoch')
    axes[row, 1].set_ylabel('Cost')
    axes[row, 1].grid(True)
    
    # Decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    _, mesh_probs = nn.predict(mesh_points_scaled)
    mesh_probs = mesh_probs.reshape(xx.shape)
    
    axes[row, 2].contourf(xx, yy, mesh_probs, levels=50, alpha=0.8, cmap='coolwarm')
    axes[row, 2].scatter(X_test[:, 0], X_test[:, 1], c=y_test.ravel(), 
                        cmap='coolwarm', edgecolors='black')
    axes[row, 2].set_title(f'Decision Boundary\nTest Acc: {test_accuracy:.3f}')

plt.tight_layout()
plt.show()
```

#### Using PyTorch for Neural Networks

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class PyTorchNeuralNetwork(nn.Module):
    """Neural network using PyTorch"""
    
    def __init__(self, input_size, hidden_sizes, output_size):
        super(PyTorchNeuralNetwork, self).__init__()
        
        # Create layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Regularization
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Convert data to PyTorch tensors
X_tensor = torch.FloatTensor(X_train_scaled)
y_tensor = torch.FloatTensor(y_train)

# Create DataLoader for batch processing
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
pytorch_model = PyTorchNeuralNetwork(input_size=2, hidden_sizes=[16, 8], output_size=1)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

# Training loop
print("\nTraining PyTorch Neural Network...")
pytorch_costs = []

for epoch in range(500):
    epoch_loss = 0.0
    
    for batch_X, batch_y in dataloader:
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = pytorch_model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    pytorch_costs.append(avg_loss)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

# Evaluate PyTorch model
pytorch_model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    pytorch_predictions = pytorch_model(X_test_tensor)
    pytorch_predictions_binary = (pytorch_predictions > 0.5).float()
    pytorch_accuracy = (pytorch_predictions_binary == torch.FloatTensor(y_test)).float().mean()

print(f"PyTorch Model Test Accuracy: {pytorch_accuracy:.3f}")

# Compare training curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(costs, label='From Scratch')
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(pytorch_costs, label='PyTorch', color='orange')
plt.title('PyTorch Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Deep Learning Architectures

### Convolutional Neural Networks (CNNs)

CNNs are specifically designed for processing grid-like data such as images. They use convolution operations to detect local features and reduce the number of parameters compared to fully connected networks.

#### CNN Components and Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class SimpleCNN(nn.Module):
    """Simple CNN for image classification"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Assuming 32x32 input images
        self.fc2 = nn.Linear(512, num_classes)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        # Conv Block 1: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv Block 3: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Demonstrate CNN components
def demonstrate_cnn_components():
    """Visualize how CNN components work"""
    
    # Create a sample image
    sample_image = torch.randn(1, 3, 32, 32)  # Batch=1, Channels=3, Height=32, Width=32
    
    # Define different kernels for demonstration
    edge_kernel = torch.tensor([[[[-1, -1, -1],
                                  [-1,  8, -1],
                                  [-1, -1, -1]]]], dtype=torch.float32)
    
    blur_kernel = torch.tensor([[[[1/9, 1/9, 1/9],
                                  [1/9, 1/9, 1/9],
                                  [1/9, 1/9, 1/9]]]], dtype=torch.float32)
    
    # Apply convolutions
    edge_result = F.conv2d(sample_image[:, 0:1], edge_kernel, padding=1)
    blur_result = F.conv2d(sample_image[:, 0:1], blur_kernel, padding=1)
    
    # Apply pooling
    pooled_result = F.max_pool2d(sample_image, kernel_size=2, stride=2)
    
    print("CNN Components Demonstration:")
    print(f"Original image shape: {sample_image.shape}")
    print(f"After edge detection: {edge_result.shape}")
    print(f"After blur filter: {blur_result.shape}")
    print(f"After max pooling: {pooled_result.shape}")
    
    # Visualize the effects
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image (first channel)
    axes[0, 0].imshow(sample_image[0, 0].detach().numpy(), cmap='gray')
    axes[0, 0].set_title('Original Image (Channel 1)')
    axes[0, 0].axis('off')
    
    # Edge detection result
    axes[0, 1].imshow(edge_result[0, 0].detach().numpy(), cmap='gray')
    axes[0, 1].set_title('Edge Detection Filter')
    axes[0, 1].axis('off')
    
    # Blur result
    axes[0, 2].imshow(blur_result[0, 0].detach().numpy(), cmap='gray')
    axes[0, 2].set_title('Blur Filter')
    axes[0, 2].axis('off')
    
    # Max pooling visualization
    axes[1, 0].imshow(sample_image[0, 0].detach().numpy(), cmap='gray')
    axes[1, 0].set_title('Before Pooling (32x32)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pooled_result[0, 0].detach().numpy(), cmap='gray')
    axes[1, 1].set_title('After Max Pooling (16x16)')
    axes[1, 1].axis('off')
    
    # Feature map visualization
    model = SimpleCNN()
    model.eval()
    with torch.no_grad():
        conv1_output = model.conv1(sample_image)
        
    # Show first 6 feature maps from conv1
    feature_maps = conv1_output[0, :6].detach().numpy()
    
    axes[1, 2].axis('off')
    axes[1, 2].set_title('First 6 Feature Maps')
    
    # Create a grid of feature maps
    grid_img = np.zeros((96, 96))  # 3x2 grid of 32x32 feature maps
    for i in range(6):
        row = i // 3
        col = i % 3
        grid_img[row*32:(row+1)*32, col*32:(col+1)*32] = feature_maps[i]
    
    axes[1, 2].imshow(grid_img, cmap='viridis')
    
    plt.tight_layout()
    plt.show()

demonstrate_cnn_components()

# Train a CNN on CIFAR-10
def train_cnn_example():
    """Train a CNN on CIFAR-10 dataset"""
    
    print("Training CNN on CIFAR-10...")
    
    # Data preprocessing
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets (subset for demonstration)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    
    # CIFAR-10 classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Initialize model
    model = SimpleCNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop (abbreviated for demonstration)
    model.train()
    train_losses = []
    
    print("Starting training...")
    for epoch in range(2):  # Just 2 epochs for demonstration
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            if i >= 100:  # Limit to 100 batches for demo
                break
                
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 20 == 19:
                print(f'Epoch {epoch+1}, Batch {i+1}: Loss = {running_loss/20:.4f}')
                train_losses.append(running_loss / 20)
                running_loss = 0.0
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            if i >= 50:  # Limit for demo
                break
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Batch (x20)')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Visualize some predictions
    plt.subplot(1, 2, 2)
    model.eval()
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    with torch.no_grad():
        outputs = model(images[:8])
        _, predicted = torch.max(outputs, 1)
    
    # Create a grid of images
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        img = images[i]
        # Denormalize image for display
        img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        row = i // 4
        col = i % 4
        axes[row, col].imshow(np.transpose(img.numpy(), (1, 2, 0)))
        axes[row, col].set_title(f'True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Uncomment to run CNN training (requires downloading CIFAR-10)
# train_cnn_example()
```

### Recurrent Neural Networks (RNNs)

RNNs are designed for sequential data, maintaining hidden states that carry information across time steps. They're ideal for tasks like language modeling, time series prediction, and sequence generation.

#### RNN Implementation and Variants

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class SimpleRNN(nn.Module):
    """Basic RNN implementation"""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel(nn.Module):
    """LSTM implementation"""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    """GRU implementation"""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def generate_time_series_data(seq_length=50, num_sequences=1000):
    """Generate synthetic time series data for RNN training"""
    
    X = []
    y = []
    
    for _ in range(num_sequences):
        # Generate a random sine wave with noise
        t = np.linspace(0, 4*np.pi, seq_length + 1)
        frequency = np.random.uniform(0.5, 2.0)
        amplitude = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)
        noise_level = 0.1
        
        signal = amplitude * np.sin(frequency * t + phase) + np.random.normal(0, noise_level, len(t))
        
        # Use first seq_length points as input, last point as target
        X.append(signal[:-1].reshape(-1, 1))
        y.append(signal[-1])
    
    return np.array(X), np.array(y)

def compare_rnn_variants():
    """Compare RNN, LSTM, and GRU on time series prediction"""
    
    print("Comparing RNN variants on time series prediction...")
    
    # Generate data
    seq_length = 30
    X, y = generate_time_series_data(seq_length=seq_length, num_sequences=2000)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]
    
    # Model parameters
    input_size = 1
    hidden_size = 64
    output_size = 1
    num_epochs = 50
    learning_rate = 0.001
    
    # Initialize models
    models = {
        'RNN': SimpleRNN(input_size, hidden_size, output_size),
        'LSTM': LSTMModel(input_size, hidden_size, output_size),
        'GRU': GRUModel(input_size, hidden_size, output_size)
    }
    
    # Training
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        test_losses = []
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_pred = model(X_train)
            train_loss = criterion(train_pred.squeeze(), y_train)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Testing
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test)
                test_loss = criterion(test_pred.squeeze(), y_test)
            
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            
            if epoch % 10 == 0:
                print(f'  Epoch {epoch}: Train Loss = {train_loss.item():.6f}, Test Loss = {test_loss.item():.6f}')
        
        results[name] = {
            'model': model,
            'train_losses': train_losses,
            'test_losses': test_losses
        }
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Training curves
    for i, (name, result) in enumerate(results.items()):
        axes[0, i].plot(result['train_losses'], label='Train Loss')
        axes[0, i].plot(result['test_losses'], label='Test Loss')
        axes[0, i].set_title(f'{name} Training Curve')
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].legend()
        axes[0, i].grid(True)
        axes[0, i].set_yscale('log')
    
    # Prediction examples
    for i, (name, result) in enumerate(results.items()):
        model = result['model']
        model.eval()
        
        with torch.no_grad():
            # Take a few test examples
            sample_inputs = X_test[:3]
            sample_targets = y_test[:3]
            sample_predictions = model(sample_inputs).squeeze()
        
        x_axis = range(seq_length + 1)
        
        for j in range(3):
            if j == 0:
                axes[1, i].plot(x_axis[:-1], sample_inputs[j].squeeze().numpy(), 'b-', alpha=0.7, label='Input Sequence')
                axes[1, i].scatter([seq_length], [sample_targets[j].item()], color='green', s=50, label='True Next Value')
                axes[1, i].scatter([seq_length], [sample_predictions[j].item()], color='red', s=50, label='Predicted Next Value')
            else:
                axes[1, i].plot(x_axis[:-1], sample_inputs[j].squeeze().numpy(), 'b-', alpha=0.7)
                axes[1, i].scatter([seq_length], [sample_targets[j].item()], color='green', s=50)
                axes[1, i].scatter([seq_length], [sample_predictions[j].item()], color='red', s=50)
        
        axes[1, i].set_title(f'{name} Predictions')
        axes[1, i].set_xlabel('Time Step')
        axes[1, i].set_ylabel('Value')
        axes[1, i].legend()
        axes[1, i].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print final results
    print("\nFinal Results:")
    print("-" * 40)
    for name, result in results.items():
        final_train_loss = result['train_losses'][-1]
        final_test_loss = result['test_losses'][-1]
        print(f"{name}:")
        print(f"  Final Train Loss: {final_train_loss:.6f}")
        print(f"  Final Test Loss: {final_test_loss:.6f}")
        print()

compare_rnn_variants()
```

### Transformers and Attention Mechanisms

Transformers revolutionized NLP and are increasingly used in other domains. They rely entirely on attention mechanisms, eliminating the need for recurrence.

#### Attention Mechanism Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        return context, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear layer
        output = self.W_o(context)
        
        return output, attention_weights

class TransformerBlock(nn.Module):
    """A single transformer block"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

def demonstrate_attention():
    """Demonstrate attention mechanism with visualization"""
    
    # Create sample data
    batch_size = 1
    seq_length = 8
    d_model = 64
    num_heads = 8
    
    # Sample input (representing word embeddings)
    x = torch.randn(batch_size, seq_length, d_model)
    
    # Create transformer block
    transformer_block = TransformerBlock(d_model, num_heads, d_ff=256)
    
    # Forward pass
    output, attention_weights = transformer_block(x)
    
    print("Attention Demonstration:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Visualize attention weights
    # Take the first head's attention weights
    first_head_attention = attention_weights[0, 0].detach().numpy()
    
    plt.figure(figsize=(15, 5))
    
    # Plot attention matrix
    plt.subplot(1, 3, 1)
    plt.imshow(first_head_attention, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Attention Weights (Head 1)')
    
    # Plot attention for specific query positions
    plt.subplot(1, 3, 2)
    positions_to_plot = [0, 3, 7]
    for pos in positions_to_plot:
        plt.plot(first_head_attention[pos], marker='o', label=f'Query pos {pos}')
    plt.xlabel('Key Position')
    plt.ylabel('Attention Weight')
    plt.title('Attention Patterns')
    plt.legend()
    plt.grid(True)
    
    # Average attention across all heads
    avg_attention = attention_weights[0].mean(dim=0).detach().numpy()
    plt.subplot(1, 3, 3)
    plt.imshow(avg_attention, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Average Attention (All Heads)')
    
    plt.tight_layout()
    plt.show()

demonstrate_attention()

# Simple text classification with transformer
class TextClassificationTransformer(nn.Module):
    """Simple transformer for text classification"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes, max_length=512):
        super(TextClassificationTransformer, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.create_positional_encoding(max_length, d_model)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff=4*d_model)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def create_positional_encoding(self, max_length, d_model):
        """Create positional encoding matrix"""
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        seq_length = x.size(1)
        
        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_length, :].to(x.device)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x, _ = transformer_block(x)
        
        # Global average pooling and classification
        x = x.mean(dim=1)  # Average over sequence length
        x = self.classifier(x)
        
        return x

print("Transformer components demonstrated successfully!")
```

### Generative Models

Generative models learn to create new data samples that resemble the training data. Common types include Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs).

#### Variational Autoencoder Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class VAE(nn.Module):
    """Variational Autoencoder implementation"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent variable to output"""
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

def vae_loss_function(recon_x, x, mu, log_var):
    """VAE loss function (reconstruction + KL divergence)"""
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kld_loss

# Simple GAN implementation
class Generator(nn.Module):
    """Simple generator for GAN"""
    
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    """Simple discriminator for GAN"""
    
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def demonstrate_generative_models():
    """Demonstrate VAE and GAN on synthetic 2D data"""
    
    # Generate synthetic 2D data (two Gaussian clusters)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create two clusters
    cluster1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], 500)
    cluster2 = np.random.multivariate_normal([-2, -2], [[0.5, 0], [0, 0.5]], 500)
    real_data = np.vstack([cluster1, cluster2])
    
    # Normalize data to [0, 1] for VAE
    real_data_normalized = (real_data - real_data.min()) / (real_data.max() - real_data.min())
    real_data_tensor = torch.FloatTensor(real_data_normalized)
    
    print("Training Generative Models on 2D synthetic data...")
    
    # Train VAE
    vae = VAE(input_dim=2, hidden_dim=64, latent_dim=2)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    
    vae_losses = []
    for epoch in range(200):
        vae_optimizer.zero_grad()
        recon_data, mu, log_var = vae(real_data_tensor)
        loss = vae_loss_function(recon_data, real_data_tensor, mu, log_var)
        loss.backward()
        vae_optimizer.step()
        vae_losses.append(loss.item())
        
        if epoch % 50 == 0:
            print(f'VAE Epoch {epoch}: Loss = {loss.item():.4f}')
    
    # Train GAN
    latent_dim = 2
    generator = Generator(latent_dim, 2)
    discriminator = Discriminator(2)
    
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    
    criterion = nn.BCELoss()
    
    g_losses = []
    d_losses = []
    
    for epoch in range(200):
        batch_size = 64
        
        # Train Discriminator
        d_optimizer.zero_grad()
        
        # Real data
        real_batch = real_data_tensor[torch.randperm(len(real_data_tensor))[:batch_size]]
        real_labels = torch.ones(batch_size, 1)
        d_real = discriminator(real_batch)
        d_loss_real = criterion(d_real, real_labels)
        
        # Fake data
        z = torch.randn(batch_size, latent_dim)
        fake_data = generator(z)
        fake_labels = torch.zeros(batch_size, 1)
        d_fake = discriminator(fake_data.detach())
        d_loss_fake = criterion(d_fake, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        z = torch.randn(batch_size, latent_dim)
        fake_data = generator(z)
        d_fake_g = discriminator(fake_data)
        g_loss = criterion(d_fake_g, real_labels)  # Want discriminator to think fake is real
        g_loss.backward()
        g_optimizer.step()
        
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        
        if epoch % 50 == 0:
            print(f'GAN Epoch {epoch}: G_Loss = {g_loss.item():.4f}, D_Loss = {d_loss.item():.4f}')
    
    # Generate samples and visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original data
    axes[0, 0].scatter(real_data[:, 0], real_data[:, 1], alpha=0.6, s=20)
    axes[0, 0].set_title('Original Data')
    axes[0, 0].grid(True)
    
    # VAE reconstruction
    vae.eval()
    with torch.no_grad():
        recon_data, _, _ = vae(real_data_tensor)
        recon_data_denorm = recon_data.numpy() * (real_data.max() - real_data.min()) + real_data.min()
    
    axes[0, 1].scatter(recon_data_denorm[:, 0], recon_data_denorm[:, 1], alpha=0.6, s=20, color='orange')
    axes[0, 1].set_title('VAE Reconstruction')
    axes[0, 1].grid(True)
    
    # VAE generation
    with torch.no_grad():
        z_sample = torch.randn(1000, 2)
        vae_generated = vae.decode(z_sample)
        vae_generated_denorm = vae_generated.numpy() * (real_data.max() - real_data.min()) + real_data.min()
    
    axes[0, 2].scatter(vae_generated_denorm[:, 0], vae_generated_denorm[:, 1], alpha=0.6, s=20, color='green')
    axes[0, 2].set_title('VAE Generated Samples')
    axes[0, 2].grid(True)
    
    # GAN generation
    generator.eval()
    with torch.no_grad():
        z_sample = torch.randn(1000, latent_dim)
        gan_generated = generator(z_sample)
    
    axes[1, 0].scatter(gan_generated[:, 0], gan_generated[:, 1], alpha=0.6, s=20, color='red')
    axes[1, 0].set_title('GAN Generated Samples')
    axes[1, 0].grid(True)
    
    # Training curves
    axes[1, 1].plot(vae_losses, label='VAE Loss')
    axes[1, 1].set_title('VAE Training Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale('log')
    
    axes[1, 2].plot(g_losses, label='Generator Loss', alpha=0.7)
    axes[1, 2].plot(d_losses, label='Discriminator Loss', alpha=0.7)
    axes[1, 2].set_title('GAN Training Losses')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()

demonstrate_generative_models()
```

### Transfer Learning

Transfer learning leverages pre-trained models to solve new tasks with limited data, significantly reducing training time and improving performance.

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

def demonstrate_transfer_learning():
    """Demonstrate transfer learning concepts"""
    
    print("Transfer Learning Demonstration:")
    print("=" * 40)
    
    # Load pre-trained ResNet
    pretrained_resnet = models.resnet18(pretrained=True)
    print(f"Pre-trained ResNet-18 loaded")
    print(f"Original classifier: {pretrained_resnet.fc}")
    
    # Freeze all layers except the final classifier
    for param in pretrained_resnet.parameters():
        param.requires_grad = False
    
    # Replace the final layer for new task (e.g., 5 classes instead of 1000)
    num_classes = 5
    pretrained_resnet.fc = nn.Linear(pretrained_resnet.fc.in_features, num_classes)
    
    print(f"Modified classifier: {pretrained_resnet.fc}")
    print(f"Trainable parameters: {sum(p.numel() for p in pretrained_resnet.parameters() if p.requires_grad)}")
    print(f"Total parameters: {sum(p.numel() for p in pretrained_resnet.parameters())}")
    
    # Fine-tuning strategy
    class FineTuningStrategy:
        def __init__(self, model):
            self.model = model
            
        def freeze_all(self):
            """Freeze all layers"""
            for param in self.model.parameters():
                param.requires_grad = False
                
        def unfreeze_classifier(self):
            """Unfreeze only the classifier"""
            for param in self.model.fc.parameters():
                param.requires_grad = True
                
        def unfreeze_last_n_blocks(self, n):
            """Unfreeze last n blocks"""
            # For ResNet, unfreeze last n layers
            layers = list(self.model.children())
            for layer in layers[-n:]:
                for param in layer.parameters():
                    param.requires_grad = True
                    
        def unfreeze_all(self):
            """Unfreeze all layers for fine-tuning"""
            for param in self.model.parameters():
                param.requires_grad = True
    
    # Demonstrate different strategies
    strategy = FineTuningStrategy(pretrained_resnet)
    
    print("\nTransfer Learning Strategies:")
    print("-" * 30)
    
    # Strategy 1: Feature extraction (freeze backbone)
    strategy.freeze_all()
    strategy.unfreeze_classifier()
    trainable_1 = sum(p.numel() for p in pretrained_resnet.parameters() if p.requires_grad)
    print(f"1. Feature Extraction: {trainable_1:,} trainable parameters")
    
    # Strategy 2: Fine-tune last block
    strategy.freeze_all()
    strategy.unfreeze_last_n_blocks(2)  # Last 2 layers
    trainable_2 = sum(p.numel() for p in pretrained_resnet.parameters() if p.requires_grad)
    print(f"2. Fine-tune last 2 blocks: {trainable_2:,} trainable parameters")
    
    # Strategy 3: Full fine-tuning
    strategy.unfreeze_all()
    trainable_3 = sum(p.numel() for p in pretrained_resnet.parameters() if p.requires_grad)
    print(f"3. Full fine-tuning: {trainable_3:,} trainable parameters")
    
    # Learning rate scheduling for transfer learning
    print("\nRecommended Learning Rate Strategies:")
    print("-" * 40)
    print("1. Feature Extraction: LR = 1e-3 (higher LR for new classifier)")
    print("2. Fine-tuning: LR = 1e-4 to 1e-5 (lower LR to preserve pre-trained features)")
    print("3. Discriminative LR: Different LR for different layers")
    
    # Example of discriminative learning rates
    def setup_discriminative_lr(model, base_lr=1e-4):
        """Setup different learning rates for different parts of the model"""
        
        # Lower learning rate for pre-trained features
        feature_params = []
        for name, param in model.named_parameters():
            if 'fc' not in name:  # Not the final classifier
                feature_params.append(param)
        
        # Higher learning rate for new classifier
        classifier_params = list(model.fc.parameters())
        
        optimizer = torch.optim.Adam([
            {'params': feature_params, 'lr': base_lr * 0.1},      # 10x lower LR
            {'params': classifier_params, 'lr': base_lr}           # Normal LR
        ])
        
        return optimizer
    
    # Data augmentation for transfer learning
    transfer_learning_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\nTransfer Learning Best Practices:")
    print("-" * 40)
    print("✅ Use pre-trained weights from similar domains")
    print("✅ Start with frozen backbone, gradually unfreeze")
    print("✅ Use lower learning rates for pre-trained layers")
    print("✅ Apply appropriate data augmentation")
    print("✅ Monitor for overfitting with small datasets")
    print("✅ Use discriminative learning rates")

demonstrate_transfer_learning()

print("\nDeep Learning Architectures Overview Complete!")
print("="*50)
print("Covered:")
print("• CNNs for image processing")
print("• RNNs, LSTMs, GRUs for sequences") 
print("• Transformers and attention mechanisms")
print("• Generative models (VAE, GAN)")
print("• Transfer learning strategies")
```

## Learning Objectives

By the end of this comprehensive section, you should be able to:

### Core ML Concepts
- **Distinguish between supervised and unsupervised learning** with practical examples
- **Implement classification and regression algorithms** from scratch and using libraries
- **Apply proper data splitting strategies** (train/validation/test) and cross-validation
- **Identify and mitigate overfitting** using regularization techniques
- **Engineer meaningful features** from raw data to improve model performance

### Neural Networks & Deep Learning
- **Build neural networks from scratch** understanding forward/backward propagation
- **Implement and compare different activation functions** and their properties
- **Design appropriate loss functions** for different types of problems
- **Apply various optimization algorithms** (SGD, Adam, etc.) effectively
- **Construct CNN architectures** for image classification and computer vision tasks
- **Develop RNN/LSTM models** for sequential data and time series prediction
- **Implement attention mechanisms** and understand transformer architectures
- **Create generative models** (VAE, GAN) for data generation tasks
- **Apply transfer learning** to leverage pre-trained models effectively

### Practical Skills
- **Debug and troubleshoot** neural network training issues
- **Visualize and interpret** model behavior and predictions
- **Optimize model performance** through hyperparameter tuning
- **Handle real-world datasets** with preprocessing and augmentation
- **Implement models in PyTorch/TensorFlow** following best practices
- **Evaluate model performance** using appropriate metrics and validation strategies

### Self-Assessment Checklist

Before proceeding to advanced topics, ensure you can:

□ **Implement a neural network from scratch** with proper backpropagation  
□ **Explain the vanishing gradient problem** and solutions (ReLU, skip connections)  
□ **Design CNN architectures** for different image classification tasks  
□ **Build RNN models** for sequence prediction and text processing  
□ **Implement attention mechanisms** and understand self-attention  
□ **Create and train GANs** for generating synthetic data  
□ **Apply transfer learning** to adapt pre-trained models to new domains  
□ **Diagnose overfitting/underfitting** and apply appropriate solutions  
□ **Choose appropriate loss functions** for different ML problems  
□ **Optimize neural networks** using modern optimization techniques  

## Study Materials

### Essential Books
- **"Deep Learning"** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- **"Hands-On Machine Learning"** by Aurélien Géron (2nd Edition)
- **"Pattern Recognition and Machine Learning"** by Christopher Bishop
- **"The Elements of Statistical Learning"** by Hastie, Tibshirani, and Friedman

### Online Courses
- **CS231n: Convolutional Neural Networks** (Stanford) - Free online
- **CS224n: Natural Language Processing** (Stanford) - Free online  
- **Deep Learning Specialization** (Andrew Ng) - Coursera
- **Fast.ai Deep Learning Course** - Free practical approach

### Practical Resources
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **TensorFlow Guides**: https://www.tensorflow.org/guide
- **Papers With Code**: https://paperswithcode.com/ (implementations)
- **Distill.pub**: Visual explanations of ML concepts

### Development Environment

**Required Libraries:**
```bash
# Core ML libraries
pip install torch torchvision torchaudio
pip install tensorflow
pip install scikit-learn
pip install numpy pandas matplotlib seaborn

# For visualization and experimentation  
pip install jupyter plotly
pip install tensorboard
pip install wandb  # For experiment tracking

# For computer vision
pip install opencv-python pillow

# For NLP
pip install transformers datasets tokenizers
```

**Hardware Recommendations:**
- **GPU**: NVIDIA RTX 3060+ or cloud GPUs (Google Colab, AWS)
- **RAM**: 16GB+ for local development
- **Storage**: SSD with 50GB+ free space for datasets

### Hands-on Projects

**Beginner Projects:**
1. **Image Classification**: Build CNN for CIFAR-10 dataset
2. **Time Series Prediction**: Use LSTM for stock price prediction  
3. **Text Classification**: Implement sentiment analysis with RNNs
4. **Clustering Analysis**: Apply K-means and hierarchical clustering

**Intermediate Projects:**
5. **Object Detection**: Implement YOLO or R-CNN from scratch
6. **Language Model**: Build character/word-level RNN language model
7. **Recommendation System**: Collaborative filtering with neural networks
8. **Anomaly Detection**: Autoencoder for fraud detection

**Advanced Projects:**
9. **GAN Implementation**: Create DCGAN for image generation
10. **Transformer from Scratch**: Build attention-based model for translation
11. **Transfer Learning**: Fine-tune pre-trained models for custom datasets
12. **Reinforcement Learning**: Implement Deep Q-Network (DQN)

### Practice Exercises

**Mathematical Foundations:**
```python
# Exercise 1: Implement gradient descent from scratch
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    # Your implementation here
    pass

# Exercise 2: Implement different activation functions and derivatives
def sigmoid(x):
    # Implement sigmoid and its derivative
    pass

# Exercise 3: Build matrix operations for neural networks  
def matrix_multiply_forward_backward(A, B):
    # Implement forward and backward pass
    pass
```

**Implementation Challenges:**
```python
# Challenge 1: Multi-class classification from scratch
class MultiClassPerceptron:
    def __init__(self, input_size, num_classes):
        # Initialize weights and biases
        pass
    
    def train(self, X, y, epochs=100):
        # Implement training loop
        pass

# Challenge 2: Convolutional layer implementation
class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size):
        # Initialize convolution parameters
        pass
    
    def forward(self, x):
        # Implement convolution operation
        pass

# Challenge 3: Attention mechanism from scratch
class ScaledDotProductAttention:
    def forward(self, Q, K, V, mask=None):
        # Implement attention computation
        pass
```

**Real-world Applications:**
- **Medical Image Analysis**: Classify X-rays or MRI scans
- **Natural Language Processing**: Build chatbot or document classifier  
- **Computer Vision**: Create face recognition or object tracking system
- **Time Series Analysis**: Forecast sales, weather, or sensor data
- **Recommender Systems**: Build movie/product recommendation engine

### Assessment Methods

**Knowledge Tests:**
- Multiple choice questions on ML concepts and algorithms
- Code debugging exercises with intentional errors
- Algorithm complexity analysis and comparison

**Practical Assessments:**
- Implement neural networks from mathematical descriptions
- Debug and optimize poorly performing models
- Design architectures for specific problem domains
- Reproduce results from research papers

**Portfolio Projects:**
- End-to-end ML projects with data collection to deployment
- Comparative analysis of different algorithms on same dataset
- Research reproduction with novel improvements
- Open-source contributions to ML libraries

## Next Steps

After mastering these fundamentals, you'll be ready to explore:

- **[Advanced Neural Architectures](02_Advanced_Neural_Networks.md)** - ResNet, EfficientNet, Vision Transformers
- **[Natural Language Processing](03_NLP_Deep_Learning.md)** - BERT, GPT, T5, and language models  
- **[Computer Vision](04_Computer_Vision.md)** - Object detection, segmentation, GANs
- **[Reinforcement Learning](05_Reinforcement_Learning.md)** - Q-learning, policy gradients, actor-critic
- **[MLOps and Production](06_MLOps_Production.md)** - Model deployment, monitoring, and scaling

### Loss Functions

Loss functions measure how well our model's predictions match the actual target values. The choice of loss function depends on the type of problem we're solving.

#### Classification Loss Functions

**1. Binary Cross-Entropy Loss**
Used for binary classification problems.

```python
import numpy as np
import matplotlib.pyplot as plt

def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy loss function"""
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_derivative(y_true, y_pred):
    """Derivative of binary cross-entropy"""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)

# Demonstrate BCE loss behavior
y_true = np.array([1, 0, 1, 0, 1])  # True labels
predictions = np.linspace(0.01, 0.99, 100)  # Range of predictions

# Calculate loss for different predictions when true label is 1
loss_when_true_1 = [-np.log(p) for p in predictions]
# Calculate loss for different predictions when true label is 0
loss_when_true_0 = [-np.log(1 - p) for p in predictions]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(predictions, loss_when_true_1, 'b-', label='True label = 1', linewidth=2)
plt.plot(predictions, loss_when_true_0, 'r-', label='True label = 0', linewidth=2)
plt.xlabel('Predicted Probability')
plt.ylabel('Loss')
plt.title('Binary Cross-Entropy Loss')
plt.legend()
plt.grid(True)

# Demonstrate on sample data
sample_predictions = np.array([0.9, 0.1, 0.8, 0.2, 0.95])
sample_loss = binary_cross_entropy(y_true, sample_predictions)

print(f"True labels: {y_true}")
print(f"Predictions: {sample_predictions}")
print(f"Binary Cross-Entropy Loss: {sample_loss:.4f}")

# Show individual losses
individual_losses = []
for i in range(len(y_true)):
    if y_true[i] == 1:
        loss = -np.log(sample_predictions[i])
    else:
        loss = -np.log(1 - sample_predictions[i])
    individual_losses.append(loss)
    print(f"Sample {i+1}: True={y_true[i]}, Pred={sample_predictions[i]:.2f}, Loss={loss:.4f}")

plt.subplot(1, 3, 2)
bars = plt.bar(range(len(y_true)), individual_losses, 
               color=['blue' if label == 1 else 'red' for label in y_true])
plt.xlabel('Sample')
plt.ylabel('Individual Loss')
plt.title('Individual Sample Losses')
plt.xticks(range(len(y_true)), [f'Sample {i+1}' for i in range(len(y_true))])

# Add value labels on bars
for i, (bar, loss) in enumerate(zip(bars, individual_losses)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{loss:.3f}', ha='center', va='bottom')
```

**2. Categorical Cross-Entropy Loss**
Used for multi-class classification problems.

```python
def categorical_cross_entropy(y_true, y_pred):
    """Categorical cross-entropy loss for multi-class classification"""
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def softmax(z):
    """Softmax activation for multi-class classification"""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Example with 3-class classification
n_samples = 5
n_classes = 3

# One-hot encoded true labels
y_true_multiclass = np.array([
    [1, 0, 0],  # Class 0
    [0, 1, 0],  # Class 1
    [0, 0, 1],  # Class 2
    [1, 0, 0],  # Class 0
    [0, 1, 0]   # Class 1
])

# Raw logits (before softmax)
logits = np.array([
    [2.0, 1.0, 0.1],   # Should predict class 0
    [0.5, 3.0, 0.2],   # Should predict class 1
    [0.1, 0.2, 2.5],   # Should predict class 2
    [1.8, 1.2, 0.5],   # Should predict class 0
    [0.3, 2.8, 0.1]    # Should predict class 1
])

# Apply softmax to get probabilities
y_pred_multiclass = softmax(logits)

# Calculate loss
multiclass_loss = categorical_cross_entropy(y_true_multiclass, y_pred_multiclass)

print(f"\nMulti-class Classification Example:")
print(f"True labels (one-hot):\n{y_true_multiclass}")
print(f"Predicted probabilities:\n{y_pred_multiclass.round(3)}")
print(f"Categorical Cross-Entropy Loss: {multiclass_loss:.4f}")

# Visualize predictions
plt.subplot(1, 3, 3)
classes = ['Class 0', 'Class 1', 'Class 2']
x = np.arange(len(y_true_multiclass))
width = 0.25

for i in range(n_classes):
    plt.bar(x + i * width, y_pred_multiclass[:, i], width, 
            label=f'Class {i}', alpha=0.8)

# Mark true classes
true_classes = np.argmax(y_true_multiclass, axis=1)
for i, true_class in enumerate(true_classes):
    plt.scatter(i + true_class * width, y_pred_multiclass[i, true_class], 
               color='red', s=100, marker='*', zorder=5)

plt.xlabel('Sample')
plt.ylabel('Predicted Probability')
plt.title('Multi-class Predictions\n(Red stars = true class)')
plt.legend()
plt.xticks(x + width, [f'Sample {i+1}' for i in range(len(y_true_multiclass))])

plt.tight_layout()
plt.show()
```

#### Regression Loss Functions

**1. Mean Squared Error (MSE)**
Most common loss for regression problems.

```python
def mean_squared_error(y_true, y_pred):
    """Mean squared error loss function"""
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    """Derivative of MSE"""
    return 2 * (y_pred - y_true) / len(y_true)

# Generate sample regression data
np.random.seed(42)
x_reg = np.linspace(0, 10, 50)
y_true_reg = 2 * x_reg + 1 + np.random.normal(0, 2, len(x_reg))

# Different prediction scenarios
y_pred_good = 2 * x_reg + 1.2  # Good predictions
y_pred_bad = x_reg + 5         # Bad predictions
y_pred_overfitted = y_true_reg + np.random.normal(0, 0.1, len(x_reg))  # Overfitted

# Calculate losses
mse_good = mean_squared_error(y_true_reg, y_pred_good)
mse_bad = mean_squared_error(y_true_reg, y_pred_bad)
mse_overfitted = mean_squared_error(y_true_reg, y_pred_overfitted)

print(f"\nRegression Loss Comparison:")
print(f"Good model MSE: {mse_good:.3f}")
print(f"Bad model MSE: {mse_bad:.3f}")
print(f"Overfitted model MSE: {mse_overfitted:.3f}")

# Visualize regression results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Good predictions
axes[0, 0].scatter(x_reg, y_true_reg, alpha=0.6, label='True values')
axes[0, 0].plot(x_reg, y_pred_good, 'r-', label=f'Predictions (MSE={mse_good:.3f})')
axes[0, 0].set_title('Good Model')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Bad predictions
axes[0, 1].scatter(x_reg, y_true_reg, alpha=0.6, label='True values')
axes[0, 1].plot(x_reg, y_pred_bad, 'r-', label=f'Predictions (MSE={mse_bad:.3f})')
axes[0, 1].set_title('Bad Model')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Overfitted predictions
axes[1, 0].scatter(x_reg, y_true_reg, alpha=0.6, label='True values')
axes[1, 0].scatter(x_reg, y_pred_overfitted, alpha=0.6, color='red', label=f'Predictions (MSE={mse_overfitted:.3f})')
axes[1, 0].set_title('Overfitted Model')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Loss landscape visualization
residuals_good = y_true_reg - y_pred_good
residuals_bad = y_true_reg - y_pred_bad

axes[1, 1].hist(residuals_good, alpha=0.7, label=f'Good Model (std={np.std(residuals_good):.2f})', bins=15)
axes[1, 1].hist(residuals_bad, alpha=0.7, label=f'Bad Model (std={np.std(residuals_bad):.2f})', bins=15)
axes[1, 1].set_xlabel('Residuals (True - Predicted)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Residual Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
```

**2. Mean Absolute Error (MAE)**
More robust to outliers than MSE.

```python
def mean_absolute_error(y_true, y_pred):
    """Mean absolute error loss function"""
    return np.mean(np.abs(y_true - y_pred))

def mae_derivative(y_true, y_pred):
    """Derivative of MAE (subgradient)"""
    return np.sign(y_pred - y_true) / len(y_true)

# Add outliers to demonstrate MAE vs MSE
y_true_with_outliers = y_true_reg.copy()
y_true_with_outliers[10] = 50  # Add outlier
y_true_with_outliers[30] = -20  # Add outlier

mae_good = mean_absolute_error(y_true_with_outliers, y_pred_good)
mae_bad = mean_absolute_error(y_true_with_outliers, y_pred_bad)

mse_good_outliers = mean_squared_error(y_true_with_outliers, y_pred_good)
mse_bad_outliers = mean_squared_error(y_true_with_outliers, y_pred_bad)

print(f"\nWith Outliers:")
print(f"Good model - MSE: {mse_good_outliers:.3f}, MAE: {mae_good:.3f}")
print(f"Bad model - MSE: {mse_bad_outliers:.3f}, MAE: {mae_bad:.3f}")

# Compare loss functions behavior
errors = np.linspace(-5, 5, 100)
mse_losses = errors ** 2
mae_losses = np.abs(errors)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(errors, mse_losses, 'b-', label='MSE', linewidth=2)
plt.plot(errors, mae_losses, 'r-', label='MAE', linewidth=2)
plt.xlabel('Prediction Error')
plt.ylabel('Loss Value')
plt.title('MSE vs MAE Loss Functions')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(x_reg, y_true_with_outliers, alpha=0.6, label='True values (with outliers)')
plt.plot(x_reg, y_pred_good, 'r-', label='Predictions')
plt.scatter([x_reg[10], x_reg[30]], [y_true_with_outliers[10], y_true_with_outliers[30]], 
           color='red', s=100, marker='x', label='Outliers')
plt.title('Data with Outliers')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

**3. Huber Loss**
Combines benefits of MSE and MAE.

```python
def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss function - robust to outliers"""
    error = y_true - y_pred
    condition = np.abs(error) <= delta
    
    # Use MSE for small errors, MAE for large errors
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    
    return np.where(condition, squared_loss, linear_loss).mean()

# Compare different loss functions with outliers
deltas = [0.5, 1.0, 2.0]
errors_range = np.linspace(-5, 5, 100)

plt.figure(figsize=(15, 5))

for i, delta in enumerate(deltas):
    plt.subplot(1, 3, i+1)
    
    # Calculate losses
    mse_vals = errors_range ** 2
    mae_vals = np.abs(errors_range)
    huber_vals = []
    
    for error in errors_range:
        if abs(error) <= delta:
            huber_vals.append(0.5 * error ** 2)
        else:
            huber_vals.append(delta * (abs(error) - 0.5 * delta))
    
    plt.plot(errors_range, mse_vals, 'b-', label='MSE', alpha=0.7)
    plt.plot(errors_range, mae_vals, 'r-', label='MAE', alpha=0.7)
    plt.plot(errors_range, huber_vals, 'g-', label=f'Huber (δ={delta})', linewidth=2)
    
    plt.xlabel('Prediction Error')
    plt.ylabel('Loss Value')
    plt.title(f'Huber Loss (δ={delta})')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 10)

plt.tight_layout()
plt.show()

# Calculate Huber losses for our data
huber_good = huber_loss(y_true_with_outliers, y_pred_good, delta=1.0)
huber_bad = huber_loss(y_true_with_outliers, y_pred_bad, delta=1.0)

print(f"\nHuber Loss Comparison (δ=1.0):")
print(f"Good model - Huber: {huber_good:.3f}")
print(f"Bad model - Huber: {huber_bad:.3f}")
```

### Backpropagation

Backpropagation is the algorithm used to train neural networks by computing gradients of the loss function with respect to the network's parameters.

#### Mathematical Foundation

```python
import numpy as np
import matplotlib.pyplot as plt

class BackpropagationDemo:
    """Detailed implementation of backpropagation for educational purposes"""
    
    def __init__(self):
        # Simple network: 2 inputs -> 3 hidden -> 1 output
        self.W1 = np.random.randn(2, 3) * 0.5  # Input to hidden weights
        self.b1 = np.zeros((1, 3))             # Hidden layer biases
        self.W2 = np.random.randn(3, 1) * 0.5  # Hidden to output weights
        self.b2 = np.zeros((1, 1))             # Output layer bias
        
        # Store intermediate values for backprop
        self.z1 = None  # Hidden layer linear output
        self.a1 = None  # Hidden layer activation
        self.z2 = None  # Output layer linear output
        self.a2 = None  # Output layer activation
        
    def sigmoid(self, z):
        """Sigmoid activation function"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid function"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward propagation with detailed tracking"""
        print(f"Input X shape: {X.shape}")
        print(f"X:\n{X}")
        
        # Layer 1: Input to Hidden
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        print(f"\nHidden layer:")
        print(f"z1 = X·W1 + b1, shape: {self.z1.shape}")
        print(f"z1:\n{self.z1}")
        print(f"a1 = sigmoid(z1):\n{self.a1}")
        
        # Layer 2: Hidden to Output
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        print(f"\nOutput layer:")
        print(f"z2 = a1·W2 + b2, shape: {self.z2.shape}")
        print(f"z2:\n{self.z2}")
        print(f"a2 = sigmoid(z2):\n{self.a2}")
        
        return self.a2
    
    def backward(self, X, y, learning_rate=0.1):
        """Backward propagation with detailed steps"""
        m = X.shape[0]  # Number of samples
        
        print(f"\n{'='*50}")
        print("BACKPROPAGATION")
        print(f"{'='*50}")
        
        # Step 1: Calculate output layer error
        dz2 = self.a2 - y  # Derivative of loss w.r.t. z2
        print(f"Output layer error:")
        print(f"dz2 = a2 - y = {self.a2.flatten()} - {y.flatten()} = {dz2.flatten()}")
        
        # Step 2: Calculate output layer gradients
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.mean(dz2, axis=0, keepdims=True)
        print(f"\nOutput layer gradients:")
        print(f"dW2 = a1.T·dz2 / m:\n{dW2}")
        print(f"db2 = mean(dz2):\n{db2}")
        
        # Step 3: Calculate hidden layer error
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.z1)
        print(f"\nHidden layer error:")
        print(f"dz1 = (dz2·W2.T) * sigmoid'(z1):\n{dz1}")
        
        # Step 4: Calculate hidden layer gradients
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.mean(dz1, axis=0, keepdims=True)
        print(f"\nHidden layer gradients:")
        print(f"dW1 = X.T·dz1 / m:\n{dW1}")
        print(f"db1 = mean(dz1):\n{db1}")
        
        # Step 5: Update parameters
        print(f"\nParameter updates (learning rate = {learning_rate}):")
        print(f"W2: {self.W2.flatten()} -> {(self.W2 - learning_rate * dW2).flatten()}")
        print(f"W1 shape: {self.W1.shape}, dW1 shape: {dW1.shape}")
        
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
        return dW1, db1, dW2, db2

# Demonstrate backpropagation step by step
print("BACKPROPAGATION DEMONSTRATION")
print("="*50)

# Create simple dataset
X_demo = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR problem
y_demo = np.array([[0], [1], [1], [0]])              # XOR outputs

print(f"Dataset (XOR problem):")
print(f"X:\n{X_demo}")
print(f"y:\n{y_demo.flatten()}")

# Initialize network
bp_demo = BackpropagationDemo()

print(f"\nInitial weights:")
print(f"W1 (input to hidden):\n{bp_demo.W1}")
print(f"W2 (hidden to output):\n{bp_demo.W2}")

# Forward pass
print(f"\n{'='*50}")
print("FORWARD PROPAGATION")
print(f"{'='*50}")
predictions = bp_demo.forward(X_demo)

# Calculate loss
loss = np.mean((predictions - y_demo) ** 2)
print(f"\nMean Squared Error: {loss:.6f}")

# Backward pass
gradients = bp_demo.backward(X_demo, y_demo, learning_rate=1.0)

print(f"\nUpdated weights:")
print(f"W1:\n{bp_demo.W1}")
print(f"W2:\n{bp_demo.W2}")
```

#### Gradient Descent Variants

```python
class GradientDescentOptimizers:
    """Implementation of different gradient descent variants"""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        # For momentum
        self.velocity_w = None
        self.velocity_b = None
        self.beta1 = 0.9
        
        # For Adam
        self.m_w = None  # First moment estimate
        self.v_w = None  # Second moment estimate
        self.m_b = None
        self.v_b = None
        self.beta1_adam = 0.9
        self.beta2_adam = 0.999
        self.epsilon = 1e-8
        self.t = 0  # Time step
    
    def sgd(self, weights, biases, dw, db):
        """Standard Stochastic Gradient Descent"""
        weights -= self.learning_rate * dw
        biases -= self.learning_rate * db
        return weights, biases
    
    def momentum(self, weights, biases, dw, db):
        """SGD with Momentum"""
        if self.velocity_w is None:
            self.velocity_w = np.zeros_like(weights)
            self.velocity_b = np.zeros_like(biases)
        
        # Update velocity
        self.velocity_w = self.beta1 * self.velocity_w + (1 - self.beta1) * dw
        self.velocity_b = self.beta1 * self.velocity_b + (1 - self.beta1) * db
        
        # Update parameters
        weights -= self.learning_rate * self.velocity_w
        biases -= self.learning_rate * self.velocity_b
        
        return weights, biases
    
    def adam(self, weights, biases, dw, db):
        """Adam Optimizer"""
        if self.m_w is None:
            self.m_w = np.zeros_like(weights)
            self.v_w = np.zeros_like(weights)
            self.m_b = np.zeros_like(biases)
            self.v_b = np.zeros_like(biases)
        
        self.t += 1
        
        # Update biased first moment estimate
        self.m_w = self.beta1_adam * self.m_w + (1 - self.beta1_adam) * dw
        self.m_b = self.beta1_adam * self.m_b + (1 - self.beta1_adam) * db
        
        # Update biased second moment estimate
        self.v_w = self.beta2_adam * self.v_w + (1 - self.beta2_adam) * (dw ** 2)
        self.v_b = self.beta2_adam * self.v_b + (1 - self.beta2_adam) * (db ** 2)
        
        # Bias correction
        m_w_corrected = self.m_w / (1 - self.beta1_adam ** self.t)
        m_b_corrected = self.m_b / (1 - self.beta1_adam ** self.t)
        v_w_corrected = self.v_w / (1 - self.beta2_adam ** self.t)
        v_b_corrected = self.v_b / (1 - self.beta2_adam ** self.t)
        
        # Update parameters
        weights -= self.learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)
        biases -= self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
        
        return weights, biases

# Compare optimization algorithms on a simple problem
def optimization_comparison():
    """Compare different optimization algorithms"""
    
    # Generate data for a simple quadratic function
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2).reshape(-1, 1)
    
    # Initialize parameters for each optimizer
    optimizers = {
        'SGD': GradientDescentOptimizers(learning_rate=0.01),
        'Momentum': GradientDescentOptimizers(learning_rate=0.01),
        'Adam': GradientDescentOptimizers(learning_rate=0.01)
    }
    
    # Track losses
    losses = {name: [] for name in optimizers.keys()}
    
    # Initialize weights (same for all optimizers)
    initial_weights = np.random.randn(2, 1) * 0.5
    initial_bias = np.zeros((1, 1))
    
    weights = {name: initial_weights.copy() for name in optimizers.keys()}
    biases = {name: initial_bias.copy() for name in optimizers.keys()}
    
    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        for name, optimizer in optimizers.items():
            # Forward pass
            predictions = np.dot(X, weights[name]) + biases[name]
            loss = np.mean((predictions - y) ** 2)
            losses[name].append(loss)
            
            # Backward pass
            dw = 2 * np.dot(X.T, predictions - y) / len(X)
            db = 2 * np.mean(predictions - y, axis=0, keepdims=True)
            
            # Update parameters
            if name == 'SGD':
                weights[name], biases[name] = optimizer.sgd(weights[name], biases[name], dw, db)
            elif name == 'Momentum':
                weights[name], biases[name] = optimizer.momentum(weights[name], biases[name], dw, db)
            elif name == 'Adam':
                weights[name], biases[name] = optimizer.adam(weights[name], biases[name], dw, db)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for name, loss_history in losses.items():
        plt.plot(loss_history, label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    for name, loss_history in losses.items():
        plt.plot(loss_history[-100:], label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Last 100 Epochs (Convergence)')
    plt.legend()
    plt.grid(True)
    
    # Show final weights
    plt.subplot(1, 3, 3)
    names = list(weights.keys())
    final_weights = [weights[name].flatten() for name in names]
    
    x_pos = np.arange(len(names))
    for i in range(2):  # Two weight parameters
        plt.bar(x_pos + i * 0.25, [w[i] for w in final_weights], 
               width=0.25, label=f'Weight {i+1}', alpha=0.8)
    
    plt.xlabel('Optimizer')
    plt.ylabel('Final Weight Value')
    plt.title('Final Learned Weights')
    plt.xticks(x_pos + 0.125, names)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print final results
    print("Optimization Results:")
    print("-" * 40)
    for name in names:
        final_loss = losses[name][-1]
        print(f"{name:10s}: Final Loss = {final_loss:.6f}")
        print(f"           Final Weights = {weights[name].flatten()}")
        print(f"           Final Bias = {biases[name].flatten()}")
        print()

optimization_comparison()
```

### Gradient Descent

Gradient descent is the fundamental optimization algorithm used to train neural networks by iteratively moving in the direction of steepest descent of the loss function.

#### Understanding Gradient Descent Geometrically

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gradient_descent_visualization():
    """Visualize gradient descent on a 2D surface"""
    
    # Define a simple quadratic function
    def f(x, y):
        return x**2 + y**2 + 2*x*y + x + y
    
    def gradient_f(x, y):
        df_dx = 2*x + 2*y + 1
        df_dy = 2*y + 2*x + 1
        return np.array([df_dx, df_dy])
    
    # Create mesh for plotting
    x = np.linspace(-3, 2, 100)
    y = np.linspace(-3, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    # Gradient descent implementation
    def gradient_descent_2d(start_point, learning_rate, num_iterations):
        path = [start_point]
        point = start_point.copy()
        
        for i in range(num_iterations):
            grad = gradient_f(point[0], point[1])
            point = point - learning_rate * grad
            path.append(point.copy())
            
            # Early stopping if gradient is very small
            if np.linalg.norm(grad) < 1e-6:
                break
                
        return np.array(path)
    
    # Test different learning rates
    start_point = np.array([2.0, 1.5])
    learning_rates = [0.01, 0.1, 0.3, 0.5]
    
    fig = plt.figure(figsize=(20, 15))
    
    for i, lr in enumerate(learning_rates):
        # Run gradient descent
        path = gradient_descent_2d(start_point, lr, 100)
        
        # 3D surface plot
        ax1 = fig.add_subplot(3, 4, i+1, projection='3d')
        ax1.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
        
        # Plot path
        path_z = [f(p[0], p[1]) for p in path]
        ax1.plot(path[:, 0], path[:, 1], path_z, 'r-', linewidth=3, marker='o', markersize=4)
        ax1.scatter(start_point[0], start_point[1], f(start_point[0], start_point[1]), 
                   color='red', s=100, label='Start')
        ax1.scatter(path[-1, 0], path[-1, 1], f(path[-1, 0], path[-1, 1]), 
                   color='green', s=100, label='End')
        
        ax1.set_title(f'3D View (LR={lr})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('f(x,y)')
        
        # 2D contour plot
        ax2 = fig.add_subplot(3, 4, i+5)
        contour = ax2.contour(X, Y, Z, levels=20)
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, marker='o', markersize=4)
        ax2.scatter(start_point[0], start_point[1], color='red', s=100, zorder=5, label='Start')
        ax2.scatter(path[-1, 0], path[-1, 1], color='green', s=100, zorder=5, label='End')
        
        ax2.set_title(f'Contour View (LR={lr})\nSteps: {len(path)-1}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        ax2.grid(True)
        
        # Loss over iterations
        ax3 = fig.add_subplot(3, 4, i+9)
        loss_values = [f(p[0], p[1]) for p in path]
        ax3.plot(loss_values, 'b-', linewidth=2, marker='o', markersize=4)
        ax3.set_title(f'Loss vs Iteration (LR={lr})')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
        
        # Add convergence info
        if len(path) > 1:
            final_loss = loss_values[-1]
            ax3.text(0.05, 0.95, f'Final Loss: {final_loss:.4f}', 
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Analyze convergence behavior
    print("Gradient Descent Analysis:")
    print("-" * 40)
    for lr in learning_rates:
        path = gradient_descent_2d(start_point, lr, 100)
        final_point = path[-1]
        final_loss = f(final_point[0], final_point[1])
        steps = len(path) - 1
        
        print(f"Learning Rate {lr}:")
        print(f"  Steps to convergence: {steps}")
        print(f"  Final point: ({final_point[0]:.4f}, {final_point[1]:.4f})")
        print(f"  Final loss: {final_loss:.6f}")
        
        if lr >= 0.5:
            print(f"  Warning: High learning rate may cause instability!")
        print()

gradient_descent_visualization()
```

#### Batch vs Stochastic vs Mini-batch Gradient Descent

```python
def compare_gradient_descent_variants():
    """Compare different variants of gradient descent"""
    
    # Generate larger dataset
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 2)
    y = (3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(n_samples) * 0.1).reshape(-1, 1)
    
    # Add bias column
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    
    def mse_loss(X, y, weights):
        predictions = np.dot(X, weights)
        return np.mean((predictions - y) ** 2)
    
    def mse_gradient(X, y, weights):
        m = X.shape[0]
        predictions = np.dot(X, weights)
        return 2 * np.dot(X.T, predictions - y) / m
    
    # Initialize weights
    initial_weights = np.random.randn(3, 1) * 0.1
    
    # Hyperparameters
    learning_rate = 0.01
    epochs = 100
    batch_size = 32
    
    # 1. Batch Gradient Descent
    weights_batch = initial_weights.copy()
    losses_batch = []
    
    print("Running Batch Gradient Descent...")
    for epoch in range(epochs):
        loss = mse_loss(X_with_bias, y, weights_batch)
        losses_batch.append(loss)
        
        gradient = mse_gradient(X_with_bias, y, weights_batch)
        weights_batch -= learning_rate * gradient
    
    # 2. Stochastic Gradient Descent
    weights_sgd = initial_weights.copy()
    losses_sgd = []
    
    print("Running Stochastic Gradient Descent...")
    for epoch in range(epochs):
        epoch_loss = 0
        indices = np.random.permutation(n_samples)
        
        for i in indices:
            X_single = X_with_bias[i:i+1]
            y_single = y[i:i+1]
            
            loss = mse_loss(X_single, y_single, weights_sgd)
            epoch_loss += loss
            
            gradient = mse_gradient(X_single, y_single, weights_sgd)
            weights_sgd -= learning_rate * gradient
        
        losses_sgd.append(epoch_loss / n_samples)
    
    # 3. Mini-batch Gradient Descent
    weights_mini = initial_weights.copy()
    losses_mini = []
    
    print("Running Mini-batch Gradient Descent...")
    for epoch in range(epochs):
        epoch_loss = 0
        indices = np.random.permutation(n_samples)
        num_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_with_bias[batch_indices]
            y_batch = y[batch_indices]
            
            loss = mse_loss(X_batch, y_batch, weights_mini)
            epoch_loss += loss
            num_batches += 1
            
            gradient = mse_gradient(X_batch, y_batch, weights_mini)
            weights_mini -= learning_rate * gradient
        
        losses_mini.append(epoch_loss / num_batches)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Loss curves
    plt.subplot(2, 3, 1)
    plt.plot(losses_batch, label='Batch GD', linewidth=2)
    plt.plot(losses_mini, label='Mini-batch GD', linewidth=2)
    plt.plot(losses_sgd, label='SGD', linewidth=2, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Zoomed loss curves (last 50 epochs)
    plt.subplot(2, 3, 2)
    plt.plot(losses_batch[-50:], label='Batch GD', linewidth=2)
    plt.plot(losses_mini[-50:], label='Mini-batch GD', linewidth=2)
    plt.plot(losses_sgd[-50:], label='SGD', linewidth=2, alpha=0.7)
    plt.xlabel('Epoch (Last 50)')
    plt.ylabel('Loss')
    plt.title('Convergence Behavior')
    plt.legend()
    plt.grid(True)
    
    # Weight convergence
    true_weights = np.array([[0], [3], [2]])  # True weights (with bias)
    
    plt.subplot(2, 3, 3)
    methods = ['Batch GD', 'Mini-batch GD', 'SGD']
    final_weights = [weights_batch, weights_mini, weights_sgd]
    
    x_pos = np.arange(len(methods))
    width = 0.25
    
    for i in range(3):  # 3 weights (bias, w1, w2)
        actual_weights = [w[i, 0] for w in final_weights]
        plt.bar(x_pos + i * width, actual_weights, width, 
               label=f'Weight {i}', alpha=0.8)
        
        # Add true weight line
        plt.axhline(y=true_weights[i, 0], color='red', linestyle='--', alpha=0.7)
    
    plt.xlabel('Method')
    plt.ylabel('Weight Value')
    plt.title('Final Learned Weights')
    plt.xticks(x_pos + width, methods)
    plt.legend()
    
    # Training time simulation (iterations per epoch)
    plt.subplot(2, 3, 4)
    iterations_per_epoch = [1, n_samples, n_samples // batch_size]
    total_iterations = [iters * epochs for iters in iterations_per_epoch]
    
    plt.bar(methods, total_iterations, color=['blue', 'orange', 'green'], alpha=0.7)
    plt.xlabel('Method')
    plt.ylabel('Total Iterations')
    plt.title('Computational Cost (Total Iterations)')
    plt.yscale('log')
    
    # Add iteration counts as text
    for i, (method, count) in enumerate(zip(methods, total_iterations)):
        plt.text(i, count * 1.1, f'{count:,}', ha='center', va='bottom')
    
    # Memory usage simulation
    plt.subplot(2, 3, 5)
    memory_usage = [n_samples, 1, batch_size]  # Relative memory usage
    
    plt.bar(methods, memory_usage, color=['blue', 'orange', 'green'], alpha=0.7)
    plt.xlabel('Method')
    plt.ylabel('Memory Usage (Relative)')
    plt.title('Memory Requirements')
    
    # Noise in gradient estimates
    plt.subplot(2, 3, 6)
    gradient_variance = [0, 1.0, 0.1]  # Relative gradient variance
    
    plt.bar(methods, gradient_variance, color=['blue', 'orange', 'green'], alpha=0.7)
    plt.xlabel('Method')
    plt.ylabel('Gradient Variance (Relative)')
    plt.title('Gradient Estimation Noise')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nGradient Descent Variants Comparison:")
    print("=" * 50)
    
    print("1. Batch Gradient Descent:")
    print(f"   - Final loss: {losses_batch[-1]:.6f}")
    print(f"   - Total iterations: {epochs:,}")
    print(f"   - Memory usage: High (full dataset)")
    print(f"   - Convergence: Smooth, deterministic")
    
    print("\n2. Stochastic Gradient Descent:")
    print(f"   - Final loss: {losses_sgd[-1]:.6f}")
    print(f"   - Total iterations: {epochs * n_samples:,}")
    print(f"   - Memory usage: Low (single sample)")
    print(f"   - Convergence: Noisy, faster initially")
    
    print("\n3. Mini-batch Gradient Descent:")
    print(f"   - Final loss: {losses_mini[-1]:.6f}")
    print(f"   - Total iterations: {epochs * (n_samples // batch_size):,}")
    print(f"   - Memory usage: Medium (batch size: {batch_size})")
    print(f"   - Convergence: Balanced noise and stability")

compare_gradient_descent_variants()
```
