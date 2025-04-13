import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit app title
st.title("Diabetes Dataset Analysis with PCA and Regression")

# File uploader for the diabetes dataset
uploaded_file = st.file_uploader("Upload diabetes.csv", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    diabetes = pd.read_csv(uploaded_file)

    # Display a quick look at the data
    st.subheader("Dataset Preview")
    st.write(diabetes.head())
    st.write(f"Dataset shape: {diabetes.shape}")

    # Feature distributions
    st.subheader("Feature Distributions")
    fig, ax = plt.subplots(figsize=(12, 10))
    diabetes.hist(ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    # Prepare features and target
    X = diabetes.drop('Outcome', axis=1)
    y = diabetes['Outcome']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA()
    pca.fit(X_scaled)

    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Plot explained variance
    st.subheader("PCA Explained Variance")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.5,
           label='Individual explained variance')
    ax.step(range(1, len(cumulative_variance)+1), cumulative_variance, where='mid',
            label='Cumulative explained variance')
    ax.axhline(y=0.95, color='r', linestyle='--', label='95% Variance threshold')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('Explained Variance by Components')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # Display cumulative variance
    st.subheader("Cumulative Variance by Components")
    for i, var in enumerate(cumulative_variance):
        st.write(f"Components: {i+1}, Cumulative Variance: {var:.4f}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Transform data using PCA with optimal components
    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Model 1: Linear Regression on original data
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    # Model 1: Linear Regression on PCA data
    lr_pca = LinearRegression()
    lr_pca.fit(X_train_pca, y_train)
    y_pred_lr_pca = lr_pca.predict(X_test_pca)
    mse_lr_pca = mean_squared_error(y_test, y_pred_lr_pca)
    r2_lr_pca = r2_score(y_test, y_pred_lr_pca)

    # Model 2: Ridge Regression on original data
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)

    # Model 2: Ridge Regression on PCA data
    ridge_pca = Ridge(alpha=1.0)
    ridge_pca.fit(X_train_pca, y_train)
    y_pred_ridge_pca = ridge_pca.predict(X_test_pca)
    mse_ridge_pca = mean_squared_error(y_test, y_pred_ridge_pca)
    r2_ridge_pca = r2_score(y_test, y_pred_ridge_pca)

    # Prepare results for plotting
    models = ['Linear Regression', 'Linear Regression\nwith PCA',
              'Ridge Regression', 'Ridge Regression\nwith PCA']
    mse_values = [mse_lr, mse_lr_pca, mse_ridge, mse_ridge_pca]
    r2_values = [r2_lr, r2_lr_pca, r2_ridge, r2_ridge_pca]

    # Plot MSE and R² comparison
    st.subheader("Model Performance Comparison")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # MSE comparison
    sns.barplot(x=models, y=mse_values, ax=ax1)
    ax1.set_title('Mean Squared Error Comparison')
    ax1.set_ylabel('MSE')
    ax1.tick_params(axis='x', rotation=45)

    # R² comparison
    sns.barplot(x=models, y=r2_values, ax=ax2)
    ax2.set_title('R² Score Comparison')
    ax2.set_ylabel('R²')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("Please upload the diabetes.csv file to proceed.")
