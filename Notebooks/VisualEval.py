import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Function to plot raw time series of original and generated datasets
def plot_raw_time_series(original_data, generated_data, feature_name, time_col):
    """
    Plots the raw time series for the original and generated datasets.
    :param original_data: pd.DataFrame or np.array, original dataset with time steps.
    :param generated_data: pd.DataFrame or np.array, generated dataset with time steps.
    :param feature_name: str, name of the feature to be plotted.
    :param time_col: str, name of the time column (if DataFrame).
    """
    plt.figure(figsize=(10, 6))
    
    if isinstance(original_data, pd.DataFrame):
        plt.plot(original_data.index, original_data[feature_name], label='Original')
        plt.plot(generated_data.index, generated_data[feature_name], label='Generated')
    else:
        plt.plot(original_data[:, 0], original_data[:, 1], label='Original')
        plt.plot(generated_data[:, 0], generated_data[:, 1], label='Generated')
    
    plt.xlabel('Time')
    plt.ylabel(feature_name)
    plt.title(f'Raw Time Series Comparison: {feature_name}')
    plt.legend()
    plt.show()

# Function to plot histograms for a given feature in original and generated datasets
def plot_histograms(original_data, generated_data, feature_name):
    """
    Plots histograms for a specified feature from both original and generated datasets.
    :param original_data: pd.DataFrame or np.array, original dataset.
    :param generated_data: pd.DataFrame or np.array, generated dataset.
    :param feature_name: str, name of the feature to be plotted.
    """
    plt.figure(figsize=(10, 6))
    
    if isinstance(original_data, pd.DataFrame):
        plt.hist(original_data[feature_name], bins=50, alpha=0.5, label='Original', density=True)
        plt.hist(generated_data[feature_name], bins=50, alpha=0.5, label='Generated', density=True)
    else:
        plt.hist(original_data, bins=50, alpha=0.5, label='Original', density=True)
        plt.hist(generated_data, bins=50, alpha=0.5, label='Generated', density=True)
    
    plt.xlabel(feature_name)
    plt.ylabel('Density')
    plt.title(f'Histogram Comparison: {feature_name}')
    plt.legend()
    plt.show()

# Function to create box plots comparing original and generated datasets
def plot_boxplots(original_data, generated_data, feature_name):
    """
    Plots boxplots for the specified feature from both original and generated datasets.
    :param original_data: pd.DataFrame or np.array, original dataset.
    :param generated_data: pd.DataFrame or np.array, generated dataset.
    :param feature_name: str, name of the feature to be plotted.
    """
    data_to_plot = []
    labels = []
    
    if isinstance(original_data, pd.DataFrame):
        data_to_plot.append(original_data[feature_name])
        labels.append('Original')
        data_to_plot.append(generated_data[feature_name])
        labels.append('Generated')
    else:
        data_to_plot.append(original_data)
        labels.append('Original')
        data_to_plot.append(generated_data)
        labels.append('Generated')
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel(feature_name)
    plt.title(f'Box Plot Comparison: {feature_name}')
    plt.show()

# Function to apply PCA and plot a 2D projection for original and generated datasets
def plot_pca(original_data, generated_data):
    """
    Applies PCA to both original and generated datasets and plots the 2D projection.
    :param original_data: pd.DataFrame or np.array, original dataset.
    :param generated_data: pd.DataFrame or np.array, generated dataset.
    """
    pca = PCA(n_components=2)
    
    original_pca = pca.fit_transform(original_data)
    generated_pca = pca.fit_transform(generated_data)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(original_pca[:, 0], original_pca[:, 1], alpha=0.6, label='Original', c='blue')
    plt.scatter(generated_pca[:, 0], generated_pca[:, 1], alpha=0.6, label='Generated', c='red')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA 2D Projection: Original vs Generated')
    plt.legend()
    plt.show()

# Function to apply t-SNE and plot a 2D projection for original and generated datasets
def plot_tsne(original_data, generated_data):
    """
    Applies t-SNE to both original and generated datasets and plots the 2D projection.
    :param original_data: pd.DataFrame or np.array, original dataset.
    :param generated_data: pd.DataFrame or np.array, generated dataset.
    """
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    
    original_tsne = tsne.fit_transform(original_data)
    generated_tsne = tsne.fit_transform(generated_data)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(original_tsne[:, 0], original_tsne[:, 1], alpha=0.6, label='Original', c='blue')
    plt.scatter(generated_tsne[:, 0], generated_tsne[:, 1], alpha=0.6, label='Generated', c='red')
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE 2D Projection: Original vs Generated')
    plt.legend()
    plt.show()


def plot_boxplots_many(feature_name, *datasets):
    """
    Plots boxplots for a specified feature from an arbitrary number of datasets.
    :param feature_name: str, name of the feature to be plotted.
    :param datasets: List of tuples containing dataset name and data.
                     e.g., ('Original', original_data), ('Generated', generated_data)
    """
    data_to_plot = []
    labels = []

    for name, dataset in datasets:
        if isinstance(dataset, pd.DataFrame):
            data_to_plot.append(dataset[feature_name])
        else:
            data_to_plot.append(dataset)
        labels.append(name)
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel(feature_name)
    plt.title(f'Box Plot Comparison: {feature_name}')
    plt.show()


def plot_boxplots_means(*datasets):
    """
    Plots boxplots comparing the means of all features from an arbitrary number of datasets.
    :param datasets: List of tuples containing dataset name and data.
                     e.g., ('Original', original_data), ('Generated', generated_data)
                     The data should be in pd.DataFrame or np.array format.
    """
    data_to_plot = []
    labels = []

    for name, dataset in datasets:
        # Ensure the dataset is in DataFrame format for easier processing
        if isinstance(dataset, pd.DataFrame):
            means = dataset.mean(axis=0)  # Calculate the mean across all features
        else:
            means = np.mean(dataset, axis=0)  # If numpy array, compute mean across axis 0
        data_to_plot.append(means)
        labels.append(name)
    
    # Convert lists to arrays for plotting
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel('Mean of Features')
    plt.title('Box Plot Comparison of Feature Means Across Datasets')
    plt.show()
    
def plot_histograms_means(*datasets, bins=50):
    """
    Plots histograms comparing the means of all features from an arbitrary number of datasets.
    :param datasets: List of tuples containing dataset name and data.
                     e.g., ('Original', original_data), ('Generated', generated_data)
                     The data should be in pd.DataFrame or np.array format.
    :param bins: int, number of bins for the histogram.
    """
    plt.figure(figsize=(10, 6))

    for name, dataset in datasets:
        # Ensure the dataset is in DataFrame format for easier processing
        if isinstance(dataset, pd.DataFrame):
            means = dataset.mean(axis=0)  # Calculate the mean across all features
        else:
            means = np.mean(dataset, axis=0)  # If numpy array, compute mean across axis 0

        # Plot histogram of the means
        plt.hist(means, bins=bins, alpha=0.5, label=name, density=True)
    
    plt.xlabel('Mean of Features')
    plt.ylabel('Density')
    plt.title('Histogram Comparison of Feature Means Across Datasets')
    plt.legend()
    plt.show()