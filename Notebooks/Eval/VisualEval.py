import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

visualization_metrics.py

Note: Use PCA or tSNE for generated and original data visualization
"""

def visualization(ori_data, generated_data, analysis):
  """Using PCA or tSNE for generated and original data visualization.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """  
  # Analysis sample size (for faster computation)
  anal_sample_no = min([1000, len(ori_data)])
  idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    
  # Data preprocessing
  ori_data = np.asarray(ori_data)
  generated_data = np.asarray(generated_data)  
  
  ori_data = ori_data[idx]
  generated_data = generated_data[idx]
  
  no, seq_len, dim = ori_data.shape  
  
  for i in range(anal_sample_no):
    if (i == 0):
      prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
      prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
    else:
      prep_data = np.concatenate((prep_data, 
                                  np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
      prep_data_hat = np.concatenate((prep_data_hat, 
                                      np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
    
  # Visualization parameter        
  colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    
    
  if analysis == 'pca':
    # PCA Analysis
    pca = PCA(n_components = 2)
    pca.fit(prep_data)
    pca_results = pca.transform(prep_data)
    pca_hat_results = pca.transform(prep_data_hat)
    
    # Plotting
    f, ax = plt.subplots(1)    
    plt.scatter(pca_results[:,0], pca_results[:,1],
                c = colors[:anal_sample_no], alpha = 0.6, label = "Original")
    plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                c = colors[anal_sample_no:], alpha = 0.4, label = "Synthetic")
  
    ax.legend()  
    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.show()
    
  elif analysis == 'tsne':
    
    # Do t-SNE Analysis together       
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)
    
    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(prep_data_final)
      
    # Plotting
    f, ax = plt.subplots(1)
      
    plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
  
    ax.legend()
      
    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.show()    

# Function to plot raw time series of original and generated datasets
def plot_raw_time_series(original_data, generated_data, feature_name, time_col,
                  original_label='Original',
                  generated_label='Generated'):
    """
    Plots the raw time series for the original and generated datasets.
    :param original_data: pd.DataFrame or np.array, original dataset with time steps.
    :param generated_data: pd.DataFrame or np.array, generated dataset with time steps.
    :param feature_name: str, name of the feature to be plotted.
    :param time_col: str, name of the time column (if DataFrame).
    """
    plt.figure(figsize=(10, 6))
    
    if isinstance(original_data, pd.DataFrame):
        plt.plot(original_data.index, original_data[feature_name], label=original_label)
        plt.plot(generated_data.index, generated_data[feature_name], label=generated_label)
    else:
        plt.plot(original_data[:, 0], original_data[:, 1], label=original_label)
        plt.plot(generated_data[:, 0], generated_data[:, 1], label=generated_label)
    
    plt.xlabel('Time')
    plt.ylabel(feature_name)
    plt.title(f'Raw Time Series Comparison: {feature_name}')
    plt.legend()
    plt.show()

# Function to plot histograms for a given feature in original and generated datasets
def plot_histograms(original_data, generated_data, feature_name,
                  original_label='Original',
                  generated_label='Generated'
                  ):
    """
    Plots histograms for a specified feature from both original and generated datasets.
    :param original_data: pd.DataFrame or np.array, original dataset.
    :param generated_data: pd.DataFrame or np.array, generated dataset.
    :param feature_name: str, name of the feature to be plotted.
    """
    plt.figure(figsize=(10, 6))
    
    if isinstance(original_data, pd.DataFrame):
        plt.hist(original_data[feature_name], bins=50, alpha=0.5, label=original_label, density=True)
        plt.hist(generated_data[feature_name], bins=50, alpha=0.5, label=generated_label, density=True)
    else:
        plt.hist(original_data, bins=50, alpha=0.5, label=original_label, density=True)
        plt.hist(generated_data, bins=50, alpha=0.5, label=generated_label, density=True)
    
    plt.xlabel(feature_name)
    plt.ylabel('Density')
    plt.title(f'Histogram Comparison: {feature_name}')
    plt.legend()
    plt.show()

# Function to create box plots comparing original and generated datasets
def plot_boxplots(original_data, generated_data, feature_name,
                  original_label='Original',
                  generated_label='Generated'
                  ):
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
        labels.append(original_label)
        data_to_plot.append(generated_data[feature_name])
        labels.append(generated_label)
    else:
        data_to_plot.append(original_data)
        labels.append(original_label)
        data_to_plot.append(generated_data)
        labels.append(generated_label)
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel(feature_name)
    plt.title(f'Box Plot Comparison: {feature_name}')
    plt.show()

# Function to apply PCA and plot a 2D projection for original and generated datasets
def plot_pca(original_data, generated_data, 
             original_label='Original', 
             generated_label='Generated', 
             component_to_plot=(0, 1),
             custom_title=''):
    """
    Applies PCA to both original and generated datasets and plots the 2D projection.
    :param original_data: pd.DataFrame or np.array, original dataset.
    :param generated_data: pd.DataFrame or np.array, generated dataset.
    """
    pca = PCA(n_components=0.99)
    
    # original_pca = pca.fit_transform(original_data)
    # generated_pca = pca.fit_transform(generated_data)
    plt.figure(figsize=(10, 6))
    original_pca, generated_pca = [], []
    
    original_pca.append(pca.fit_transform(original_data[0]))
    plt.scatter(original_pca[0][:, component_to_plot[0]], original_pca[0][:, component_to_plot[1]], alpha=0.6, label=original_label, c='gray')
    for i in range(1, len(original_data)):
        original_pca.append(pca.fit_transform(original_data[i]))
        plt.scatter(original_pca[i][:, component_to_plot[0]], original_pca[i][:, component_to_plot[1]], alpha=0.6, c='gray')
        
    generated_pca.append(pca.transform(generated_data[0]))
    plt.scatter(generated_pca[0][:, component_to_plot[0]], generated_pca[0][:, component_to_plot[1]], label=generated_label, alpha=0.4, c='red')
    
    for i in range(1, len(generated_data)):
        generated_pca.append(pca.transform(generated_data[i]))
        plt.scatter(generated_pca[i][:, component_to_plot[0]], generated_pca[i][:, component_to_plot[1]], alpha=0.4, c='red')
    
    print(pca.n_components_)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    if custom_title!='':
        plt.title(custom_title)
    else:
        plt.title(f'PCA Projection: {original_label} vs {generated_label}')
    plt.legend()
    plt.show()

# Function to apply t-SNE and plot a 2D projection for original and generated datasets
def plot_tsne(original_data, generated_data, 
              original_label='Original',
              generated_label='Generated',
              custom_title=''):
    """
    Applies t-SNE to both original and generated datasets and plots the 2D projection.
    :param original_data: pd.DataFrame or np.array, original dataset.
    :param generated_data: pd.DataFrame or np.array, generated dataset.
    """
    length = len(generated_data)
    # tsne = TSNE(n_components=2, perplexity=15, n_iter=1000, random_state=42)
    if length<=1500:
        tsne = TSNE(n_components=2, perplexity=10, n_iter=1000, random_state=42)
    elif 5000>length>1500:
        tsne = TSNE(n_components=2, perplexity=15, n_iter=1000, random_state=42)
    else:
        tsne = TSNE(n_components=2, perplexity=50, n_iter=2000, metric='cosine', random_state=42)
    
    original_tsne = tsne.fit_transform(original_data)
    generated_tsne = tsne.fit_transform(generated_data)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(original_tsne[:, 0], original_tsne[:, 1], alpha=0.6, label=original_label, c='gray')
    plt.scatter(generated_tsne[:, 0], generated_tsne[:, 1], alpha=0.4, label=generated_label, c='red')
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    if custom_title!='':
        plt.title(custom_title)
    else:
        plt.title(f't-SNE Projection: {original_label} vs {generated_label}')
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