import numpy as np
import pandas as pd
from sklearn import metrics
from scipy.stats import skew, kurtosis
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from sklearn.neural_network import MLPClassifier
from scipy.stats import ttest_ind

# MMD Linear Kernel
def mmd_linear(X, Y):
    """MMD using linear kernel"""
    delta = X.mean(axis=0) - Y.mean(axis=0)
    return delta.dot(delta.T)

# MMD RBF Kernel
def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel"""
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

# MMD Polynomial Kernel
def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel"""
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()

# Basic Statistics Comparison
def basic_statistics_comparison(df_original, df_generated):
    stats = pd.DataFrame(columns=['mean', 'variance', 'skewness', 'kurtosis'])

    for column in df_original.columns:
        stats = pd.concat([stats, pd.DataFrame({
            'mean': [np.abs(df_original[column].mean() - df_generated[column].mean())],
            'variance': [np.abs(df_original[column].var() - df_generated[column].var())],
            'skewness': [np.abs(skew(df_original[column]) - skew(df_generated[column]))],
            'kurtosis': [np.abs(kurtosis(df_original[column]) - kurtosis(df_generated[column]))]
        })], ignore_index=True)
        
    return stats['mean'].mean(), stats['variance'].mean(), \
            stats['skewness'].mean(), stats['kurtosis'].mean()

# KS Test Comparison
def ks_test_comparison(df_original, df_generated):
    ks_results = pd.DataFrame(columns=['ks_statistic', 'p_value'])
    
    for column in df_original.columns:
        ks_stat, p_value = ks_2samp(df_original[column], df_generated[column])
        ks_results = pd.concat([
            ks_results,
            pd.DataFrame({
                'ks_statistic': [ks_stat],
                'p_value': [p_value]
            })
        ], ignore_index=True)
    
    return ks_results['ks_statistic'].mean(), ks_results['p_value'].mean()

# Wasserstein Distance Comparison
def wasserstein_distance_comparison(df_original, df_generated):
    wasserstein_results = pd.DataFrame(columns=['wasserstein_distance'])
    
    for column in df_original.columns:
        distance = wasserstein_distance(df_original[column], df_generated[column])
        wasserstein_results = pd.concat([
            wasserstein_results,
            pd.DataFrame({
                'wasserstein_distance': [distance]
            })
        ], ignore_index=True)
    
    return wasserstein_results['wasserstein_distance'].mean()

# Jensen-Shannon Divergence Comparison
def jensen_shannon_divergence(df_original, df_generated):
    js_results = pd.DataFrame(columns=['js_divergence'])
    
    for column in df_original.columns:
        hist_original, _ = np.histogram(df_original[column], bins=20, density=True)
        hist_generated, _ = np.histogram(df_generated[column], bins=20, density=True)
        
        js_div = jensenshannon(hist_original + 1e-10, hist_generated + 1e-10)
        js_results = pd.concat([
            js_results,
            pd.DataFrame({
                'js_divergence': [js_div]
            })
        ], ignore_index=True)
    
    return js_results['js_divergence'].mean()

# Train on Original, Test on Generated
def train_on_original_test_on_generated(df_original, df_generated, target_column):
    X_original = df_original.drop(columns=[target_column])
    y_original = df_original[target_column]
    
    X_generated = df_generated.drop(columns=[target_column])
    y_generated = df_generated[target_column]
    
    model = RandomForestClassifier()
    model.fit(X_original, y_original)
    
    y_pred = model.predict(X_generated)
    accuracy = accuracy_score(y_generated, y_pred)
    
    return accuracy

# Correlation Coefficient Comparison
def correlation_comparison(df_original, df_generated):
    corr_original = df_original.corr()
    corr_generated = df_generated.corr()
    
    corr_diff = (corr_original - corr_generated).abs().mean().mean()
    
    return corr_diff

# Inception Score (IS)
def inception_score(df_original, df_generated, n_splits=10):
    """
    Calculate the Inception Score for generated samples.
    
    A classifier (neural network) is used to predict class probabilities, and
    the score is based on the divergence between individual sample predictions
    and the marginal distribution across all samples.
    """
    model = MLPClassifier(max_iter=500)
    
    df = pd.concat([df_original, df_generated], ignore_index=True).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)
    labels = df.pop('gen')
    
    model.fit(df, labels)
    preds = model.predict_proba(df)
    
    scores = []
    for i in range(n_splits):
        split_preds = preds[i::n_splits]
        marginal = np.mean(split_preds, axis=0)
        kl_div = split_preds * np.log(split_preds / marginal)
        kl_sum = np.mean(np.sum(kl_div, axis=1))
        res = np.exp(kl_sum)
        res = np.where(np.isnan(res), 0., res)
        scores.append(res)
        
    return np.mean(scores), np.std(scores)

# Evaluation Function
def evaluate_datasets(df_original, df_generated, target_column=None):
    report = {}
    
    df_original_stats = df_original.drop(columns='gen')    
    df_generated_stats = df_generated.drop(columns='gen')    
    # 1. Basic Statistics
    report['mean_difference'], report['variance_difference'], report['skewness_difference'], report['kurtosis_difference'] = basic_statistics_comparison(df_original_stats, df_generated_stats)
    
    # 2. KS Test
    report['ks_test'], report['ks_test_p_value'] = ks_test_comparison(df_original_stats, df_generated_stats)
    
    # 3. Wasserstein Distance
    report['wasserstein_distance'] = wasserstein_distance_comparison(df_original_stats, df_generated_stats)
    
    # 4. Jensen-Shannon Divergence
    report['js_divergence'] = jensen_shannon_divergence(df_original_stats, df_generated_stats)
    
    # 5. Correlation Difference
    report['correlation_difference'] = correlation_comparison(df_original_stats, df_generated_stats)

    # 6. MMD Metrics
    X_original = df_original_stats.to_numpy()
    X_generated = df_generated_stats.to_numpy()
    
    report['mmd_linear'] = mmd_linear(X_original, X_generated)
    report['mmd_rbf'] = mmd_rbf(X_original, X_generated)
    # report['mmd_poly'] = mmd_poly(X_original, X_generated)
    
    # 7. Inception Score
    report['inception_score_mean'], report['inception_score_std'] = inception_score(df_original, df_generated)

    t_stat, t_stat_p_value = ttest_ind(df_original_stats, df_generated_stats)
    report['t_stat'], report['t_stat_p_value'] = np.mean(t_stat), np.mean(t_stat_p_value)

    # 8. Optional: Model-based evaluation (if target column is provided)
    if target_column:
        report['model_accuracy'] = train_on_original_test_on_generated(df_original, df_generated, target_column)
        report['model_accuracy_on_Gen'] = train_on_original_test_on_generated(df_generated, df_original, target_column)
        
        X = pd.concat([df_original, df_generated]).sample(300).reset_index(drop=True)
        report['model_accuracy_train_on_both'] = train_on_original_test_on_generated(X, df_original, target_column)
        report['model_accuracy_train_on_both_Gen'] = train_on_original_test_on_generated(X, df_generated, target_column)
        
    return report
