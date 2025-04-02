import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
import numpy.ma as ma
import os

# -----------------------------
# DATA
# -----------------------------
# Base method results
base = {
    "blood": [0.74, 0.7466666666666667, 0.76, 0.76, 0.74],
    "tic-tac-toe": [0.984375, 0.9947916666666666, 0.9947916666666666, 0.9947916666666666, 0.9895833333333334],
    "cmc": [0.5457627118644067, 0.5288135593220339, 0.5152542372881356, 0.5457627118644067, 0.5864406779661017],
    "heart": [0.842391304347826, 0.8913043478260869, 0.842391304347826, 0.875, 0.8695652173913043],
    "pc1": [0.9324324324324325, 0.918918918918919, 0.954954954954955, 0.9234234234234234, 0.918918918918919],
    "vehicle": [0.7411764705882353, 0.7705882352941177, 0.7823529411764706, 0.7176470588235294, 0.7294117647058823],
    "balance-scale": [0.864, 0.856, 0.848, 0.896, 0.848],
    "breast-w": [0.9785714285714285, 0.9714285714285714, 0.9428571428571428, 0.9571428571428572, 0.9428571428571428],
    "eucalyptus": [0.668918918918919, 0.668918918918919, 0.6216216216216216, 0.6418918918918919, 0.6621621621621622],
    "car": [0.9884393063583815, 0.9913294797687862, 0.9913294797687862, 0.9942196531791907, 0.9797687861271677],
    "credit-g": [0.765, 0.745, 0.73, 0.745, 0.785],
}

# LLMFE method results
llmfe = {
    "heart": [0.8260869565217391, 0.875, 0.842391304347826, 0.8695652173913043, 0.8641304347826086],
    "cmc": [0.49830508474576274, 0.5457627118644067, 0.5186440677966102, 0.511864406779661, 0.5423728813559322],
    "car": [1.0, 0.9942196531791907, 0.9942196531791907, 1.0, 0.9826589595375722],
    "eucalyptus": [0.6283783783783784, 0.7094594594594594, 0.6554054054054054, 0.6216216216216216, 0.6216216216216216],
    "credit-g": [0.75, 0.75, 0.74, 0.785, 0.77],
    "tic-tac-toe": [0.9947916666666666, 0.9947916666666666, 1.0, 0.9947916666666666, 0.984375],
    "pc1": [0.917, 0.945, 0.925, 0.938, 0.935],
    "breast-w": [0.9785714285714285, 0.9714285714285714, 0.9571428571428572, 0.9642857142857143, 0.9642857142857143],
    "blood": [0.7466666666666667, 0.76, 0.7733333333333333, 0.76, 0.74],
    "balance-scale": [0.992, 0.96, 0.976, 0.992, 0.968],
    "vehicle": [0.8058823529411765, 0.8, 0.8470588235294118, 0.8058823529411765, 0.8235294117647058],
}

# OCTree method results
octree = {
    "tic-tac-toe": [0.984375, 1.0, 0.984375, 0.9947916666666666, 0.9947916666666666],
    "blood": [0.7667, 0.78, 0.7533, 0.8066, 0.7533],
    "balance-scale": [0.888, 0.912, 0.872, 0.896, 0.928],
    "breast-w": [0.9571, 0.95, 0.9714, 0.9786, 0.9643],
    "vehicle": [0.7412, 0.7706, 0.8059, 0.7176, 0.7293],
    "car": [0.9884, 0.9913, 0.9913, 0.9942, 0.9798],
    "cmc": [0.5458, 0.5153, 0.5559, 0.5356, 0.5458],
    "eucalyptus": [0.6486, 0.6689, 0.6249, 0.6419, 0.6554],
    "heart": [0.8641, 0.9022, 0.8424, 0.875, 0.875],
    "pc1": [0.9324, 0.9009, 0.9595, 0.9324, 0.9324],
    "credit-g": [0.72, 0.769, 0.737, 0.755, 0.730],
}

# CAAFE method results
caafe = {
    "balance-scale": [1.0, 1.0, 0.976, 1.0, 1.0],
    "blood": [0.7267, 0.7533, 0.76, 0.7267, 0.74],
    "breast-w": [0.9714, 0.9429, 0.9571, 0.9643, 0.9429],
    "car": [0.9884, 0.9942, 1.0, 0.9884, 0.9855],
    "cmc": [0.5220, 0.5186, 0.5085, 0.5322, 0.5661],
    "credit-g": [0.755, 0.72, 0.715, 0.76, 0.775],
    "eucalyptus": [0.6081, 0.6622, 0.6486, 0.6486, 0.6554],
    "heart": [0.8424, 0.8859, 0.8587, 0.8424, 0.8587],
    "pc1": [0.9369, 0.9234, 0.9685, 0.9279, 0.9144],
    "tic-tac-toe": [0.9948, 1.0, 1.0, 1.0, 1.0],
    "vehicle": [0.7647, 0.7765, 0.7882, 0.7529, 0.7647],
}

# Define all methods
methods = {
    "Base": base,
    "LLMFE": llmfe,
    "OCTree": octree,
    "CAAFE": caafe
}

def create_heatmap(method1, method2, method1_name, method2_name):
    """Create a triangular heatmap comparing two methods."""
    # Get all dataset names
    datasets = sorted(list(set(method1.keys()) & set(method2.keys())))
    n_datasets = len(datasets)
    
    # Create a matrix to store p-values
    p_values = np.zeros((n_datasets, n_datasets))
    
    # Calculate p-values between each pair of datasets using independent t-test
    for i, dataset_i in enumerate(datasets):
        for j, dataset_j in enumerate(datasets):
            if i == j:
                # Same dataset comparison has p-value of 1
                p_values[i, j] = 1.0
            else:
                # Calculate p-value using independent t-test
                # Compare the performance difference between method1 and method2 for both datasets
                diff_i = np.array(method1[dataset_i]) - np.array(method2[dataset_i])
                diff_j = np.array(method1[dataset_j]) - np.array(method2[dataset_j])
                
                # Use t-test to compare the differences
                try:
                    _, p_value = ttest_ind(diff_i, diff_j, equal_var=False)  # Using Welch's t-test (unequal variances)
                    p_values[i, j] = p_value
                except ValueError:
                    # If the test fails (e.g., due to identical values), set p-value to 1
                    p_values[i, j] = 1.0
    
    # Create a DataFrame for better visualization
    p_value_df = pd.DataFrame(p_values, index=datasets, columns=datasets)
    
    # Define significance levels
    def significance_level(p):
        if p < 0.001:
            return 5  # Extremely significant (p < 0.001)
        elif p < 0.01:
            return 4  # Highly significant (0.001 ≤ p < 0.01)
        elif p < 0.05:
            return 3  # Significant (0.01 ≤ p < 0.05)
        elif p < 0.1:
            return 2  # Marginally significant (0.05 ≤ p < 0.1)
        else:
            return 1  # Not significant (p ≥ 0.1)
    
    # Convert p-values to significance levels
    significance_df = p_value_df.applymap(significance_level)
    
    # Create a custom colormap for significance levels
    cmap = plt.cm.get_cmap('RdYlGn_r', 5)  # 5 significance levels
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(p_values, dtype=bool))
    
    # Create the heatmap with the mask
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(significance_df, 
                    mask=mask,  # Apply the mask to show only lower triangle
                    annot=p_value_df.round(4), 
                    fmt='.4f', 
                    cmap=cmap, 
                    linewidths=0.5, 
                    linecolor='gray',
                    cbar_kws={'label': 'Significance Level', 
                              'ticks': [1.4, 2.2, 3.0, 3.8, 4.6],
                              'boundaries': [1, 2, 3, 4, 5, 6]})
    
    # Set colorbar tick labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(['Not significant\n(p ≥ 0.1)', 
                         'Marginally significant\n(0.05 ≤ p < 0.1)', 
                         'Significant\n(0.01 ≤ p < 0.05)', 
                         'Highly significant\n(0.001 ≤ p < 0.01)', 
                         'Extremely significant\n(p < 0.001)'])
    
    # Set title and labels
    plt.title(f'Statistical Significance Heatmap: {method1_name} vs {method2_name}', fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{method1_name}_vs_{method2_name}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_dataset_heatmaps():
    """Create heatmaps for individual datasets."""
    # Create output directory for heatmaps if it doesn't exist
    output_dir = "dataset_heatmaps"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all datasets
    all_datasets = sorted(list(base.keys()))
    
    # Create a heatmap for each dataset
    for dataset in all_datasets:
        # Check if all methods have this dataset
        if all(dataset in method_data for method_data in methods.values()):
            # Create a matrix to store p-values between methods for this dataset
            method_names = list(methods.keys())
            n_methods = len(method_names)
            p_values = np.zeros((n_methods, n_methods))
            
            # Calculate p-values between each pair of methods using t-test
            for i, method1 in enumerate(method_names):
                for j, method2 in enumerate(method_names):
                    if i == j:
                        # Same method comparison has p-value of 1
                        p_values[i, j] = 1.0
                    else:
                        # Get results for both methods for this dataset
                        results1 = np.array(methods[method1][dataset])
                        results2 = np.array(methods[method2][dataset])
                        
                        # Use independent t-test to compare the methods
                        try:
                            _, p_value = ttest_ind(results1, results2, equal_var=False)
                            p_values[i, j] = p_value
                        except ValueError:
                            # If the test fails, set p-value to 1
                            p_values[i, j] = 1.0
            
            # Create a DataFrame for better visualization
            p_value_df = pd.DataFrame(p_values, index=method_names, columns=method_names)
            
            # Convert p-values to significance levels
            significance_df = p_value_df.applymap(significance_level)
            
            # Create a custom colormap for significance levels
            cmap = plt.cm.get_cmap('RdYlGn_r', 5)  # 5 significance levels
            
            # Create a mask for the upper triangle
            mask = np.triu(np.ones_like(p_values, dtype=bool))
            
            # Create the triangular heatmap
            plt.figure(figsize=(8, 6))
            ax = sns.heatmap(significance_df, 
                            mask=mask,  # Apply the mask to show only lower triangle
                            annot=p_value_df.round(4), 
                            fmt='.4f', 
                            cmap=cmap, 
                            linewidths=0.5, 
                            linecolor='gray',
                            cbar_kws={'label': 'Significance Level', 
                                      'ticks': [1.4, 2.2, 3.0, 3.8, 4.6],
                                      'boundaries': [1, 2, 3, 4, 5, 6]})
            
            # Set colorbar tick labels
            cbar = ax.collections[0].colorbar
            cbar.set_ticklabels(['Not significant\n(p ≥ 0.1)', 
                                 'Marginally significant\n(0.05 ≤ p < 0.1)', 
                                 'Significant\n(0.01 ≤ p < 0.05)', 
                                 'Highly significant\n(0.001 ≤ p < 0.01)', 
                                 'Extremely significant\n(p < 0.001)'])
            
            # Set title and labels
            plt.title(f'Statistical Significance Heatmap: {dataset}', fontsize=14)
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(f'{output_dir}/{dataset}_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Heatmap for {dataset} has been generated.")
        else:
            print(f"Skipping {dataset} as it's not available in all methods.")

# Helper function for significance levels
def significance_level(p):
    if p < 0.001:
        return 5  # Extremely significant (p < 0.001)
    elif p < 0.01:
        return 4  # Highly significant (0.001 ≤ p < 0.01)
    elif p < 0.05:
        return 3  # Significant (0.01 ≤ p < 0.05)
    elif p < 0.1:
        return 2  # Marginally significant (0.05 ≤ p < 0.1)
    else:
        return 1  # Not significant (p ≥ 0.1)

def create_method_comparison_heatmap():
    """Create a heatmap comparing all methods across all datasets."""
    # Get common datasets across all methods
    common_datasets = set(base.keys())
    for method_data in methods.values():
        common_datasets = common_datasets.intersection(set(method_data.keys()))
    common_datasets = sorted(list(common_datasets))
    
    # Create a matrix to store p-values between methods
    method_names = list(methods.keys())
    n_methods = len(method_names)
    p_values = np.zeros((n_methods, n_methods))
    
    # Calculate p-values between each pair of methods using paired t-test
    for i, method1 in enumerate(method_names):
        for j, method2 in enumerate(method_names):
            if i == j:
                # Same method comparison has p-value of 1
                p_values[i, j] = 1.0
            else:
                # Collect all results for each method across all datasets
                all_results1 = []
                all_results2 = []
                
                # For each dataset, collect the paired results
                for dataset in common_datasets:
                    all_results1.extend(methods[method1][dataset])
                    all_results2.extend(methods[method2][dataset])
                
                # Use paired t-test to compare the methods
                try:
                    _, p_value = ttest_ind(all_results1, all_results2)
                    p_values[i, j] = p_value
                except ValueError:
                    # If the test fails, set p-value to 1
                    p_values[i, j] = 1.0
    
    # Create a DataFrame for better visualization
    p_value_df = pd.DataFrame(p_values, index=method_names, columns=method_names)
    
    # Convert p-values to significance levels
    significance_df = p_value_df.applymap(significance_level)
    
    # Create a custom colormap for significance levels
    cmap = plt.cm.get_cmap('RdYlGn_r', 5)  # 5 significance levels
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(p_values, dtype=bool))
    
    # Create the triangular heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(significance_df, 
                    mask=mask,  # Apply the mask to show only lower triangle
                    annot=p_value_df.round(4), 
                    fmt='.4f', 
                    cmap=cmap, 
                    linewidths=0.5, 
                    linecolor='gray',
                    cbar_kws={'label': 'Significance Level', 
                              'ticks': [1.4, 2.2, 3.0, 3.8, 4.6],
                              'boundaries': [1, 2, 3, 4, 5, 6]})
    
    # Set colorbar tick labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(['Not significant\n(p ≥ 0.1)', 
                         'Marginally significant\n(0.05 ≤ p < 0.1)', 
                         'Significant\n(0.01 ≤ p < 0.05)', 
                         'Highly significant\n(0.001 ≤ p < 0.01)', 
                         'Extremely significant\n(p < 0.001)'])
    
    # Set title and labels
    plt.title('Statistical Significance Heatmap: Method Comparison Across All Datasets', fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('method_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Method comparison heatmap has been generated.")

if __name__ == "__main__":
    # Create comparison heatmaps
    create_heatmap(llmfe, base, "LLMFE", "Base")
    create_heatmap(octree, base, "OCTree", "Base")
    create_heatmap(caafe, base, "CAAFE", "Base")
    
    # Create dataset-specific heatmaps
    create_dataset_heatmaps()
    
    # Create method comparison heatmap
    create_method_comparison_heatmap()
    
    print("All heatmaps have been generated.")