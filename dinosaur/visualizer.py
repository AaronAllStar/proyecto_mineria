import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Visualizer:
    """
    Visualization engine for Dinosaur.
    Automatically generates plots for EDA.
    """
    
    @staticmethod
    def plot_distributions(df, columns=None, max_cols=3):
        """
        Plots histograms/densities for numeric columns.
        """
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns[:9] # Limit to top 9
            
        n_cols = len(columns)
        rows = (n_cols + max_cols - 1) // max_cols
        
        fig, axes = plt.subplots(rows, max_cols, figsize=(15, 5 * rows))
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            sns.histplot(df[col], kde=True, ax=axes[i], color='teal')
            axes[i].set_title(f'Distribution of {col}', fontweight='bold')
            
        # Hide empty axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlation(df):
        """
        Plots a heatmap of numeric correlations.
        """
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            print("No numeric columns found for correlation.")
            return
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Dinosaur Correlation Heatmap', fontsize=15, pad=20)
        plt.show()

    @staticmethod
    def plot_missing(df):
        """
        Visualizes missing values in the dataset.
        """
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title('Dinosaur Missing Values Matrix', fontsize=15)
        plt.show()
