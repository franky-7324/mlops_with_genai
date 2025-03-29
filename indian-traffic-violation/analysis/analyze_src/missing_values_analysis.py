from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs a complete missing values analysis by identifying and visualizing missing values.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        pass


class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Displays missing values count in a visually appealing way in Jupyter Notebook.
        """
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

        if missing_values.empty:
            display(Markdown("✅ **No missing values found in the dataset!**"))
            return

        # Convert missing values to a DataFrame for better styling
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Values': missing_values.values,
            'Percentage (%)': (missing_values.values / len(df) * 100).round(2)
        })

        display(Markdown("## 📊 **Missing Values Summary**"))
        display(
            missing_df.style
            .background_gradient(cmap="Oranges")
            .set_caption("🔍 Missing Data Overview")
            .hide(axis="index")
        )

    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Creates a heatmap for missing values.
        """
        if df.isnull().sum().sum() == 0:
            return

        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
        plt.title("🔍 Missing Values Heatmap", fontsize=14)
        plt.show()
