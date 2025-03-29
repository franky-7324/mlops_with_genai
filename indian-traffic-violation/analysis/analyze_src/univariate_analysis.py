from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import ollama
from IPython.display import Markdown, display

# Fix missing font glyph warning
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


# Set seaborn theme for better visuals
sns.set_theme(style="whitegrid")


# Abstract Base Class for Univariate Analysis Strategy
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        pass


# Concrete Strategy for Numerical Features
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots histogram & KDE for numerical feature and generates AI summary.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30, color="royalblue", edgecolor="black", alpha=0.7)
        plt.title(f"📊 Distribution of {feature}", fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.show()

        # Extract feature statistics
        summary_stats = df[feature].describe().to_dict()
        self.generate_insight(feature, summary_stats)

    def generate_insight(self, feature, stats):
        """
        Uses Ollama to generate a summary of the univariate analysis.
        """
        prompt = f"""
        Given the numerical feature: {feature}, here are its statistics:

        - Count: {stats['count']}
        - Mean: {stats['mean']:.2f}
        - Standard Deviation: {stats['std']:.2f}
        - Min: {stats['min']}
        - 25% Quartile: {stats['25%']}
        - Median (50%): {stats['50%']}
        - 75% Quartile: {stats['75%']}
        - Max: {stats['max']}

        Provide a concise summary explaining the distribution, key insights, potential outliers, and overall trends.
        """

        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
        insight = response['message']['content']
        
        # Display formatted Markdown output
        display(Markdown(f"### 📌 AI-Generated Insights for **{feature}**\n\n{insight}"))


# Concrete Strategy for Categorical Features
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots bar chart for categorical feature and generates AI summary.
        """
        plt.figure(figsize=(10, 6))
        sorted_categories = df[feature].value_counts().sort_values().index
        
        # Fix FutureWarning by explicitly setting `hue` and `legend`
        sns.countplot(x=feature, hue=feature, data=df, palette="coolwarm", order=sorted_categories, legend=False)

        plt.title(f"📊 Distribution of {feature}", fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.show()

        # Extract top categories
        top_categories = df[feature].value_counts().to_dict()
        self.generate_insight(feature, top_categories)

    def generate_insight(self, feature, categories):
        """
        Uses Ollama to generate insights for categorical features.
        """
        prompt = f"""
        Given the categorical feature: {feature}, here are the top categories and their counts:

        {categories}

        Provide a summary explaining key trends, dominant categories, and any interesting insights.
        """

        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
        insight = response['message']['content']

        # Display formatted Markdown output
        display(Markdown(f"### 📌 AI-Generated Insights for **{feature}**\n\n{insight}"))


# Context Class for Univariate Analysis
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        self._strategy.analyze(df, feature)
