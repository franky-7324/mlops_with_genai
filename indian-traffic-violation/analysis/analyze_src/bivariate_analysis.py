from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ollama
from IPython.display import Markdown, display

# Abstract Base Class for Bivariate Analysis Strategy
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        pass

# Concrete Strategy for Numerical vs Numerical Analysis
class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df, alpha=0.7, edgecolor=None)
        plt.title(f"📊 {feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.grid(True)
        plt.show()
        self.generate_ai_insight(feature1, feature2)

    def generate_ai_insight(self, feature1, feature2):
        summary_prompt = f"""
        Analyze the relationship between {feature1} and {feature2}. 
        - Discuss correlation, trends, and any notable outliers.
        - Provide key insights in a clear, professional tone.
        - Exclude code from the response.
        - Format the response in Markdown.
        """

        response = ollama.chat(model='mistral', messages=[{"role": "user", "content": summary_prompt}])
        insights = response['message']['content']

        # Display formatted Markdown output
        display(Markdown(f"## 📌 AI-Generated Insights\n{insights}"))

# Concrete Strategy for Categorical vs Numerical Analysis
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=feature1, y=feature2, data=df, palette="coolwarm")
        plt.title(f"📦 {feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
        self.generate_ai_insight(feature1, feature2)

    def generate_ai_insight(self, feature1, feature2):
        summary_prompt = f"""
        Analyze how {feature1} (categorical) influences {feature2} (numerical).
        - Identify trends, outliers, and key patterns.
        - Discuss variability and distribution.
        - Exclude code from the response.
        - Format the response in Markdown.
        """

        response = ollama.chat(model='mistral', messages=[{"role": "user", "content": summary_prompt}])
        insights = response['message']['content']

        # Display formatted Markdown output
        display(Markdown(f"## 📌 AI-Generated Insights\n{insights}"))

# Context Class
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        self._strategy.analyze(df, feature1, feature2)
