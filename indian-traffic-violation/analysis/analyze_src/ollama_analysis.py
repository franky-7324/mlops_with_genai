import ollama
from analyze_src.basic_data_inspection import DataInspector, SummaryStatisticsInspectionStrategy_ai

def generate_ai_insights(df, model="mistral"):
    """
    Generates AI-powered insights using Ollama from dataframe summary statistics.

    Parameters:
    df (pd.DataFrame): The dataframe to analyze.
    model (str): The Ollama model to use ("mistral" or "gemma").

    Returns:
    str: AI-generated insights in markdown format.
    """
    

    # Perform summary statistics inspection
    inspector = DataInspector(SummaryStatisticsInspectionStrategy_ai())
    summary_text = inspector.execute_inspection(df)

    # Construct prompt for AI
    prompt = f"""
🚀 **Hello, Data Analyst!** You are an expert in data analytics. Your task is to analyze the following summary statistics of a dataset and generate a structured, visually appealing markdown report with key insights. 

📊 **Dataset Overview:**  
{summary_text}  

### 🔍 **Insights & Findings**  

#### 📈 1. Summary of Numerical Features  
- 🔢 Highlight **mean, min, max, standard deviation**, and **distribution** trends.  
- 📊 Provide insights using **3-5 key numerical columns** as examples.  
- 📉 Mention any **outliers** or unusual data points.  

#### 🔤 2. Summary of Categorical Features  
- 🏷️ Identify **dominant categories** and their proportions.  
- 📌 Note **unique values** and any **interesting patterns**.  
- 🎭 Highlight any **skewness or imbalances** in categorical data.  

#### ⚠️ 3. Key Observations  
- ❗ **Missing Data:** Identify columns with missing values and their impact.  
- 📊 **Anomalies & Outliers:** Mention any extreme values and their possible causes.  
- 🔍 **Correlation Trends:** If possible, highlight relationships between key variables.  

#### 🎯 4. Recommendations  
- 🛠️ **Data Cleaning:** Suggest preprocessing steps such as **imputation, scaling, or encoding**.  
- 🧹 **Feature Engineering:** Recommend possible **transformations or new feature creation**.  
- ✅ **Actionable Insights:** Provide **next steps** based on the data findings.  

💡 **Make the report engaging with:**  
- ✅ **Icons & symbols** for clarity 🏆  
- ✅ **Bullet points & formatting** for readability 📖  
- ✅ **Graphs or tables if applicable** 📊  

📌 **Final Output Should Be:**  
- Structured, concise, and **easy to interpret**.  
- Designed for both **technical and non-technical stakeholders**.  
- **Visually enriched** for quick learning.  

Now, let's create an amazing **AI-powered data insight report! 🚀**  
"""


    # Get AI response
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])

    # Return AI-generated markdown report
    return response['message']['content']
