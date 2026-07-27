import pandas as pd

def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def basic_info(df):
    info = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Duplicate Rows": df.duplicated().sum(),
        "Missing Values": df.isnull().sum().sum(),
    }
    return info

def missing_value_summary(df):
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    summary = pd.DataFrame({"Missing Count": missing, "Missing %": percent})
    return summary[summary["Missing Count"] > 0]

def dtype_summary(df):
    return pd.DataFrame(df.dtypes, columns=["Data Type"])

def get_numeric_columns(df):
    return df.select_dtypes(include="number").columns.tolist()

def get_categorical_columns(df):
    return df.select_dtypes(include="object").columns.tolist()

def summarize_for_llm(df):
    """Create a compact text summary of the dataset to send to the LLM
    (avoid sending the entire raw dataset — keep prompt small)."""
    summary = f"""
Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns

Columns and types:
{df.dtypes.to_string()}

Missing values per column:
{df.isnull().sum().to_string()}

Basic statistics:
{df.describe(include='all').to_string()}

Sample rows:
{df.head(5).to_string()}
"""
    return summary