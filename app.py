import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from groq import Groq
from dotenv import load_dotenv

# ============================================================
# SETUP
# ============================================================
load_dotenv()

# Works both locally (.env) and on Streamlit Cloud (secrets)
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

client = Groq(api_key=api_key) if api_key else None

st.set_page_config(page_title="EDA Analyzer", layout="wide")
st.title("📊 EDA Analyzer (powered by Groq LLM)")

if not api_key:
    st.warning("⚠️ Groq API key not found. Add it to your .env file (local) or Streamlit secrets (cloud).")


# ============================================================
# EDA HELPER FUNCTIONS
# ============================================================
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


def basic_info(df):
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Duplicate Rows": int(df.duplicated().sum()),
        "Missing Values": int(df.isnull().sum().sum()),
    }


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
    """Compact text summary sent to the LLM (keeps prompt small)."""
    return f"""
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


# ============================================================
# GROQ / LLM FUNCTIONS
# ============================================================
def get_ai_insights(data_summary):
    prompt = f"""
You are a data analyst. Analyze the dataset summary below and provide:
1. Key observations about data quality (missing values, duplicates, types)
2. Notable patterns or trends in the numeric/categorical data
3. Suggestions for further analysis or cleaning
4. Any red flags an analyst should investigate

Dataset summary:
{data_summary}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful and precise data analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=1024
    )
    return response.choices[0].message.content


def chat_with_data(data_summary, question, history=None):
    messages = [
        {"role": "system", "content": f"You are analyzing this dataset:\n{data_summary}"}
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.4,
        max_tokens=512
    )
    return response.choices[0].message.content


# ============================================================
# STREAMLIT UI
# ============================================================
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.success("File loaded successfully!")

    st.subheader("Preview")
    st.dataframe(df.head())

    # --- Basic Info ---
    st.subheader("Dataset Overview")
    st.json(basic_info(df))

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Data Types**")
        st.dataframe(dtype_summary(df))
    with col2:
        st.write("**Missing Values**")
        st.dataframe(missing_value_summary(df))

    # --- Statistics ---
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(include="all"))

    # --- Visualizations ---
    st.subheader("Visualizations")
    numeric_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)

    if numeric_cols:
        col = st.selectbox("Select numeric column for histogram", numeric_cols)
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

        if len(numeric_cols) > 1:
            st.write("**Correlation Heatmap**")
            fig, ax = plt.subplots()
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    if cat_cols:
        col = st.selectbox("Select categorical column for value counts", cat_cols)
        st.bar_chart(df[col].value_counts())

    # --- AI Insights ---
    st.subheader("🤖 AI-Generated Insights")
    if st.button("Generate Insights with Groq LLM"):
        if not client:
            st.error("Groq API key missing — cannot generate insights.")
        else:
            with st.spinner("Analyzing with llama-3.3-70b-versatile..."):
                summary = summarize_for_llm(df)
                insights = get_ai_insights(summary)
                st.session_state["data_summary"] = summary
                st.markdown(insights)

    # --- Chat with data (bonus) ---
    st.subheader("💬 Ask Questions About Your Data")
    user_question = st.text_input("Ask something about the dataset")
    if user_question and "data_summary" in st.session_state:
        with st.spinner("Thinking..."):
            answer = chat_with_data(st.session_state["data_summary"], user_question)
            st.write(answer)
    elif user_question:
        st.warning("Click 'Generate Insights' first so the AI has context about your data.")

else:
    st.info("Upload a file to begin.")
