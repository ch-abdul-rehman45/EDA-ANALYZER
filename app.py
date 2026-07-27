import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from eda_utils import (
    load_data, basic_info, missing_value_summary,
    dtype_summary, get_numeric_columns, get_categorical_columns, summarize_for_llm
)
from groq_utils import get_ai_insights, chat_with_data

st.set_page_config(page_title="EDA Analyzer", layout="wide")
st.title("📊 EDA Analyzer (powered by Groq LLM)")

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
