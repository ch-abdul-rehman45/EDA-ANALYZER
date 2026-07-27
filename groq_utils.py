import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
    """Optional: lets user ask follow-up questions about the dataset."""
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