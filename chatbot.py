# banking_chatbot.py

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import openai
import datetime

# ------------------ SETUP -------------------
# OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Load sentence-transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight & fast

# Load or create FAQ database
faq_data = {
    "Question": [
        "How do I open a savings account?",
        "What is the minimum balance required?",
        "What are your home loan interest rates?",
        "How can I apply for a personal loan?",
        "How do I block a lost debit card?",
        "What are credit card rewards?",
        "What are your branch timings?",
        "How do I update my KYC details?"
    ],
    "Answer": [
        "To open a savings account, you can visit any of our branches or apply online through our website with valid ID proof.",
        "The minimum balance required depends on the account type; typically it's ₹5000 for regular savings accounts.",
        "Our home loan interest rates start from 8.5% per annum and vary based on tenure and credit profile.",
        "You can apply for a personal loan via our online portal or at the nearest branch by submitting required documents.",
        "To block your lost debit card, call our 24x7 helpline or use mobile banking immediately.",
        "Credit card rewards include cashback, reward points, and discounts on partner merchants depending on card type.",
        "Branch timings are usually from 10 AM to 5 PM, Monday to Saturday (closed on Sundays and holidays).",
        "You can update your KYC details online through net banking or at any branch with valid documents."
    ]
}

faq_df = pd.DataFrame(faq_data)

# Precompute embeddings for FAQ questions
faq_embeddings = model.encode(faq_df['Question'].tolist(), convert_to_tensor=True)

# ------------------ UTILITY FUNCTIONS -------------------

def get_faq_answer(user_question):
    """
    Match user's question with FAQ using sentence embeddings.
    If similarity < threshold, fallback to OpenAI GPT API for response.
    """
    question_embedding = model.encode(user_question, convert_to_tensor=True)
    similarity_scores = util.cos_sim(question_embedding, faq_embeddings)
    max_score_idx = similarity_scores.argmax()
    max_score = similarity_scores[0][max_score_idx].item()

    threshold = 0.6  # similarity threshold
    if max_score > threshold:
        return faq_df.iloc[max_score_idx]['Answer']
    else:
        # Use OpenAI GPT as fallback for smart answer
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful banking assistant."},
                {"role": "user", "content": user_question}
            ],
            max_tokens=200
        )
        answer = response['choices'][0]['message']['content']
        # Log unanswered questions
        log_unanswered_question(user_question)
        return answer

def log_unanswered_question(question):
    """Log unanswered questions to a CSV for future FAQ updates."""
    try:
        df = pd.read_csv("unanswered_questions.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Question", "Timestamp"])
    df = df.append({"Question": question, "Timestamp": datetime.datetime.now()}, ignore_index=True)
    df.to_csv("unanswered_questions.csv", index=False)

# EMI calculator
def calculate_emi(principal, rate, tenure_years):
    r = rate / (12*100)
    n = tenure_years * 12
    emi = principal * r * ((1 + r)**n) / (((1 + r)**n) - 1)
    return emi

# Simple interest calculator
def calculate_interest(principal, rate, tenure_years):
    return (principal * rate * tenure_years) / 100

# ------------------ STREAMLIT UI -------------------

st.set_page_config(page_title="Banking FAQ Chatbot", layout="wide")

st.title("🏦 Banking FAQ Chatbot")
st.write("Ask me any banking-related question and I can also calculate EMI or interest for loans!")

# Multi-language support
language = st.selectbox("Choose Language / भाषा चुनें", ["English", "Hindi"])

user_question = st.text_input("Type your question here:" if language=="English" else "अपना सवाल यहां टाइप करें:")

if st.button("Ask" if language=="English" else "पूछें") and user_question.strip() != "":
    answer = get_faq_answer(user_question)
    st.markdown(f"**Answer / उत्तर:** {answer}")

st.markdown("---")

# EMI Calculator
st.subheader("💰 EMI Calculator")
principal = st.number_input("Principal Amount (₹)", min_value=1000)
rate = st.number_input("Annual Interest Rate (%)", min_value=0.1, max_value=25.0, step=0.1)
tenure = st.number_input("Tenure (Years)", min_value=1, max_value=30, step=1)

if st.button("Calculate EMI"):
    emi = calculate_emi(principal, rate, tenure)
    st.success(f"Your estimated EMI is ₹{emi:,.2f} per month.")

# Interest Calculator
st.subheader("💵 Simple Interest Calculator")
if st.button("Calculate Interest"):
    interest = calculate_interest(principal, rate, tenure)
    st.info(f"Total interest payable: ₹{interest:,.2f}")

st.markdown("---")
st.write("This chatbot logs any unanswered questions so we can improve the FAQ over time.")

