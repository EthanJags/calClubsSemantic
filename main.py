# app.py

import streamlit as st
import pandas as pd
import numpy as np
import openai
import os
from sklearn.metrics.pairwise import cosine_similarity

# Set your OpenAI API key securely
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the DataFrame with embeddings
df = pd.read_pickle('clubs_with_embeddings.pkl')

# Convert embeddings to a numpy array for faster computation
embeddings = np.array(df['embedding'].tolist())

# Function to get the embedding of the query
def get_query_embedding(query):
    return get_embedding(query)

# Function to get embeddings
def get_embedding(text, model='text-embedding-ada-002'):
    text = text.replace('\n', ' ')
    response = openai.Embedding.create(input=[text], model=model)
    embedding = response['data'][0]['embedding']
    return embedding

# Function to perform the search
def search(query, df, embeddings, n=5):
    query_embedding = get_embedding(query)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    df['similarity'] = similarities
    results = df.sort_values('similarity', ascending=False).head(n)
    return results

# Streamlit App Interface
st.title('Club Semantic Search')

query = st.text_input('Enter your search query:', '')

if query:
    with st.spinner('Searching...'):
        results = search(query, df, embeddings)
    st.success('Search completed!')
    for idx, row in results.iterrows():
        st.subheader(row['Name'])
        st.write(f"**Summary:** {row['Summary']}")
        st.write(f"**Similarity Score:** {row['similarity']:.4f}")
        st.write('---')