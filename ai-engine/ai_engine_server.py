from flask import Flask, request, jsonify
import uuid
import asyncio
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from openai import OpenAI


import requests

import openai


# import requests

# def get_embedding(text):
#     response = requests.post(
#         'https://api.openai.com/v1/embeddings',
#         json={
#             'model': 'text-embedding-ada-002',
#             'input': text,
#         }
#     )
#     return response.json()['data'][0]['embedding']

# import faiss
# import numpy as np

# documents = ["Document 1 text", "Document 2 text", ...]  # List of documents
# embeddings = np.array([get_embedding(doc) for doc in documents]).astype('float32')

# # Create a FAISS index
# index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
# index.add(embeddings)  # Add embeddings to the index


# def retrieve_documents(query, k=5):
#     query_embedding = get_embedding(query)
#     query_vector = np.array([query_embedding]).astype('float32')
#     distances, indices = index.search(query_vector, k)  # Search for k nearest neighbors
#     return [documents[i] for i in indices[0]]  # Return the documents

# def generate_response(query):
#     retrieved_docs = retrieve_documents(query)
#     context = "\n".join(retrieved_docs)
    
#     # Prepare the prompt with the context and the query
#     prompt = f"Context:\n{context}\n\nQuery: {query}\n\nResponse:"
    
#     response = requests.post(
#         'https://api.openai.com/v1/chat/completions',
#         headers={'Authorization': f'Bearer YOUR_OPENAI_API_KEY'},
#         json={
#             'model': 'gpt-3.5-turbo',
#             'messages': [{'role': 'user', 'content': prompt}],
#             'max_tokens': 150,
#         }
#     )
    
#     return response.json()['choices'][0]['message']['content'].strip()




# Define the URL and parameters
url = "http://host:90/callback"
params = {"uid": "vnijenveinv"}

# Initialize the OpenAI client with your API key

# Flask app initialization
app = Flask(__name__)

# Configuration
SUMMARY_WORD_LIMIT = 150  # Word limit for each summary

# Function to get a summary of text using OpenAI's ChatCompletion model
def get_summary(text, word_limit):
    prompt = f"Summarize the following text in {word_limit} words:\n\n{text}\n\nSummary:"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=word_limit * 2,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# Function to scrape and summarize a page
def scrape_and_summarize_page(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        page_text = soup.get_text(separator="\n", strip=True)
        return get_summary(page_text,350)
    else:
        return f"Failed to retrieve page content. Status code: {response.status_code}"

# Main scraping function
def scrape_main_and_links(base_url,uuid):
    # Scrape and summarize the main page
    main_summary = scrape_and_summarize_page(base_url)
    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Failed to retrieve the main page. Status code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("a", href=True)

    # Limit the number of links to process
    link_summaries = []
    for link_tag in links[:2]:
        link_uxrl = urljoin(base_url, link_tag['href'])
        link_title = link_tag.get_text(strip=True) or link_url
        summary = scrape_and_summarize_page(link_url)
        link_summaries.append((link_title, summary))

    # Save the summaries to a file
    unique_id = uuid
    filename = f"my_file_{unique_id}.txt"
    file_path = os.path.join("data", filename)
    os.makedirs("data", exist_ok=True)

    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping save.")

    with open(file_path, "w", encoding="utf-8") as file:
        file.write("Main Page Summary:\n")
        file.write(main_summary + "\n\n")
        file.write("Link Summaries:\n")
        for title, summary in link_summaries:
            file.write(f"\n--- {title} ---\n")
            file.write(summary + "\n")
    return unique_id

@app.route('/scrape', methods=['POST'])
async def scrape_url():
    print("hi")
    data = request.get_json()
    print(data)
    if 'url' not in data:
        return jsonify({"error": "Please provide 'url'."}), 400
    
    url = data['url']
    uuid = data.get('uuid', '')  # Ensure 'uuid' is passed in the request data
    unique_id = scrape_main_and_links(url, uuid)

    # Check if scraping was successful
    if not unique_id:
        # Send failure callback if unique_id was not generated
        requests.post(
            url, 
            params={"uuid": uuid},
            json={"status": "FAILED", "uuid": uuid}
        )
    else:
        # Send success callback
        requests.post(
            url, 
            params={"uuid": uuid},
            json={"status": "SUCCESS", "uuid": uuid}
        )

    return jsonify({"message": "Scraping initiated"}), 200

@app.route('/summarize/<string:unique_id>', methods=['GET'])
async def summarize_data(unique_id):
    file_path = os.path.join("data", f"my_file_{unique_id}.txt")
    if not os.path.exists(file_path):
        return jsonify({"error": "No data found for the provided UUID."}), 404

    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Summarize the following text:\n\n{file_content}"}],
        max_tokens=50,
        temperature=0.7,
    )
    summary = response.choices[0].message.content.strip()
    return jsonify({"uuid": unique_id, "msg": summary})

@app.route('/key_takeaways/<string:unique_id>', methods=['GET'])
async def key_takeaways(unique_id):
    file_path = os.path.join("data", f"my_file_{unique_id}.txt")
    if not os.path.exists(file_path):
        return jsonify({"error": "No data found for the provided UUID."}), 404

    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    prompt = (
        "You are an AI assistant tasked with extracting key takeaways from a given text. "
        "The text contains important information that should be summarized into concise points. "
        "Please provide the main ideas, critical insights, and significant facts in a clear and straightforward manner. "
        "List the key takeaways in bullet points format: Send me this an an HTML list that can directly be rendered on the UI.\n\n"
        f"Here is the text:\n\n{file_content}\n\n"
        "Please list the key takeaways as bullet points."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7,
    )
    takeaways = response.choices[0].message.content.strip()
    return jsonify({"uuid": unique_id, "msg": takeaways})

    # ----


from flask import Flask, request, jsonify
import os
from openai import OpenAI

# app = Flask(__name__)

# Initialize the OpenAI client

# Define constants
SUMMARY_THRESHOLD = 150  # Max number of words for context

# Load context from a file associated with the UUID
def load_context(file_id):
    file_path = f"./data/my_file_{file_id}.txt"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return file.read()
    return ""

# Save context back to the file
def save_context(file_id, context):
    file_path = f"./data/my_file_{file_id}.txt"
    with open(file_path, 'w') as file:
        file.write(context)

# Summarize context if it exceeds a certain length
def summarize_context(context):
    prompt = f"Summarize the following text:\n\n{context}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

@app.route('/chat/<string:uuid>', methods=['POST'])
def chat(uuid):
    # Load existing context
    context = load_context(uuid)
    file_path = f"./data/my_file_{uuid}.txt"

    user_role = request.json.get('role')
    print(user_role)
    if user_role == 'SUPPORT':
        user_role = 'Support Person'
    elif user_role == 'PRODUCT_MANAGER':
        user_role = 'Product Manager'
    else:
        user_role = 'Software Developer'

    # Get user input
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "Message is required."}), 400

    # Add the user input to the context
    context += f"\nUser: {user_input}\n Please answer this query as a {user_role}. Only answer questions that can be answered using the context below, or else write that it cannot be answered.' \n"

    # Check token length and summarize if necessary
    token_length = len(context.split())
    if token_length > SUMMARY_THRESHOLD:
        context = summarize_context(context)

    # Generate a response using the current context and file path information
    response_prompt = f"File path for context: {file_path}\n\nContext:\n{context}\n\nAssistant:"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": response_prompt}],
        max_tokens=50,
        temperature=0.7,
    )
    assistant_response = response.choices[0].message.content.strip()

    # Update context with assistant's response
    context += f"Assistant: {assistant_response}\n"

    # Save the updated context back to the file
    save_context(uuid, context)

    return jsonify({"response": assistant_response})

if __name__ == "__main__":
    app.run(debug=True)

