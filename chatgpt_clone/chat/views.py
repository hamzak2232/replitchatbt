import os
import json
import requests
import faiss
import sqlite3
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
from django.conf import settings

# Directory to store uploaded files and FAISS index
UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'uploaded_files')
INDEX_PATH = os.path.join(UPLOAD_DIR, 'faiss_index.index')

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize global variables
documents = []
vectorizer = None
index = None

# Establish a connection to the SQLite database
db_path = r"C:\Users\Hamza\Documents\AIChatBot2 - Copy (4) - Copy\chatgpt_clone\db2.db"

# Function to load text files from a given directory
def load_text_files(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

# Function to build and save a FAISS index from documents
def build_and_save_faiss_index(documents, index_path):
    vectorizer = TfidfVectorizer().fit(documents)
    vectors = vectorizer.transform(documents).toarray().astype('float32')  # Convert to float32 for FAISS compatibility

    # Create a FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 distance (Euclidean)
    index.add(vectors)

    # Save the index to a file
    faiss.write_index(index, index_path)

    return index, vectorizer

# Function to load a FAISS index from a file
def load_faiss_index(index_path):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        return index
    return None

# Function to retrieve the most relevant documents based on a query using FAISS
def retrieve_documents_faiss(query, index, vectorizer, documents, top_n=3):
    if vectorizer is None or index is None:
        print("Error: Vectorizer or FAISS index not initialized.")
        return []

    query_vector = vectorizer.transform([query]).toarray().astype('float32')  # Convert query vector to float32

    # Search FAISS index
    _, top_indices = index.search(query_vector, top_n)

    # Retrieve and return the most similar documents
    return [documents[i] for i in top_indices[0]]

# Function to fetch conversation history from the database
def fetch_conversation_history():
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Fetch all messages from the database
        cursor.execute('''
        SELECT user_prompt, chatbot_response FROM chat_messages
        ORDER BY timestamp_prompt ASC
        ''')

        messages = cursor.fetchall()
        conn.close()

        # Prepare messages for API input
        conversation_history = []
        for user_prompt, chatbot_response in messages:
            conversation_history.append({"role": "user", "content": user_prompt})
            conversation_history.append({"role": "system", "content": chatbot_response})

        return conversation_history

    except Exception as e:
        print(f"Error fetching conversation history from database: {e}")
        return []

# Function to generate a response using the retrieved documents and user query
def generate_response(prompt, retrieved_texts, use_history=False):
    url = "https://api.deepinfra.com/v1/openai/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }

    # Conditionally include conversation history
    conversation_history = []
    if use_history:
        conversation_history = fetch_conversation_history()

    # Add system role message at the start
    messages = [{"role": "system", "content": "You are a helpful assistant."}] + conversation_history

    # Include context if RAG is applicable
    if retrieved_texts:
        messages.append({"role": "user", "content": f"Context: {' '.join(retrieved_texts)}\n\nUser Query: {prompt}"})
    else:
        messages.append({"role": "user", "content": prompt})

    data = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048,
        "stop": [],
        "stream": False
    }

    try:
        # Send the POST request to the API
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()['choices'][0]['message']['content']

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return "Error contacting the API."

    except KeyError:
        print("Unexpected response format from the API")
        return "Error processing the response."

# Function to save chat messages to the SQLite database
def save_to_db(user_message, chatbot_response):
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_prompt TEXT NOT NULL,
            chatbot_response TEXT NOT NULL,
            timestamp_prompt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            timestamp_response TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Insert the message into the table
        cursor.execute('''
        INSERT INTO chat_messages (user_prompt, chatbot_response, timestamp_prompt, timestamp_response)
        VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ''', (user_message, chatbot_response))

        # Commit changes and close the connection
        conn.commit()
        conn.close()

    except Exception as e:
        print(f"Error saving to database: {e}")

# Function to get chatbot response
def chatbot_response(user_query, use_history=False, use_rag=False):
    # Check if RAG is possible and requested
    if use_rag and index is not None and vectorizer is not None:
        retrieved_texts = retrieve_documents_faiss(user_query, index, vectorizer, documents)
        # Generate a response using RAG
        response = generate_response(user_query, retrieved_texts, use_history)
    else:
        # Generate a response without RAG
        response = generate_response(user_query, [], use_history)

    return response

# View to handle file uploads and process documents
@csrf_exempt
def upload_file(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        # Save the uploaded file
        with open(file_path, 'wb') as file:
            for chunk in uploaded_file.chunks():
                file.write(chunk)

        # Load documents and build a new FAISS index
        global documents, vectorizer, index
        documents = load_text_files(UPLOAD_DIR)

        # Build a new FAISS index
        print("Building new FAISS index...")
        index, vectorizer = build_and_save_faiss_index(documents, INDEX_PATH)

        if vectorizer is not None and index is not None:
            print("FAISS index and vectorizer successfully initialized.")
        else:
            print("Failed to initialize FAISS index or vectorizer.")

        return JsonResponse({'message': 'File uploaded and processed successfully.'})
    return JsonResponse({'error': 'Invalid request.'}, status=400)

# View to handle chatbot interaction
@csrf_exempt
def send_message(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_query = data.get('message', '')
        chat_mode = data.get('chat_mode', 'regular')

        # Decide whether to use history and/or RAG
        use_history = True if user_query.strip().lower() in ['what was my first prompt?', 'can you recall our conversation?'] else False
        use_rag = chat_mode == 'rag'

        # Generate a response based on the selected mode
        response = chatbot_response(user_query, use_history, use_rag)

        # Save to the database
        save_to_db(user_query, response)

        return JsonResponse({'response': response})

    return JsonResponse({'error': 'Invalid request.'}, status=400)

# Main view for rendering the chat interface
def chat_view(request):
    return render(request, 'chat/index.html')
