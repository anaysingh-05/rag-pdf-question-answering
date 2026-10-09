##  Live Demo

[Click here to open the RAG App](https://rag-pdf-question-answering-bejnrbkqzemu6nsaxl5rwe.streamlit.app/)

# PDF-Based Question Answering System (RAG)

## Project Overview

This project answers questions based on information retrieved from three PDF documents using Retrieval-Augmented Generation (RAG).

## Technologies Used

* Python
* Streamlit
* PyPDF
* Sentence Transformers
* FAISS
* Google Gemini API
* python-dotenv

## How It Works

1. Extracts text from PDF documents.
2. Splits the text into smaller chunks.
3. Converts chunks into numerical embeddings.
4. Stores embeddings in a FAISS index.
5. Retrieves relevant chunks based on the user's question.
6. Sends the retrieved context to Gemini to generate an answer.
7. Displays the answer and source information in Streamlit.

## Project Structure

* `app.py` — Main application
* `pdfs/` — PDF documents
* `requirements.txt` — Required Python packages
* `.gitignore` — Excludes sensitive and unnecessary files

## Setup Instructions

1. Install Python.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project folder and add your Gemini API key:

   `GEMINI_API_KEY=your_api_key_here`

4. Run the application:

   ```bash
   streamlit run app.py
   ```

## Note

The included PDFs are sample documents for demonstration purposes.

