import os
import streamlit as st
import time
from google.genai import errors
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

load_dotenv()
st.title("PDF Question Answering System")
st.write("Ask a question based on the provided PDF documents.")

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

pdf_folder = "pdfs"

documents = []

for file_name in os.listdir(pdf_folder):
    if file_name.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, file_name)

        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text:
                documents.append({
                    "text": text,
                    "source": file_name,
                    "page": page_number
                })


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = []
metadata = []

for document in documents:
    document_chunks = text_splitter.split_text(document["text"])

    for chunk in document_chunks:
        chunks.append(chunk)
        metadata.append({
            "source": document["source"],
            "page": document["page"]
        })

print("Total chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0])


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("Embeddings created!")
print("Embedding shape:", embeddings.shape)

import faiss
import numpy as np

embedding_dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(embedding_dimension)

index.add(np.array(embeddings).astype("float32"))

print("FAISS index created!")
print("Number of vectors:", index.ntotal)

question = st.text_input("Ask a question about the documents:")

if question:

    question_embedding = model.encode([question])

    distances, indices = index.search(
        np.array(question_embedding).astype("float32"),
        k=2
    )

    context = "\n\n".join([chunks[i] for i in indices[0]])

    prompt = f"""
    Answer the question using only the information provided in the context below.
    
    Context:
    {context}

    Question:
    {question}

    If the answer is not available in the context, say:
    "I could not find the answer in the provided documents."

    Give a clear and concise answer.
    """

    
    response = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )
            break

        except errors.ServerError as e:
            if attempt == 2:
                st.error(
                    "Gemini is temporarily unavailable. "
                    "Please try again in a little while."
                )
            else:
                time.sleep(2 ** (attempt + 1))

    
    if response is not None:
        answer = (response.text or "").strip()

        st.subheader("Answer")
        st.write(answer)

        fallback = "I could not find the answer in the provided documents."

        if fallback.lower() in answer.lower():
            st.subheader("Sources")
            st.write("No relevant source found for this answer.")
        
        else:
            st.subheader("Sources")

            i = indices[0][0]

            source = (
                f"{metadata[i]['source']} — Page {metadata[i]['page']}"
            )

            st.write(f"📄 {source}")
           
    