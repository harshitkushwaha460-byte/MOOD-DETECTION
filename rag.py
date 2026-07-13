import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------
# Folder containing PDFs
# -----------------------------

PDF_FOLDER = "knowledge"

# -----------------------------
# Load all PDFs
# -----------------------------

documents = []

for file in os.listdir(PDF_FOLDER):

    if file.endswith(".pdf"):

        path = os.path.join(PDF_FOLDER, file)

        loader = PyPDFLoader(path)

        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages.")

# -----------------------------
# Split into chunks
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=100

)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

# -----------------------------
# Embedding Model
# -----------------------------

embedding_model = HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"

)

# -----------------------------
# Create FAISS Database
# -----------------------------

vector_db = FAISS.from_documents(

    chunks,

    embedding_model

)

# -----------------------------
# Save Database
# -----------------------------

vector_db.save_local("vector_db")

print("✅ Vector database created successfully!")