import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

app = FastAPI(title="RAG Application")

# --------------- state ---------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

vector_store = None  # will hold the FAISS index after upload


# --------------- helpers ---------------

def load_document(file_path: str):
    """Load a PDF or text file and return LangChain Documents."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Only .pdf and .txt files are supported.")
    
    return loader.load()
    # TODO 


def build_vector_store(documents):
    """Split documents into chunks and build a FAISS vector store."""
    # TODO 


def get_qa_chain(store):
    """Create a RetrievalQA chain from the vector store."""
    # TODO 

# --------------- routes ---------------

@app.get("/", response_class=HTMLResponse)
async def home():
    # TODO 


class QueryRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # TODO


@app.post("/query")
async def query_document(req: QueryRequest):
    #TODO 
