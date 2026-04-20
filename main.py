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
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


def get_qa_chain(store):
    """Create a RetrievalQA chain from the vector store."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )

# --------------- routes ---------------

@app.get("/", response_class=HTMLResponse)
async def home():
    return Path("static/index.html").read_text()


class QueryRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store
    if not file.filename:
        raise HTTPException(400, "No file provided.")
    if Path(file.filename).suffix.lower() not in {".pdf", ".txt"}:
        raise HTTPException(400, "Only .pdf and .txt files are supported.")
    safe_name = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_name
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        documents = load_document(str(file_path))
        vector_store = build_vector_store(documents)
    except Exception as e:
        raise HTTPException(500, f"Failed to process document: {e}")
    return {"message": f"'{safe_name}' uploaded and indexed successfully.", "pages": len(documents)}


@app.post("/query")
async def query_document(req: QueryRequest):
    if vector_store is None:
        raise HTTPException(400, "No document uploaded yet. Please upload a document first.")
    chain = get_qa_chain(vector_store)
    result = chain.invoke({"query": req.question})
    sources = [
        {"content": doc.page_content[:300], "metadata": doc.metadata}
        for doc in result.get("source_documents", [])
    ]
    return {"answer": result["result"], "sources": sources}
