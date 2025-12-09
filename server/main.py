from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import shutil

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG Q&A System", version="1.0.0")

# IMPORTANT: CORS Configuration for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
vectorstore = None
qa_chain = None
UPLOAD_DIR = "documents"
uploaded_files = []  # Track uploaded files

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)


# Pydantic models
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    source_documents: list = []


class StatusResponse(BaseModel):
    status: str
    vectorstore_initialized: bool
    documents_count: int
    uploaded_files: list


# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "RAG Q&A System API is running!",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "query": "/query",
            "files": "/files",
            "reset": "/reset"
        }
    }


# Health check endpoint
@app.get("/health", response_model=StatusResponse)
def health_check():
    return StatusResponse(
        status="healthy",
        vectorstore_initialized=vectorstore is not None,
        documents_count=len(uploaded_files),
        uploaded_files=uploaded_files
    )


# Get uploaded files list
@app.get("/files")
def get_files():
    return {
        "files": uploaded_files,
        "count": len(uploaded_files)
    }


# Upload document endpoint
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a text or PDF file and process it for RAG
    """
    global vectorstore, qa_chain, uploaded_files
    
    try:
        # Check file type
        if not file.filename.endswith(('.txt', '.pdf')):
            raise HTTPException(
                status_code=400, 
                detail="Only .txt and .pdf files are supported"
            )
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read file content
        if file.filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
        elif file.filename.endswith('.pdf'):
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text_content = ""
            for page in reader.pages:
                text_content += page.extract_text()
        
        # Check if content is empty
        if not text_content.strip():
            raise HTTPException(
                status_code=400,
                detail="File appears to be empty or unreadable"
            )
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text_content)
        
        # Create or update vector store
        if vectorstore is None:
            vectorstore = Chroma.from_texts(
                texts=chunks,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
        else:
            vectorstore.add_texts(texts=chunks)
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        # Track uploaded file
        if file.filename not in uploaded_files:
            uploaded_files.append(file.filename)
        
        return {
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "total_documents": len(uploaded_files),
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing document: {str(e)}"
        )


# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_document(query: QueryRequest):
    """
    Ask a question about the uploaded documents
    """
    global qa_chain
    
    if qa_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Please upload a document first."
        )
    
    try:
        # Get answer from QA chain
        result = qa_chain({"query": query.question})
        
        # Extract source documents
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                sources.append({
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                })
        
        return QueryResponse(
            answer=result["result"],
            source_documents=sources
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )


# Reset endpoint
@app.post("/reset")
def reset_system():
    """
    Reset the vector store and QA chain
    """
    global vectorstore, qa_chain, uploaded_files
    
    vectorstore = None
    qa_chain = None
    uploaded_files = []
    
    # Clean up chroma db directory
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    
    # Clean up uploaded documents
    if os.path.exists(UPLOAD_DIR):
        for file in os.listdir(UPLOAD_DIR):
            os.remove(os.path.join(UPLOAD_DIR, file))
    
    return {
        "message": "System reset successfully",
        "status": "success"
    }


# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)