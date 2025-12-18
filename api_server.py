import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieve import PaperRetriever
from corpus_loader import IndexedCorpusDataset
from config import Config

# Load env vars
load_dotenv()

app = FastAPI(title="Medical RAG API", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize retriever and corpus loader globally
print("Initializing retriever...")
retriever = PaperRetriever(
    model_path="best_model.pt",
    use_pretrained=False,
    cross_encoder_path="./output/cross-encoder-pubmedbert",
    corpus_file=Config.CORPUS_FILE
)

print("Initializing corpus loader...")
corpus_loader = IndexedCorpusDataset(Config.CORPUS_FILE)

# Initialize Gemini
if not os.getenv("GOOGLE_API_KEY"):
    print("WARNING: GOOGLE_API_KEY not found in env. Chatbot will fail.")

# Initialize model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# System Prompt
SYSTEM_TEMPLATE = """You are an expert medical research assistant. Your goal is to answer the user's query based ONLY on the provided context (scientific papers).

Instructions:
1.  **Analyze the Query:** Understand the user's medical question or patient description.
2.  **Analyze the Context:** Read the provided paper abstracts/texts carefully.
3.  **Synthesize Answer:**
    *   Provide a clear, easy-to-understand summary answer.
    *   Explain the mechanism or findings if relevant.
    *   Cite the papers using their PMIDs (e.g., [PMID: 12345]) when mentioning specific findings.
4.  **Limitations:** If the provided context does not contain the answer, state clearly that you cannot find the information in the retrieved papers. Do NOT hallucinate or use outside knowledge.
5.  **Language:** Answer the query using English.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("human", "{query}")
])

chain = prompt | llm | StrOutputParser()

# --- Request/Response Models ---

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    rerank: bool = True
    top_k_candidates: int = 50
    score_threshold: float = 0.0

class PaperResult(BaseModel):
    pmid: str
    score: float
    cross_score: Optional[float] = None
    distance: Optional[float] = None

class QueryResponse(BaseModel):
    query: str
    results: List[PaperResult]
    total: int

class CorpusRequest(BaseModel):
    pmids: List[str]

class CorpusDocument(BaseModel):
    pmid: str
    text: str
    found: bool

class CorpusResponse(BaseModel):
    documents: List[CorpusDocument]
    total_requested: int
    total_found: int

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str
    references: List[str]

# --- Endpoints ---

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Medical RAG API is running",
        "endpoints": {
            "retrieve": "/api/retrieve - POST - Retrieve paper IDs by query",
            "corpus": "/api/corpus - POST - Get corpus text by PMID(s)",
            "chat": "/chat - POST - RAG Chatbot"
        }
    }


@app.post("/api/retrieve", response_model=QueryResponse)
async def retrieve_papers(request: QueryRequest):
    """
    API 1: Retrieve paper IDs based on patient description query
    """
    try:
        # Perform search
        results = retriever.search(
            patient_text=request.query,
            top_k=request.top_k,
            rerank=request.rerank,
            top_k_candidates=request.top_k_candidates,
            score_threshold=request.score_threshold
        )
        
        # Format results
        paper_results = [
            PaperResult(
                pmid=str(paper['pmid']),
                score=paper['score'],
                cross_score=paper.get('cross_score'),
                distance=paper.get('distance')
            )
            for paper in results
        ]
        
        return QueryResponse(
            query=request.query,
            results=paper_results,
            total=len(paper_results)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")


@app.post("/api/corpus", response_model=CorpusResponse)
async def get_corpus_by_ids(request: CorpusRequest):
    """
    API 2: Get corpus text by PMID or list of PMIDs
    """
    try:
        documents = []
        found_count = 0
        
        for pmid in request.pmids:
            text = corpus_loader.__getitembyid__(str(pmid))
            
            if text:
                documents.append(CorpusDocument(
                    pmid=str(pmid),
                    text=text,
                    found=True
                ))
                found_count += 1
            else:
                documents.append(CorpusDocument(
                    pmid=str(pmid),
                    text="",
                    found=False
                ))
        
        return CorpusResponse(
            documents=documents,
            total_requested=len(request.pmids),
            total_found=found_count
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Corpus retrieval error: {str(e)}")


@app.get("/api/corpus/{pmid}")
async def get_single_corpus(pmid: str):
    """
    Get a single corpus document by PMID (alternative endpoint)
    """
    try:
        text = corpus_loader.__getitembyid__(pmid)
        
        if text:
            return {
                "pmid": pmid,
                "text": text,
                "found": True
            }
        else:
            raise HTTPException(status_code=404, detail=f"PMID {pmid} not found in corpus")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    API 3: RAG Chatbot
    """
    # 1. Retrieve
    print(f"Retrieving for query: {request.query}")
    retrieved_docs = retriever.search(
        patient_text=request.query,
        top_k=request.top_k,
        rerank=True,
        top_k_candidates=50
    )

    # 2. Get Context
    context_parts = []
    references = []
    for doc in retrieved_docs:
        pmid = str(doc['pmid'])
        text = corpus_loader.__getitembyid__(pmid)
        if text:
            context_parts.append(f"--- Paper PMID: {pmid} ---\n{text}\n")
            references.append(pmid)
    
    full_context = "\n".join(context_parts)

    # 3. Generate
    print("Generating answer...")
    try:
        response = chain.invoke({
            "context": full_context,
            "query": request.query
        })
        return ChatResponse(answer=response, references=references)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the API server
    # Access at: http://localhost:8000
    # API docs at: http://localhost:8000/docs
    uvicorn.run(app, host="0.0.0.0", port=8000)
