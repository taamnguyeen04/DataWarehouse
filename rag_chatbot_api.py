import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
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

# Initialize App
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Initialize Models (Global)
print("Initializing RAG components...")
# Reuse the same configuration as the main API
retriever = PaperRetriever(
    model_path="best_model.pt",
    use_pretrained=False,
    cross_encoder_path="./output/cross-encoder-pubmedbert",
    corpus_file=Config.CORPUS_FILE
)
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
5.  **Language:** Answer in the same language as the query (likely Vietnamese or English).

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("human", "{query}")
])

chain = prompt | llm | StrOutputParser()

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str
    references: List[str]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
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
    # Run on port 8001 to avoid conflict with main API (8000)
    uvicorn.run(app, host="0.0.0.0", port=8001)
