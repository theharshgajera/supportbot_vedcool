from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
import pickle
import numpy as np
from chatbot import (
    parse_manual,
    truncate_text_to_tokens,
    get_embedding_with_retry,
    answer_question,
    manual_text,
    EMBEDDINGS_CACHE_FILE,
    MAX_TOKENS_FOR_EMBEDDING
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI(
    title="VedCool Chatbot API",
    description="AI-powered chatbot for VedCool platform user manual",
    version="1.0.0"
)

# CORS middleware - IMPORTANT for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I create a new admission?"
            }
        }

class QuestionResponse(BaseModel):
    question: str
    answer: str
    status: str = "success"

# Global variable to store section data
section_data_for_chatbot = []

@app.on_event("startup")
async def startup_event():
    """Load embeddings on application startup"""
    global section_data_for_chatbot
    
    logging.info("Starting VedCool Chatbot API...")
    
    # Parse manual
    parsed_manual_sections = parse_manual(manual_text)
    
    if not parsed_manual_sections:
        logging.error("Failed to parse manual sections!")
        raise RuntimeError("Manual parsing failed - cannot start API")
    
    logging.info(f"Parsed {len(parsed_manual_sections)} manual sections")
    
    # Load or compute embeddings
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        try:
            with open(EMBEDDINGS_CACHE_FILE, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, list) and all(
                isinstance(item, tuple) and len(item) == 3 
                for item in cached
            ):
                section_data_for_chatbot = cached
                logging.info(f"Loaded {len(section_data_for_chatbot)} embeddings from cache")
            else:
                logging.warning("Cached data invalid. Recomputing embeddings.")
        except Exception as e:
            logging.warning(f"Error loading embeddings cache: {str(e)}")

    if not section_data_for_chatbot:
        logging.info("Computing embeddings fresh...")
        section_data_for_chatbot = []
        
        for i, (heading, content) in enumerate(parsed_manual_sections):
            logging.info(f"Processing section {i+1}/{len(parsed_manual_sections)}: '{heading}'")
            text_to_embed = f"Section Title: {heading}\n\nContent:\n{content}"
            truncated = truncate_text_to_tokens(text_to_embed, MAX_TOKENS_FOR_EMBEDDING)
            
            try:
                emb = get_embedding_with_retry(truncated)
                if emb is not None and emb.size > 0:
                    section_data_for_chatbot.append((heading, content, emb))
                else:
                    logging.warning(f"Skipping section '{heading}' due to invalid embedding")
            except Exception as e:
                logging.error(f"Failed to embed section '{heading}': {str(e)}")
                continue
        
        if section_data_for_chatbot:
            try:
                with open(EMBEDDINGS_CACHE_FILE, "wb") as f:
                    pickle.dump(section_data_for_chatbot, f)
                logging.info(f"Embeddings saved to {EMBEDDINGS_CACHE_FILE}")
            except Exception as e:
                logging.error(f"Error saving embeddings cache: {str(e)}")
        else:
            logging.error("No embeddings generated!")
            raise RuntimeError("Embedding generation failed - cannot start API")
    
    logging.info(f"API ready with {len(section_data_for_chatbot)} sections loaded")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "VedCool Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "POST /ask": "Ask a question about VedCool",
            "GET /health": "Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "sections_loaded": len(section_data_for_chatbot),
        "ready": len(section_data_for_chatbot) > 0
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(data: QuestionRequest):
    """
    Ask a question about the VedCool platform.
    
    - **question**: Your question about VedCool features or usage
    """
    q = data.question.strip()
    
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if len(q) > 500:
        raise HTTPException(
            status_code=400, 
            detail="Question is too long. Please limit to 500 characters."
        )
    
    if not section_data_for_chatbot:
        raise HTTPException(
            status_code=503,
            detail="Chatbot is not ready yet. Embeddings are still loading."
        )
    
    try:
        logging.info(f"Processing question: {q}")
        answer = answer_question(
            question=q,
            section_data=section_data_for_chatbot,
            threshold=0.40,
            top_n=3,
        )
        return QuestionResponse(question=q, answer=answer)
    except Exception as e:
        logging.error(f"Error answering question: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="An error occurred while processing your question. Please try again."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
