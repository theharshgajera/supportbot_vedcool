from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from chatbot import (
    parse_manual, truncate_text_to_tokens,
    get_embedding_with_retry, answer_question,
    manual_text, MAX_TOKENS_FOR_EMBEDDING
)
import os
import pickle
import numpy as np

# --- File candidates ---
CACHE_FILES = ["openai_embeddings_cache.pkl", "embeddings_cache.pkl"]

# --- Setup logging ---
logging.basicConfig(level=logging.INFO)

# --- FastAPI App Setup ---
app = FastAPI(title="VedCool Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Model ---
class QuestionRequest(BaseModel):
    question: str

# --- Parse Manual Once ---
parsed_manual_sections = parse_manual(manual_text)
section_data_for_chatbot = []

# --- Attempt to Load From Cache ---
for cache_file in CACHE_FILES:
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            if isinstance(cached, list):
                section_data_for_chatbot = cached
                logging.info(f"✅ Loaded embeddings from: {cache_file}")
                break
        except Exception as e:
            logging.warning(f"⚠️ Error loading cache '{cache_file}': {e}")

# --- If cache failed, compute embeddings ---
if not section_data_for_chatbot:
    logging.info("🧠 No valid cache found. Recomputing embeddings...")
    section_data_for_chatbot = []
    for heading, content in parsed_manual_sections:
        full_text = f"Section Title: {heading}\n\nContent:\n{content}"
        truncated = truncate_text_to_tokens(full_text, MAX_TOKENS_FOR_EMBEDDING)
        embedding = get_embedding_with_retry(truncated)
        if embedding is not None:
            section_data_for_chatbot.append((heading, content, embedding))
    # Save to primary cache
    try:
        with open("openai_embeddings_cache.pkl", 'wb') as f:
            pickle.dump(section_data_for_chatbot, f)
        logging.info("💾 Saved new embeddings to 'openai_embeddings_cache.pkl'")
    except Exception as e:
        logging.error(f"❌ Failed to save embeddings cache: {e}")

# --- API Route ---
@app.post("/ask")
async def ask_question(data: QuestionRequest):
    question = data.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = answer_question(
            question=question,
            section_data=section_data_for_chatbot,
            threshold=0.40,
            top_n=3
        )
        return {
            "answer": answer,
            "sections_used": [heading for heading, _, _ in section_data_for_chatbot]
        }
    except Exception as e:
        logging.error(f"❌ Failed to process question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing question.")

# --- Run locally ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
