from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# import everything from chatbot.py
from chatbot import (
    parse_manual,
    truncate_text_to_tokens,
    get_embedding_with_retry,
    answer_question,
    manual_text,
    EMBEDDINGS_CACHE_FILE,
    MAX_TOKENS_FOR_EMBEDDING
)
import os, pickle, numpy as np

# FastAPI app
app = FastAPI(title="VedCool Chatbot API")

# Request body schema
class QuestionRequest(BaseModel):
    question: str

# --- Load manual + embeddings (global, once) ---
parsed_manual_sections = parse_manual(manual_text)
section_data_for_chatbot = []

if os.path.exists(EMBEDDINGS_CACHE_FILE):
    try:
        with open(EMBEDDINGS_CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, list):
            section_data_for_chatbot = cached
    except Exception as e:
        logging.warning(f"Error loading embeddings cache: {e}")

if not section_data_for_chatbot:
    logging.info("Computing embeddings fresh...")
    section_data_for_chatbot = []
    for heading, content in parsed_manual_sections:
        text_to_embed = f"Section Title: {heading}\n\nContent:\n{content}"
        truncated = truncate_text_to_tokens(text_to_embed, MAX_TOKENS_FOR_EMBEDDING)
        emb = get_embedding_with_retry(truncated)
        if emb is not None:
            section_data_for_chatbot.append((heading, content, emb))
    try:
        with open(EMBEDDINGS_CACHE_FILE, "wb") as f:
            pickle.dump(section_data_for_chatbot, f)
    except Exception as e:
        logging.error(f"Error saving embeddings cache: {e}")


# --- API Endpoint ---
@app.post("/ask")
async def ask_question(data: QuestionRequest):
    q = data.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer = answer_question(
            question=q,
            section_data=section_data_for_chatbot,
            threshold=0.40,
            top_n=3,
        )
        return {
            "question": q,
            "answer": answer,
        }
    except Exception as e:
        logging.error(f"Error answering question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error processing question")
