from fastapi import FastAPI, HTTPException
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

   app = FastAPI(title="VedCool Chatbot API")

   class QuestionRequest(BaseModel):
       question: str

   # Load manual + embeddings
   parsed_manual_sections = parse_manual(manual_text)
   section_data_for_chatbot = []

   if os.path.exists(EMBEDDINGS_CACHE_FILE):
       try:
           with open(EMBEDDINGS_CACHE_FILE, "rb") as f:
               cached = pickle.load(f)
           if isinstance(cached, list) and all(isinstance(item, tuple) and len(item) == 3 for item in cached):
               section_data_for_chatbot = cached
               logging.info(f"Loaded {len(section_data_for_chatbot)} embeddings from cache.")
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
                   logging.warning(f"Skipping section '{heading}' due to invalid embedding.")
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
           logging.error("No embeddings generated. API may not function correctly.")

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
           return {"question": q, "answer": answer}
       except Exception as e:
           logging.error(f"Error answering question: {str(e)}")
           raise HTTPException(status_code=500, detail="Internal error processing question")
