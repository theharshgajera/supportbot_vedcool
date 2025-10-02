import re
   import numpy as np
   from scipy.spatial.distance import cosine
   import google.generativeai as genai
   import logging
   import pickle
   import os
   import time
   import sys
   from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

   # --- Configuration & Setup ---
   GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_ACTUAL_GEMINI_API_KEY")  # Use environment variable
   if GEMINI_API_KEY == "YOUR_ACTUAL_GEMINI_API_KEY" or not GEMINI_API_KEY:
       logging.error("CRITICAL ERROR: Please set GEMINI_API_KEY environment variable or replace 'YOUR_ACTUAL_GEMINI_API_KEY' with a valid key.")
       print("CRITICAL ERROR: Please set GEMINI_API_KEY environment variable or replace 'YOUR_ACTUAL_GEMINI_API_KEY' with a valid key.")
       sys.exit(1)

   genai.configure(api_key=GEMINI_API_KEY)
   logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

   EMBEDDINGS_CACHE_FILE = "gemini_embeddings_cache.pkl"
   EMBEDDING_MODEL = "models/text-embedding-004"
   CHAT_MODEL = "gemini-1.5-flash"
   MAX_TOKENS_FOR_EMBEDDING = 8000

   # --- Manual Text (Replace with your full manual content) ---
   manual_text = """
   INTRODUCTION
   Overview of VedCool Platform
   VedCool is a cutting-edge educational platform designed to simplify and enhance the management of
   academic and administrative processes. By integrating advanced technology with user-friendly features,
   VedCool caters to the needs of diverse stakeholders, including administrators, teachers, students, and parents.
   The platform offers three core solutions:
   • VedCool Stream: An engaging and interactive module for streaming educational content, fostering
   immersive learning experiences.
   • VedCool Campus: A comprehensive system for managing campus-wide operations, including
   scheduling, resource allocation, and communication.
   • VedCool Learn: A dedicated learning platform that personalizes education for students, empowering
   them to excel academically.
   VedCool ensures seamless integration and adaptability, making it an essential tool for modern educational
   institutions.
   Overview of the Branch Admin Role
   The Branch Admin role is designed to streamline the management of individual branches within the
   organization. As a Branch Admin, users are granted specific permissions to oversee branch-level operations,
   ensuring efficient functioning and adherence to organizational standards. This role bridges the gap between
   branch-level staff and higher-level administration, enabling a smooth flow of information and operations.
   Key functionalities of the Branch Admin role include:
   • Managing branch-specific data, including employee records and customer details.
   • Monitoring branch performance metrics and generating reports.
   • Addressing operational issues and implementing solutions promptly.
   • Acting as the primary point of contact for branch-level queries and escalations.
   ...
   ID Card
   Click on 'ID Card'
   • Under the Student Details section, click on the ID Card option.
   Generate or Print ID Cards
   • From here, you can generate ID Cards for students.
   • Select the Standard & Section, to view the specific class / student data
   • After selecting click on the filter button to apply the filter and view the result.
   • You will have the option to select a student and click on Checkbox of the student whose ID Card you
   want to Print.
   • After selecting the checkbox, just simply click on the print button and then you can print the ID Card
   of that particular student.
   ...
   """  # Truncated for brevity; use your full manual_text

   # --- Helper Functions ---
   def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
       max_chars = max_tokens * 4
       if len(text) > max_chars:
           logging.warning(f"Truncating text from {len(text)} to {max_chars} characters.")
           return text[:max_chars]
       return text

   # --- Core Gemini API Functions with Tenacity Retries ---
   @retry(
       wait=wait_random_exponential(min=1, max=30),
       stop=stop_after_attempt(5),
       retry=retry_if_exception_type((Exception,))
   )
   def get_embedding_with_retry(text: str, model: str = EMBEDDING_MODEL):
       if not text or not text.strip():
           logging.warning("Attempted to get embedding for empty text.")
           return None
       try:
           result = genai.embed_content(
               model=model,
               content=text,
               task_type="retrieval_document"
           )
           embedding = np.array(result['embedding'])
           if embedding.size == 0:
               logging.error("Received empty embedding from API.")
               return None
           return embedding
       except Exception as e:
           logging.error(f"Failed to generate embedding: {str(e)}")
           raise

   # --- Manual Parsing Function ---
   def parse_manual(manual_text_content: str):
       lines = manual_text_content.splitlines()
       try:
           toc_start_idx = next(i for i, line in enumerate(lines) if line.strip().upper() == "TABLE OF CONTENT")
       except StopIteration:
           logging.error("Table of Contents not found in manual.")
           return []

       toc_entry_pattern = re.compile(r"^(.*?)\s*\.{3,}\s*(\d+)\s*$")
       toc_lines_texts = []
       current_idx = toc_start_idx + 1
       max_toc_scan_lines = current_idx + 700
       consecutive_non_match_limit = 5
       non_match_count = 0
       meaningful_toc_entries_count = 0

       while current_idx < len(lines) and current_idx < max_toc_scan_lines:
           line_content = lines[current_idx].strip()
           if not line_content:
               current_idx += 1
               non_match_count = 0
               continue

           match = toc_entry_pattern.match(line_content)
           if match:
               heading_text_candidate = match.group(1).strip()
               if not heading_text_candidate.isdigit() and "user manual" not in heading_text_candidate.lower():
                   toc_lines_texts.append(line_content)
                   meaningful_toc_entries_count += 1
                   non_match_count = 0
               else:
                   non_match_count = 0
           else:
               if meaningful_toc_entries_count > 5:
                   non_match_count += 1
                   if non_match_count >= consecutive_non_match_limit:
                       logging.info(f"Stopping TOC scan at line {current_idx} after {consecutive_non_match_limit} non-matching lines.")
                       break
               elif not line_content.isupper() and len(line_content) > 60:
                   non_match_count += 1
                   if non_match_count >= 2 and meaningful_toc_entries_count < 3:
                       logging.info(f"Stopping TOC scan early due to non-matching lines with few entries found.")
                       break
           current_idx += 1

       if not toc_lines_texts:
           logging.error("No valid Table of Contents entries extracted.")
           return []

       extracted_headings_info = []
       for toc_line in toc_lines_texts:
           match = toc_entry_pattern.match(toc_line)
           if match:
               heading_text_candidate = match.group(1).strip()
               if "................................................................-xl" in heading_text_candidate:
                   heading_text_candidate = heading_text_candidate.split("................................................................-xl")[0].strip()
               heading_text_candidate = re.sub(r'\s+User Manual\s*\d*$', '', heading_text_candidate, flags=re.IGNORECASE).strip()
               heading_text_candidate = re.sub(r'\s+\d+$', '', heading_text_candidate).strip()

               if heading_text_candidate and not heading_text_candidate.isdigit() and len(heading_text_candidate) > 2:
                   extracted_headings_info.append(
                       (heading_text_candidate, heading_text_candidate.upper())
                   )

       section_positions = []
       content_lines_stripped = [line.strip() for line in lines]
       content_lines_upper_stripped = [line.upper() for line in content_lines_stripped]
       content_search_start_offset = toc_start_idx + len(toc_lines_texts) + 1

       if extracted_headings_info:
           first_toc_heading_upper = extracted_headings_info[0][1]
           try:
               first_heading_actual_pos = -1
               for i in range(content_search_start_offset, len(content_lines_upper_stripped)):
                   if content_lines_upper_stripped[i] == first_toc_heading_upper and len(content_lines_stripped[i]) < 150:
                       first_heading_actual_pos = i
                       break
               if first_heading_actual_pos != -1:
                   content_search_start_offset = first_heading_actual_pos
                   logging.info(f"Adjusted content search start offset based on first TOC heading.")
               else:
                   logging.warning(f"First TOC heading not found after TOC.")
           except Exception as e:
               logging.error(f"Error finding first TOC heading position: {e}")

       current_search_line = content_search_start_offset
       found_headings_indices = set()

       for display_heading, match_heading_upper in extracted_headings_info:
           found_line_idx = -1
           try:
               for i in range(current_search_line, len(content_lines_upper_stripped)):
                   if i in found_headings_indices:
                       continue
                   if content_lines_upper_stripped[i] == match_heading_upper and len(content_lines_stripped[i]) < 150:
                       found_line_idx = i
                       found_headings_indices.add(i)
                       break
               if found_line_idx != -1:
                   section_positions.append((display_heading, found_line_idx))
               else:
                   logging.warning(f"Heading '{display_heading}' not found in manual content.")
           except Exception as e:
               logging.error(f"Error finding position for heading '{display_heading}': {e}")
           if found_line_idx != -1:
               current_search_line = found_line_idx + 1

       section_positions.sort(key=lambda x: x[1])

       parsed_sections = []
       for i in range(len(section_positions)):
           heading_display, start_line_idx_content = section_positions[i]
           content_block_start_line = start_line_idx_content + 1
           if i < len(section_positions) - 1:
               content_block_end_line = section_positions[i + 1][1]
           else:
               content_block_end_line = len(lines)

           actual_content_lines = lines[content_block_start_line:content_block_end_line]
           content_text = '\n'.join(ln.strip() for ln in actual_content_lines if ln.strip()).strip()
           content_text_lines = content_text.split('\n')
           cleaned_content_text_lines = [line for line in content_text_lines if not re.match(r"^\s*User Manual\s*\d*\s*$", line, flags=re.IGNORECASE)]
           content_text = '\n'.join(cleaned_content_text_lines).strip()

           if content_text:
               parsed_sections.append((heading_display, content_text))
           else:
               logging.info(f"Section '{heading_display}' resulted in no content after parsing.")

       logging.info(f"Successfully parsed {len(parsed_sections)} sections.")
       return parsed_sections

   # --- Q&A Function ---
   def answer_question(question: str, section_data: list, threshold=0.40, top_n=3):
       logging.info(f"Embedding question: '{question}'")
       try:
           result = genai.embed_content(
               model=EMBEDDING_MODEL,
               content=question,
               task_type="retrieval_query"
           )
           question_embedding = np.array(result['embedding'])
       except Exception as e:
           logging.error(f"Error generating question embedding: {str(e)}")
           return "I encountered an issue processing your question with the embedding model. Please try again."

       similarities = []
       for heading, content, embedding in section_data:
           if embedding is None or not isinstance(embedding, np.ndarray) or embedding.ndim == 0 or embedding.size == 0:
               logging.warning(f"Skipping section '{heading}' due to invalid or empty embedding.")
               continue
           try:
               similarity = 1 - cosine(question_embedding, embedding)
               similarities.append((similarity, heading, content))
           except Exception as e:
               logging.error(f"Error calculating cosine similarity for section '{heading}': {str(e)}")
               continue

       if not similarities:
           logging.warning("No sections with valid embeddings available.")
           return "The user manual content could not be searched at this time due to an issue with section embeddings."

       similarities.sort(key=lambda x: x[0], reverse=True)
       logging.info(f"Top {top_n} similarities for question '{question}':")
       for i, (sim_score, head, _) in enumerate(similarities[:top_n]):
           logging.info(f"  {i+1}. Similarity: {sim_score:.4f} with Section: '{head}'")

       relevant_sections_info = []
       for sim_score, heading, content in similarities[:top_n]:
           if sim_score >= threshold:
               relevant_sections_info.append({
                   "heading": heading,
                   "content": content,
                   "similarity": sim_score
               })
           else:
               break

       if not relevant_sections_info:
           highest_sim_score = similarities[0][0] if similarities else -1.0
           logging.info(f"No sections found above threshold {threshold}. Highest similarity: {highest_sim_score:.4f}.")
           return "I've searched the VedCool user manual, but I couldn't find specific information that directly addresses your question in the available excerpts."

       combined_context = ""
       log_message_context_parts = []
       for i, section_info in enumerate(relevant_sections_info):
           combined_context += (
               f"MANUAL SECTION {i+1} TITLE: \"{section_info['heading']}\" (Similarity: {section_info['similarity']:.4f})\n"
               f"SECTION {i+1} CONTENT:\n\"\"\"\n{section_info['content']}\n\"\"\"\n\n"
           )
           log_message_context_parts.append(f"'{section_info['heading']}' (Sim: {section_info['similarity']:.4f})")

       prompt_for_llm = (
           f"You are a professional and helpful AI assistant for the VedCool platform. Your goal is to provide clear, concise, and easy-to-understand answers to user questions based *exclusively* on the provided excerpts from the VedCool user manual.\n\n"
           f"Follow these instructions carefully:\n"
           f"1. Base your answer *only* on the text provided in the 'CONTEXT FROM MANUAL' section(s) below.\n"
           f"2. Answer the 'USER'S QUESTION' concisely and accurately.\n"
           f"3. If the answer is found across multiple provided sections, synthesize the information smoothly.\n"
           f"4. If the provided context directly answers the question, provide the answer directly.\n"
           f"5. If the provided context mentions the topic but does not contain the specific details to fully answer the question, state what information is available and what is missing.\n"
           f"6. If the provided context does not contain any relevant information to answer the question, clearly state that the information is not found in the provided excerpts of the manual.\n"
           f"7. Do not use any outside knowledge or make assumptions beyond the provided text.\n"
           f"8. Present answers in a clear, well-formatted way. Use bullet points for steps or lists if appropriate.\n\n"
           f"CONTEXT FROM MANUAL:\n{combined_context}\n\n"
           f"USER'S QUESTION: \"{question}\"\n\n"
           f"PROFESSIONAL AND CLEAR ANSWER:"
       )
       logging.info(f"Generating response using section(s): {', '.join(log_message_context_parts)}")
       response = generate_response_with_retry(prompt=prompt_for_llm)
       return response

   # --- Main Execution ---
   if __name__ == "__main__":
       parsed_manual_sections = parse_manual(manual_text)
       if not parsed_manual_sections:
           logging.error("No sections parsed from manual. Exiting.")
           sys.exit(1)

       section_data_for_chatbot = []
       if os.path.exists(EMBEDDINGS_CACHE_FILE):
           try:
               with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
                   loaded_data = pickle.load(f)
               if isinstance(loaded_data, list) and all(isinstance(item, tuple) and len(item) == 3 for item in loaded_data):
                   section_data_for_chatbot = loaded_data
                   logging.info(f"Loaded {len(section_data_for_chatbot)} embeddings from cache.")
               else:
                   logging.warning("Cached data invalid. Recomputing embeddings.")
           except Exception as e:
               logging.error(f"Error loading embeddings cache: {str(e)}")
               section_data_for_chatbot = []

       if not section_data_for_chatbot:
           logging.info("Computing embeddings for manual sections...")
           for i, (heading, content) in enumerate(parsed_manual_sections):
               logging.info(f"Processing section {i+1}/{len(parsed_manual_sections)}: '{heading}'")
               text_to_embed = f"Section Title: {heading}\n\nContent:\n{content}"
               truncated_text = truncate_text_to_tokens(text_to_embed, MAX_TOKENS_FOR_EMBEDDING)
               if len(truncated_text) < len(text_to_embed):
                   logging.warning(f"Truncated section '{heading}' from {len(text_to_embed)} to {len(truncated_text)} chars.")
               if not truncated_text.strip():
                   logging.warning(f"Skipping empty section '{heading}' after truncation.")
                   continue
               try:
                   embedding_array = get_embedding_with_retry(text=truncated_text)
                   if embedding_array is not None and embedding_array.size > 0:
                       section_data_for_chatbot.append((heading, content, embedding_array))
                   else:
                       logging.warning(f"Failed to compute embedding for section: {heading}.")
               except Exception as e:
                   logging.error(f"Failed to embed section '{heading}': {str(e)}")
                   continue

           if section_data_for_chatbot:
               try:
                   with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
                       pickle.dump(section_data_for_chatbot, f)
                   logging.info(f"Embeddings saved to {EMBEDDINGS_CACHE_FILE}")
               except Exception as e:
                   logging.error(f"Error saving embeddings: {str(e)}")
           else:
               logging.error("No embeddings computed. Chatbot may not function correctly.")

       if section_data_for_chatbot:
           print("\nVedCool Chatbot ready! Ask your question.")
           while True:
               try:
                   question = input("Your question: ").strip()
                   if question.lower() in ['exit', 'quit']:
                       print("Goodbye!")
                       break
                   if not question:
                       print("Please enter a valid question.")
                       continue
                   answer = answer_question(question, section_data_for_chatbot, threshold=0.40, top_n=3)
                   print(f"\nResponse:\n{answer}\n")
               except KeyboardInterrupt:
                   print("\nExiting...")
                   break
               except Exception as e:
                   logging.error(f"Unexpected error: {str(e)}")
                   print("An error occurred. Please try again.")
       else:
           logging.error("No section data available. Exiting.")
           print("Error: No section data available. Check logs for errors.")
