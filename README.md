# RAG Chatbot Application

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) chatbot designed for answering questions based on uploaded PDF documents and sending a company profile PDF via email. The project is implemented in Streamlit and includes the functionality to extract, embed, and query PDF documents, as well as send automated emails.

---

## Table of Contents
1. [Setup Instructions](#setup-instructions)
2. [Architecture Overview](#architecture-overview)
3. [Key Design Decisions](#key-design-decisions)
4. [Known Limitations](#known-limitations)
5. [Future Improvements](#future-improvements)

---

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/rag-chatbot.git
   cd rag-chatbot
   ```

2. **Install the dependencies:**
   Make sure you have Python 3.8+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the RAG chatbot application:**
   ```bash
   streamlit run main.py
   ```
   After running, upload a PDF file to interact with the chatbot.

4. **Run the email-sending chatbot:**
   ```bash
   streamlit run test.py
   ```
   Provide an email address when prompted to receive the company profile as a PDF attachment.

5. **Notebook execution:**
   Test `rag-llama3-test-implementation.ipynb` on platforms like Kaggle. Provide a sample input PDF, use GPT T4x2 for inference, and call the `rag_chatbot` function to query the chatbot.

---

## Architecture Overview

### RAG Chatbot (main.py)
The main application uses Streamlit to provide a user interface for the RAG-based chatbot.

1. **PDF Text Extraction:**
   - The PyPDF2 library extracts text from uploaded PDFs.

2. **Dataset Creation:**
   - Text is converted into a dataset compatible with HuggingFace's `Dataset` API.

3. **Embedding Creation:**
   - Sentence embeddings are generated using the `mixedbread-ai/mxbai-embed-large-v1` model from the Sentence Transformers library.

4. **Query Processing:**
   - User queries are embedded, and the FAISS index retrieves the most relevant sections from the dataset.
   - The `meta-llama/Meta-Llama-3-8B-Instruct` model is used to generate responses, with prompts formatted based on retrieved context.

5. **Model Loading:**
   - Both the embedding and LLaMA models are loaded in CPU mode with dynamic quantization to reduce memory usage.

---

### Email-Sending Chatbot (test.py)
This module provides a chatbot interface for sending the company profile PDF via email.

1. **Email Functionality:**
   - Uses the `smtplib` library for SMTP integration and the `email` module for composing and sending emails with attachments.

2. **Streamlit UI:**
   - Interactive chatbot powered by `streamlit_chat` for a conversational interface.
   - Accepts user input to trigger email functionality.

3. **PDF Attachment:**
   - The company profile PDF is hardcoded as an attachment, ensuring consistent delivery.

---

## Key Design Decisions

1. **Streamlit for UI:**
   - Chosen for its simplicity in creating web-based user interfaces for Python applications.

2. **HuggingFace Integration:**
   - Used to access pre-trained models for both embedding generation and response generation.

3. **Quantized Models:**
   - Models are dynamically quantized to enable inference on CPU with lower memory requirements.

4. **Module Separation:**
   - The email functionality is implemented in a standalone module (`test.py`) to simplify testing and integration.

---

## Known Limitations

1. **Memory Usage:**
   - The LLaMA-3 model cannot be fully loaded on CPU for efficient inference, leading to limited functionality in environments like VS Code.

2. **Streamlit on Kaggle:**
   - Streamlit cannot run directly on Kaggle due to compatibility and tunneling issues.

3. **Performance Constraints:**
   - The chatbot's responsiveness and accuracy depend on the computational resources of the deployment environment.

---

## Future Improvements

1. **Integrated Chat Application:**
   - Merge the email-sending functionality with the RAG chatbot for a unified experience.

2. **Testing on Cloud Platforms:**
   - Use tunneling tools (e.g., `ngrok`) to test the Streamlit application on platforms like Google Colab or Kaggle.

3. **Improved Model Deployment:**
   - Explore GPU-based hosting or model optimization techniques to enable efficient deployment of large models.

4. **Enhanced User Experience:**
   - Add features like saving chat history and improving PDF indexing for larger documents.

---

## Files in the Repository

- **main.py:** Streamlit code for the RAG chatbot application, optimized for CPU environments.
- **test.py:** Code for sending the company profile PDF via email.
- **requirements.txt:** List of dependencies for the project.
- **rag-llama3-test-implementation.ipynb:** Notebook for testing RAG implementation, including PDF-based queries.

