import os
import streamlit as st
from PyPDF2 import PdfReader
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

# Login to HuggingFace
login(token="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")

# Load models outside of app flow to avoid re-initialization
def load_sentence_transformer():
    # Load the sentence transformer model in CPU mode
    return SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cpu")

# Functions
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_dataset(text):
    data = {"text": [text]}
    dataset = Dataset.from_dict(data)
    return dataset

def embed_dataset(dataset):
    # Load models
    ST = load_sentence_transformer()
    def embed(batch):
        information = batch["text"]
        return {"embeddings": ST.encode(information)}
    return dataset.map(embed, batched=True, batch_size=16)

def format_prompt(prompt, retrieved_documents, k):
    PROMPT = f"Question:{prompt}\nContext:"
    for idx in range(k):
        PROMPT += f"{retrieved_documents['text'][idx]}\n"
    return PROMPT

def generate_response(prompt, retrieved_documents, k):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Load the model in CPU mode and quantize it
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map=None  # Ensures model is not assigned to any GPU
    )
    # Quantize the model to 8-bit
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    formatted_prompt = format_prompt(prompt, retrieved_documents, k)
    formatted_prompt = formatted_prompt[:2000]  # Trim long prompts
    messages = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": formatted_prompt}]
    input_ids = tokenizer(messages, return_tensors="pt").input_ids.to("cpu")  # Ensure input IDs are on CPU
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

# Streamlit UI
st.title("RAG-based Chatbot with PDF Upload (CPU Mode)")

# Introduction and company profile
st.markdown("""
### Welcome to Reactive Space!
Reactive Space is a UAE-based company at the forefront of the space exploration and technology industry. 
Known for its innovative solutions in satellite design, space exploration, and cutting-edge technology, 
Reactive Space empowers global connectivity and supports the next generation of space missions.
""")
st.info("Upload a PDF document to explore more!")

# File upload
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# RAG System Prompt
SYS_PROMPT = """You are an assistant for answering questions.
You are given a document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer."""

if uploaded_file is not None:
    # Extract text from uploaded PDF
    with st.spinner("Extracting text from the uploaded PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
        dataset = create_dataset(extracted_text)
        dataset = embed_dataset(dataset)
        dataset = dataset.add_faiss_index("embeddings")
    
    st.success("PDF processed and indexed. You can now ask questions.")

    # Chat Interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Enter your query:", placeholder="Type your question here...")

    if query:
        with st.spinner("Generating response..."):
            def search(query, k=2):
                # Load models
                ST = load_sentence_transformer()
                embedded_query = ST.encode(query)
                scores, retrieved_examples = dataset.get_nearest_examples(
                    "embeddings", embedded_query, k=k
                )
                return scores, retrieved_examples
            
            scores, retrieved_docs = search(query, k=2)
            response = generate_response(query, retrieved_docs, k=2)
            
            st.session_state.chat_history.append((query, response))
        
        st.text_area("Response:", value=response, height=200)

    # Display chat history
    if st.session_state.chat_history:
        for idx, (user_input, bot_response) in enumerate(st.session_state.chat_history):
            st.write(f"**User:** {user_input}")
            st.write(f"**Bot:** {bot_response}")
