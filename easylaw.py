import pdf2image
from PIL import Image
import pytesseract
import os
from groq import Groq
os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata'
os.environ["GROQ_API_KEY"] = "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"
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Client
from chromadb.config import Settings
from fastapi import FastAPI
import uvicorn
import gradio as gr

app = FastAPI(title="EasyLaw API")

def pdf_to_img(pdf_file):
    return pdf2image.convert_from_path(pdf_file)

def ocr_core(file):
    text = pytesseract.image_to_string(file)
    return text

def print_pages(pdf_file):
    res = ""
    images = pdf_to_img(pdf_file)
    for pg, img in enumerate(images):
        res += ocr_core(img)
    return res

def simple_llm_call(prompt, text):
    """Call llama3.1 8b instant for small texts"""
    if "GROQ_API_KEY" not in os.environ:
        raise ValueError("Please set the GROQ_API_KEY environment variable")
    chat_completion = Groq()
    ans = chat_completion.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"{prompt} +'\n\n' + Text: {text}"
            },
        ],
        temperature=0.1,
        model="llama-3.1-8b-instant"
    )
    return ans.choices[0].message.content

def rag_qa(text, query, force_new_db=False, firm_id="default_id"):
    """
    Perform RAG over the OCR text using llama3.1 8b instant, with a firm-specific Chroma collection.
    """
    if firm_id is None:
        raise ValueError("firm_id must be provided for personal vector DBs.")

    # Split text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = splitter.split_text(text)

    # Use a consistent embedding model
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

    # Use absolute path for Chroma database
    vectorstore_path = os.path.abspath("chroma_db")

    # Create Chroma client and collection for this firm
    client = Client(Settings(
        persist_directory=vectorstore_path
    ))
    collection_name = f"firm_{firm_id}"

    # Create or get the collection
    collection = client.get_or_create_collection(name=collection_name)

    # Add documents to the collection (if empty or force_new_db)
    if force_new_db or collection.count() == 0:
        docs = chunks
        metadatas = [{"source": f"chunk_{i}"} for i in range(len(docs))]
        embeddings_list = embeddings.embed_documents(docs)
        ids = [f"{firm_id}_chunk_{i}" for i in range(len(docs))]
        collection.add(
            documents=docs,
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids
        )

    # Query the collection for relevant documents
    query_embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=6
    )
    context = "\n\n".join([doc for doc in results['documents'][0]])

    # Prepare the prompt
    prompt = f"""Based on the following text, please answer the query.

Query: {query}

Text:
{context}

Please provide a clear and concise response focusing on the key information relevant to the query."""

    # Call Groq LLM
    chat_completion = Groq()
    ans = chat_completion.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes legal documents and provides clear, concise responses."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        model="llama-3.1-8b-instant",
        temperature=0.1
    )

    return ans.choices[0].message.content

def gradio_analyze_contract(file, query, firm_id):
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as buffer:
            buffer.write(file.read())
        
        # Process the PDF
        text = print_pages(temp_path)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        if query:
            # Perform RAG analysis if query is provided
            result = rag_qa(text, query, firm_id=firm_id)
        else:
            # Perform basic analysis if no query
            result = simple_llm_call("Analyze this contract and provide a summary of key points:", text)
        
        return result
    
    except Exception as e:
        return f"Error analyzing contract: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="EasyLaw Contract Analysis") as demo:
    gr.Markdown("""
    # EasyLaw Contract Analysis
    Upload a contract PDF and get instant analysis. You can optionally provide a specific query to focus the analysis.
    """)
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload Contract PDF")
            query_input = gr.Textbox(label="Optional Query", placeholder="Enter specific questions about the contract...")
            firm_id_input = gr.Textbox(label="Firm ID", placeholder="Enter your firm ID (optional)")
            analyze_btn = gr.Button("Analyze Contract")
        
        with gr.Column():
            output = gr.Textbox(label="Analysis Results", lines=10)
    
    analyze_btn.click(
        fn=gradio_analyze_contract,
        inputs=[file_input, query_input, firm_id_input],
        outputs=output
    )

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # Get port from environment variable (Cloud Run sets this)
    port = int(os.getenv("PORT", 8080))
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=port)

