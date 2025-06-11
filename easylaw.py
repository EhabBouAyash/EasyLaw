from PIL import Image
import pytesseract
import os
from groq import Groq
import pypdfium2 as pdfium
import io
from dotenv import load_dotenv
load_dotenv()
os.environ['TESSDATA_PREFIX'] = '/usr/share/tessdata'
api_key = os.getenv("GROQ_API_KEY")

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Client
from chromadb.config import Settings
from fastapi import FastAPI
import uvicorn
import gradio as gr
import tempfile
# Initialize Groq client
groq_client = Groq(api_key=api_key)

app = FastAPI(title="EasyLaw API")

def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()

def process_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def pdf_to_img(pdf_file):
    # Open the PDF file
    pdf = pdfium.PdfDocument(pdf_file)
    images = []
    
    # Convert each page to an image
    for i in range(len(pdf)):
        page = pdf[i]
        # Render page to a bitmap
        bitmap = page.render(
            scale=2.0,  # Higher scale for better quality
            rotation=0,
        )
        # Convert bitmap to PIL Image
        pil_image = bitmap.to_pil()
        images.append(pil_image)
    
    return images

def ocr_core(file):
    text = pytesseract.image_to_string(file)
    return text

def process_pdf_file(pdf_file):
    res = ""
    images = pdf_to_img(pdf_file)
    for pg, img in enumerate(images):
        res += ocr_core(img)
    return res

def extract_text_from_file(file_path):
    file_extension = get_file_extension(file_path)
    
    if file_extension == '.pdf':
        return process_pdf_file(file_path)
    elif file_extension == '.txt':
        return process_txt_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Only .pdf and .txt files are supported.")

def simple_llm_call(prompt, text):
    """Call llama3.1 8b instant for small texts"""
    
    ans = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nText:\n{text}"
            },
        ],
        temperature=0.1,
        model="llama-3.1-8b-instant"
    )
    return ans.choices[0].message.content

def rag_qa(text, query, force_new_db=False, firm_id="default_id", form_type=None):
    """
    Perform RAG over the OCR text using llama3.1 8b instant, with a firm-specific Chroma collection.
    """
    if firm_id is None:
        raise ValueError("firm_id must be provided for personal vector DBs.")

    # Split text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
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
    chroma_client = Client(Settings(
        persist_directory=vectorstore_path
    ))
    collection_name = f"firm_{firm_id}"

    # Create or get the collection
    collection = chroma_client.get_or_create_collection(name=collection_name)

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

    # Get the appropriate prompt based on form type
    base_prompt = get_form_prompt(form_type) if form_type else "Based on the following text, please answer the query."

    # Prepare the prompt
    prompt = f"""{base_prompt}

Query: {query}

Text:
{context}

Please provide a clear and concise response focusing on the key information relevant to the query."""

    # Call Groq LLM
    
    ans = groq_client.chat.completions.create(
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

# Form-specific prompts
FORM_PROMPTS = {
    "divorce": """Analyze this divorce form, extract the names of the people involved and their roles in the divorce, and provide a detailed summary including:
1. Key terms and conditions
2. Asset division details
3. Child custody arrangements (if any)
4. Alimony/spousal support terms
5. Any special conditions or clauses.
In every point, you must mention the names of the people involved. """,

    "employment": """Analyze this employment contract and provide a detailed summary including:
1. Position and compensation details
2. Working hours and conditions
3. Benefits and perks
4. Termination clauses
5. Non-compete and confidentiality terms
6. Intellectual property rights
Please highlight any unusual or restrictive clauses.""",

    "msa": """Analyze this Master Services Agreement and provide a detailed summary including:
1. Scope of services
2. Payment terms and conditions
3. Service level agreements
4. Intellectual property rights
5. Confidentiality clauses
6. Termination conditions
Please highlight any potential risks or unusual terms.""",

    "affidavit": """Analyze this affidavit and provide a detailed summary including:
1. Main statements and declarations
2. Supporting evidence mentioned
3. Key dates and events
4. Witness information (if any)
5. Notary details
Please highlight any inconsistencies or areas that might need verification."""
}

def get_form_prompt(form_type):
    return FORM_PROMPTS.get(form_type.lower(), "Analyze this document and provide a summary of key points:")

def gradio_analyze_contract(file, query, firm_id, form_type=None):
    try:
        if file is None:
            return "Please upload a document first."
            
        # Get file extension
        file_extension = os.path.splitext(file.name)[1].lower()
        
        # Read file contents based on type
        if file_extension == '.pdf':
            # For PDF files, we need to save temporarily for PDFium
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                with open(file.name, 'rb') as f:
                    temp_file.write(f.read())
                temp_path = temp_file.name
            try:
                text = process_pdf_file(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        elif file_extension == '.txt':
            # For text files, read the actual content
            with open(file.name, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            return f"Unsupported file type: {file_extension}. Only .pdf and .txt files are supported."
        
        if not text or text.isspace():
            return "Error: The uploaded file appears to be empty or contains only whitespace."
            
        # Get the appropriate prompt based on form type
        prompt = get_form_prompt(form_type) if form_type else "Analyze this document and provide a summary of key points:"
            
        if query:
            # Perform RAG analysis if query is provided
            result = rag_qa(text, query, firm_id=firm_id, form_type=form_type)
        else:
            # Use form-specific prompt for simple analysis
            result = simple_llm_call(prompt, text)
        
        return result
    
    except Exception as e:
        return f"Error analyzing document: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="EasyLaw Document Analysis") as demo:
    gr.Markdown("""
    # EasyLaw Document Analysis
    Upload a PDF or TXT document and get instant analysis. Select a form type or provide a specific query to focus the analysis.
    """)
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload Document (PDF or TXT)")
            
            # Form type selection buttons
            with gr.Row():
                divorce_btn = gr.Button("Divorce Form", variant="primary")
                employment_btn = gr.Button("Employment Form", variant="primary")
                msa_btn = gr.Button("MSA Form", variant="primary")
                affidavit_btn = gr.Button("Affidavit Form", variant="primary")
            
            query_input = gr.Textbox(label="Optional Query", placeholder="Enter specific questions about the document...")
            firm_id_input = gr.Textbox(label="Firm ID", placeholder="Enter your firm ID (optional)")
            analyze_btn = gr.Button("Analyze Document")
        
        with gr.Column():
            output = gr.Textbox(label="Analysis Results", lines=10)
    
    # Function to handle form button clicks
    def handle_form_click(form_type):
        return form_type
    
    # Connect form buttons to the analysis function
    divorce_btn.click(
        fn=lambda: handle_form_click("divorce"),
        inputs=[],
        outputs=[query_input]
    )
    
    employment_btn.click(
        fn=lambda: handle_form_click("employment"),
        inputs=[],
        outputs=[query_input]
    )
    
    msa_btn.click(
        fn=lambda: handle_form_click("msa"),
        inputs=[],
        outputs=[query_input]
    )
    
    affidavit_btn.click(
        fn=lambda: handle_form_click("affidavit"),
        inputs=[],
        outputs=[query_input]
    )
    
    analyze_btn.click(
        fn=gradio_analyze_contract,
        inputs=[file_input, query_input, firm_id_input, query_input],
        outputs=output
    )

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # Get port from environment variable (Cloud Run sets this)
    port = int(os.getenv("PORT", 8080))
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=port)

