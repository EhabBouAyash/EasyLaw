import pdf2image
from PIL import Image
import pytesseract
import os
from groq import Groq
os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata'
os.environ["GROQ_API_KEY"] = "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"
import ollama
# Add RAG and LLM imports
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import requests
import json
from datetime import datetime
import re
import time
import csv
from urllib.parse import urljoin
from chromadb import Client
from chromadb.config import Settings
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional

app = FastAPI(title="EasyLaw API")

def pdf_to_img(pdf_file):
    return pdf2image.convert_from_path(pdf_file)


def ocr_core(file):
    text = pytesseract.image_to_string(file)
    return text


def print_pages(pdf_file):
    res=""
    images = pdf_to_img(pdf_file)
    for pg, img in enumerate(images):
        res+=ocr_core(img)
    return res

def count_tokens(text):
    return len(text.split())

# Add functions for simple LLM call and RAG QA
def simple_llm_call(prompt, text):
    """Call llama3.3 8b instant for small texts"""
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
                "content":f"{prompt} +'\n\n' + Text: {text}"
            },
        ],
        temperature=0.1,
        model="llama-3.1-8b-instant"
    )
    return ans.choices[0].message.content

def ensure_directory_permissions(directory_path):
    """Ensure directory exists with proper write permissions"""
    try:
        # Create directory if it doesn't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, mode=0o777)
            print(f"Created directory with full permissions: {directory_path}")
        else:
            # Set permissions for existing directory
            os.chmod(directory_path, 0o777)
            print(f"Updated permissions for existing directory: {directory_path}")
        
        # Set permissions for all files and subdirectories
        for root, dirs, files in os.walk(directory_path):
            os.chmod(root, 0o777)
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o777)
            for f in files:
                os.chmod(os.path.join(root, f), 0o777)
        
        return True
    except Exception as e:
        print(f"Error setting permissions: {e}")
        return False

def clear_chroma_db(vectorstore_path="chroma_db"):
    """Clear the existing Chroma database"""
    import shutil
    try:
        if os.path.exists(vectorstore_path):
            # Ensure we have write permissions before removing
            ensure_directory_permissions(vectorstore_path)
            shutil.rmtree(vectorstore_path)
            print(f"Cleared existing Chroma database at {vectorstore_path}")
    except Exception as e:
        print(f"Error clearing Chroma database: {e}")

def rag_qa(text, query, force_new_db=False, firm_id="default_id"):
    """
    Perform RAG over the OCR text using llama3.1 8b instant, with a firm-specific Chroma collection.
    """
    if firm_id is None:
        raise ValueError("firm_id must be provided for personal vector DBs.")

    print(f"\nProcessing text with RAG for firm_id: {firm_id}")
    print(f"Input text length: {len(text)} characters")

    # Split text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"Split text into {len(chunks)} chunks")

    # Use a consistent embedding model
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

    # Use absolute path for Chroma database
    vectorstore_path = os.path.abspath("chroma_db")
    print(f"Chroma database location: {vectorstore_path}")

    # Create Chroma client and collection for this firm
    client = Client(Settings(
        persist_directory=vectorstore_path
    ))
    collection_name = f"firm_{firm_id}"

    # Optionally clear the collection if force_new_db
    if force_new_db and collection_name in [c.name for c in client.list_collections()]:
        print(f"Deleting existing collection: {collection_name}")
        client.delete_collection(name=collection_name)

    # Create or get the collection
    collection = client.get_or_create_collection(name=collection_name)

    # Add documents to the collection (if empty or force_new_db)
    if force_new_db or collection.count() == 0:
        print(f"Adding {len(chunks)} documents to collection: {collection_name}")
        # Chroma expects documents and embeddings
        # We'll use the embedding model to embed the chunks
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
    print("Performing similarity search...")
    query_embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=6
    )
    # Extract the context from the results
    context = "\n\n".join([doc for doc in results['documents'][0]])

    # Prepare the prompt
    prompt = f"""Based on the following text, please answer the query.

Query: {query}

Text:
{context}

Please provide a clear and concise response focusing on the key information relevant to the query."""

    # Call Groq LLM
    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = "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"

    print("Calling Groq LLM...")
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

    response = ans.choices[0].message.content
    print("Received response from LLM")
    return response

# Replace top-level logic with interactive QA pipeline

def collect_contracts_from_edgar():
    """Collect contracts (NDA, MSA, employment agreements, etc.) from EDGAR API and store them in a 'contracts' folder."""
    # Create contracts folder if it doesn't exist
    contracts_folder = "contracts"
    if not os.path.exists(contracts_folder):
        os.makedirs(contracts_folder)

    # EDGAR API endpoint for daily index files
    edgar_archives_url = "https://www.sec.gov/Archives/edgar/daily-index/"

    # Headers required by EDGAR API
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:137.0) Gecko/20100101 Firefox/137.0",
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov"
    }
    # Get current year and quarter
    current_year = datetime.now().year
    current_quarter = (datetime.now().month - 1) // 3 + 1
    
    # Process last 2 years of data
    for year in range(current_year - 2, current_year + 1):
        for quarter in range(1, 5):
            # Skip future quarters
            if year == current_year and quarter > current_quarter:
                continue
                
            # Construct URL for the quarter's index
            quarter_url = f"{edgar_archives_url}{year}/QTR{quarter}/"
            
            try:
                # Get the quarter's index page
                response = requests.get(quarter_url, headers=headers)
                response.raise_for_status()
                
                # Find all .idx files in the page
                idx_files = re.findall(r'href="([^"]+\.idx)"', response.text)
                
                # Download each .idx file
                for idx_file in idx_files:
                    idx_url = f"{quarter_url}{idx_file}"
                    idx_response = requests.get(idx_url, headers=headers)
                    idx_response.raise_for_status()
                    
                    # Save to contracts folder
                    idx_path = os.path.join(contracts_folder, idx_file)
                    with open(idx_path, 'wb') as f:
                        f.write(idx_response.content)
                    print(f"Downloaded {idx_file}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Error processing {year} Q{quarter}: {e}")
                continue
    try:
        response = requests.get(edgar_archives_url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.text
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from EDGAR API: {e}")
        return None

def extract_document_files(idx_file_path, doc_types=None):
    """
    Extract text file extensions from .idx files for specific document types.
    
    Args:
        idx_file_path (str): Path to the .idx file
        doc_types (list): List of document types to extract (e.g., ['10-K', '8-K', 'S-1', 'EX-10'])
    
    Returns:
        dict: Dictionary mapping document types to lists of file paths
    """
    if doc_types is None:
        doc_types = ['10-K', '8-K', 'S-1', 'EX-10']
    
    results = {doc_type: [] for doc_type in doc_types}
    
    try:
        with open(idx_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip header lines
                if line.startswith('Form Type') or line.startswith('---'):
                    continue
                
                # Split the line into components
                parts = line.strip().split()
                if len(parts) < 4:  # Skip malformed lines
                    continue
                
                # Extract form type and file path
                form_type = parts[0]
                file_path = parts[-1]
                
                # Check if this is one of our target document types
                for doc_type in doc_types:
                    if form_type.startswith(doc_type):
                        results[doc_type].append(file_path)
                        break
        
        return results
    except Exception as e:
        print(f"Error processing {idx_file_path}: {e}")
        return results





def identify_form_type(text):
    """
    Identify the type of form using keyword analysis and Groq LLM.
    
    Args:
        text (str): The text content to analyze
        
    Returns:
        dict: JSON object with 'type' and 'reason' keys
    """
    # Define keyword sets for each form type
    form_keywords = {
        'NDA': ['Confidential Information', 'Disclosing Party', 'Receiving Party', 
                'Non-Disclosure', 'Use of Information'],
        'MSA': ['Scope of Work', 'Service Provider', 'Change Orders', 
                'Service Levels', 'Fees and Payment', 'Indemnification'],
        'Employment': ['Employment At Will', 'Compensation', 'Duties and Responsibilities', 
                      'Termination', 'Equity Grant', 'Non-Compete'], 
        'Registration': ['Form S-1']
    }
    
    # Count keyword occurrences for each form type
    keyword_counts = {}
    for form_type, keywords in form_keywords.items():
        count = sum(text.count(keyword) for keyword in keywords)
        keyword_counts[form_type] = count
    
    # Create a prompt for the LLM
    prompt = f"""Based on the following keyword counts and text content, identify the most likely form type.
    Keyword counts:
    {json.dumps(keyword_counts, indent=2)}
    
    Text snippet (first 1000 characters):
    {text[:1000]}

    Text snippet (last 1000 characters):
    {text[-1000:]}
    
    Return a JSON object with two fields:
    1. "type": The identified form type (NDA, MSA, or Employment)
    2. "reason": A brief explanation of why this type was chosen, mentioning the most relevant keywords found
    
    Format your response as a valid JSON object only."""
    
    # Use Groq LLM to analyze
    if len(text) > 6000:
        response = rag_qa(text, prompt, force_new_db=True)
    else:
        response = simple_llm_call(prompt, text)
    
    try:
        # Extract JSON from response
        json_str = response.strip()
        if json_str.startswith('```json'):
            json_str = json_str[7:]
        if json_str.endswith('```'):
            json_str = json_str[:-3]
        json_str = json_str.strip()
        
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "type": "Unknown",
            "reason": "Failed to parse LLM response as JSON"
        }

@app.post("/analyze-contract")
async def analyze_contract(
    file: UploadFile = File(...),
    query: Optional[str] = None,
    firm_id: Optional[str] = "default_id"
):
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
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
        
        return JSONResponse(content={"analysis": result})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    # Get port from environment variable (Cloud Run sets this)
    port = int(os.getenv("PORT", 8080))
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=port)

