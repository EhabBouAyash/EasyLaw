from PIL import Image
import pytesseract
import os
from groq import Groq
import pypdfium2 as pdfium
import io
import sys
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Client
from chromadb.config import Settings
import streamlit as st
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import base64
from ollama import chat
from datetime import datetime

# Load environment variables
os.environ['TESSDATA_PREFIX'] = '/usr/share/tessdata'

# Get API key from environment
os.environ['GROQ_API_KEY'] = "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"
# Initialize Groq client
try:
    groq_client = Groq(api_key="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")
except Exception as e:
    st.error(f"Error initializing Groq client: {str(e)}")
    sys.exit(1)

# Form-specific prompts
FORM_PROMPTS = {
    "divorce": """Analyze this divorce form, extract the names of the people involved and their roles in the divorce, and provide a detailed summary including:
1. Key terms and conditions
2. Asset division details
3. Child custody arrangements (if any)
4. Alimony/spousal support terms
5. Any special conditions or clauses.
In every point, you must mention the names of the people involved. """,

    "employment": """Analyze this employment contract, extract the names of the people involved and their roles in the employment, and provide a detailed summary including:
1. Position and compensation details
2. Working hours and conditions
3. Benefits and perks
4. Termination clauses
5. Non-compete and confidentiality terms
6. Intellectual property rights
Please highlight any unusual or restrictive clauses.
In every point, you must mention the names of the people involved.""",

    "msa": """Analyze this Master Services Agreement and provide a detailed summary including:
1. Scope of services
2. Payment terms and conditions
3. Service level agreements
4. Intellectual property rights
5. Confidentiality clauses
6. Termination conditions
Please highlight any potential risks or unusual terms.
In every point, you must mention the names of the people involved.""",

    "affidavit": """Analyze this affidavit and provide a detailed summary including:
1. Main statements and declarations
2. Supporting evidence mentioned
3. Key dates and events
4. Witness information (if any)
5. Notary details
Please highlight any inconsistencies or areas that might need verification."""
}

# Document templates for generation
DOCUMENT_TEMPLATES = {
    "nda": """Generate a comprehensive Non-Disclosure Agreement with the following details:

Parties:
- Disclosing Party: {disclosing_party}
- Receiving Party: {receiving_party}

Agreement Details:
- Purpose: {purpose}
- Term: {term} years
- Effective Date: {effective_date}
- Governing Law: {governing_law}

Key Provisions:
1. Definition of Confidential Information:
{confidential_info_definition}

2. Obligations of Receiving Party:
{obligations}

3. Exclusions from Confidential Information:
{exclusions}

4. Remedies for Breach:
{remedy_clause}

5. Return of Materials:
{return_provision}

Signatures:
- {signatory1}
- {signatory2}

The agreement should be comprehensive and legally sound, including all necessary boilerplate language and standard legal provisions. Format the document professionally with clear section headings and proper legal terminology.""",

    "employment": """Generate an Employment Agreement with the following details:
- Employee Name: {employee_name}
- Position: {position}
- Company: {company}
- Start Date: {start_date}
- Base Salary: {base_salary}
- Benefits: {benefits}

The agreement should include standard employment clauses for:
1. Position and Duties
2. Compensation and Benefits
3. Term and Termination
4. Confidentiality
5. Non-competition
6. Intellectual Property Rights""",

    "msa": """Generate a Master Services Agreement with the following details:
- Service Provider: {service_provider}
- Client: {client}
- Services: {services}
- Term: {term}
- Payment Terms: {payment_terms}

The agreement should include standard MSA clauses for:
1. Scope of Services
2. Service Level Agreements
3. Payment Terms
4. Intellectual Property Rights
5. Confidentiality
6. Term and Termination"""
}

def get_form_prompt(form_type):
    return FORM_PROMPTS.get(form_type.lower(), "Analyze this document and provide a summary of key points:")

def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()

def process_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def pdf_to_img(pdf_file):
    pdf = pdfium.PdfDocument(pdf_file)
    images = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(
            scale=2.0,
            rotation=0,
        )
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
    ans = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Use markdown formatting for emphasis, especially for important terms and names. Use * for bold text."
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

def rag_qa(text, query, force_new_db=False, firm_id="default_id", form_type=None, client_name="default_client"):
    if firm_id is None:
        raise ValueError("firm_id must be provided for personal vector DBs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = splitter.split_text(text)

    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore_path = os.path.abspath("chroma_db")

    chroma_client = Client(Settings(
        persist_directory=vectorstore_path
    ))
    collection_name = f"firm_{firm_id}_{client_name}_{form_type or 'general'}"

    collection = chroma_client.get_or_create_collection(name=collection_name)

    if force_new_db or collection.count() == 0:
        docs = chunks
        metadatas = [{"source": f"chunk_{i}"} for i in range(len(docs))]
        embeddings_list = embeddings.embed_documents(docs)
        ids = [f"{firm_id}_{client_name}_chunk_{i}" for i in range(len(docs))]
        collection.add(
            documents=docs,
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids
        )

    query_embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=6
    )
    context = "\n\n".join([doc for doc in results['documents'][0]])

    base_prompt = get_form_prompt(form_type) if form_type else "Based on the following text, please answer the query. Use markdown formatting for emphasis, especially for important terms and names. Use * for bold text."

    prompt = f"""{base_prompt}

Query: {query}

Text:
{context}

Please provide a clear and concise response focusing on the key information relevant to the query. Use markdown formatting for emphasis."""

    ans = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes legal documents and provides clear, concise responses. Use markdown formatting for emphasis, especially for important terms and names. Use * for bold text."
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

def create_pdf(text, filename):
    """Create a PDF file from the generated text."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    # Create styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=12
    )
    
    # Split text into sections and create paragraphs
    story = []
    sections = text.split('\n\n')
    
    for section in sections:
        if section.strip():
            if section.startswith('Parties:') or section.startswith('Agreement Details:') or section.startswith('Key Provisions:'):
                story.append(Paragraph(section, title_style))
            elif section.startswith('1.') or section.startswith('2.') or section.startswith('3.') or section.startswith('4.') or section.startswith('5.'):
                story.append(Paragraph(section, heading_style))
            else:
                story.append(Paragraph(section, body_style))
            story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    # Create download link
    b64 = base64.b64encode(buffer.getvalue()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}.pdf">Download PDF</a>'
    return href

def generate_document(template_type, template_vars):
    """Generate a document based on template and variables using Groq."""
    template = DOCUMENT_TEMPLATES.get(template_type.lower())
    if not template:
        return "Error: Invalid template type"
    
    # Format the template with the provided variables
    prompt = template.format(**template_vars)
    
    # Add system message for document generation
    system_message = """You are a legal document generator. Generate a professional legal document based on the provided template and variables. 
    Use proper legal formatting and include all necessary sections. The document should be complete and ready for use."""
    
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating document: {str(e)}"

def initialize_chat_history():
    """Initialize chat history in session state if it doesn't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                'role': 'system',
                'content': 'You are a legal assistant powered by deepseek-r1:8b. You provide helpful responses about legal matters and include lawyer recommendations when appropriate.'
            }
        ]

def chat_with_ollama(prompt, model="deepseek-r1:8b"):
    """Send a message to Ollama and get the response."""
    try:
        response = chat(
            model=model,
            messages=[*st.session_state.messages, {'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.set_page_config(page_title="EasyLaw Document Analysis", layout="wide")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Document Analysis", "Document Generation", "Chat Bot"])
    
    with tab1:
        st.title("EasyLaw Document Analysis")
        st.markdown("Upload a PDF or TXT document and get instant analysis. Select a form type or provide a specific query to focus the analysis.")

        col1, col2 = st.columns(2)

        with col1:
            uploaded_file = st.file_uploader("Upload Document (PDF or TXT)", type=['pdf', 'txt'])
            
            form_type = st.radio(
                "Select Form Type",
                ["None", "Divorce", "Employment", "MSA", "Affidavit"],
                horizontal=True
            )
            
            query = st.text_input("Optional Query", placeholder="Enter specific questions about the document...")
            firm_id = st.text_input("Firm ID", placeholder="Enter your firm ID (optional)")
            client_name = st.text_input("Client Name", placeholder="Enter client name (required)")

        with col2:
            if uploaded_file is not None and client_name:
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=get_file_extension(uploaded_file.name)) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    try:
                        # Process the file
                        text = extract_text_from_file(temp_path)
                        
                        if not text or text.isspace():
                            st.error("Error: The uploaded file appears to be empty or contains only whitespace.")
                        else:
                            # Get the appropriate prompt based on form type
                            form_type_lower = form_type.lower() if form_type != "None" else None
                            prompt = get_form_prompt(form_type_lower) if form_type_lower else "Analyze this document and provide a summary of key points:"
                            
                            if query:
                                # Perform RAG analysis if query is provided
                                result = rag_qa(text, query, firm_id=firm_id, form_type=form_type_lower, client_name=client_name)
                            else:
                                # Use form-specific prompt for simple analysis
                                result = simple_llm_call(prompt, text)
                            
                            # Display the result with markdown support
                            st.markdown(result)
                    
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                
                except Exception as e:
                    st.error(f"Error analyzing document: {str(e)}")
            elif uploaded_file is None:
                st.info("Please upload a document to begin analysis.")
            elif not client_name:
                st.warning("Please enter a client name.")

    with tab2:
        st.title("Document Generation")
        st.markdown("Generate legal documents based on templates and provided information.")

        # Document type selection
        doc_type = st.selectbox(
            "Select Document Type",
            ["NDA", "Employment Agreement", "Master Services Agreement"]
        )

        # Dynamic form fields based on document type
        if doc_type == "NDA":
            st.subheader("Non-Disclosure Agreement Details")
            
            # Create two columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                disclosing_party = st.text_input("Disclosing Party")
                receiving_party = st.text_input("Receiving Party")
                purpose = st.text_area("Purpose of Agreement")
                term = st.number_input("Term (years)", min_value=1, max_value=10, value=2)
                effective_date = st.date_input("Effective Date")
                governing_law = st.text_input("Governing Law", value="State of California")
                
            with col2:
                confidential_info_definition = st.text_area("Definition of Confidential Information", 
                    value="Any information disclosed by the Disclosing Party that is marked as confidential or that should reasonably be understood to be confidential.")
                obligations = st.text_area("Obligations of Receiving Party",
                    value="Maintain confidentiality, use information only for the stated purpose, and prevent unauthorized disclosure.")
                exclusions = st.text_area("Exclusions from Confidential Information",
                    value="Information that is publicly available, independently developed, or lawfully received from a third party.")
                remedy_clause = st.text_area("Remedies for Breach",
                    value="Injunctive relief, monetary damages, and recovery of legal fees.")
                return_provision = st.text_area("Return of Materials",
                    value="Return or destroy all confidential materials upon request or termination.")
                
            # Signature fields
            st.subheader("Signatures")
            col3, col4 = st.columns(2)
            with col3:
                signatory1 = st.text_input("Disclosing Party Signatory")
            with col4:
                signatory2 = st.text_input("Receiving Party Signatory")

            if st.button("Generate NDA"):
                if all([disclosing_party, receiving_party, purpose, confidential_info_definition, 
                       obligations, exclusions, remedy_clause, return_provision, signatory1, signatory2]):
                    template_vars = {
                        "disclosing_party": disclosing_party,
                        "receiving_party": receiving_party,
                        "purpose": purpose,
                        "term": term,
                        "effective_date": effective_date.strftime("%B %d, %Y"),
                        "governing_law": governing_law,
                        "confidential_info_definition": confidential_info_definition,
                        "obligations": obligations,
                        "exclusions": exclusions,
                        "remedy_clause": remedy_clause,
                        "return_provision": return_provision,
                        "signatory1": signatory1,
                        "signatory2": signatory2
                    }
                    
                    # Generate the document
                    result = generate_document("nda", template_vars)
                    
                    # Display the generated text
                    st.markdown(result)
                    
                    # Create and display PDF download link
                    pdf_link = create_pdf(result, f"NDA_{disclosing_party}_{receiving_party}")
                    st.markdown(pdf_link, unsafe_allow_html=True)
                else:
                    st.warning("Please fill in all required fields.")

        elif doc_type == "Employment Agreement":
            st.subheader("Employment Agreement Details")
            col1, col2 = st.columns(2)
            with col1:
                employee_name = st.text_input("Employee Name")
                position = st.text_input("Position")
                start_date = st.date_input("Start Date")
            with col2:
                company = st.text_input("Company Name")
                base_salary = st.text_input("Base Salary")
                benefits = st.text_area("Benefits")

            if st.button("Generate Employment Agreement"):
                if employee_name and position and company and base_salary:
                    template_vars = {
                        "employee_name": employee_name,
                        "position": position,
                        "company": company,
                        "start_date": start_date.strftime("%B %d, %Y"),
                        "base_salary": base_salary,
                        "benefits": benefits
                    }
                    result = generate_document("employment", template_vars)
                    st.markdown(result)
                else:
                    st.warning("Please fill in all required fields.")

        elif doc_type == "Master Services Agreement":
            st.subheader("Master Services Agreement Details")
            col1, col2 = st.columns(2)
            with col1:
                service_provider = st.text_input("Service Provider")
                services = st.text_area("Services Description")
                term = st.text_input("Term")
            with col2:
                client = st.text_input("Client Name")
                payment_terms = st.text_area("Payment Terms")

            if st.button("Generate MSA"):
                if service_provider and client and services and payment_terms:
                    template_vars = {
                        "service_provider": service_provider,
                        "client": client,
                        "services": services,
                        "term": term,
                        "payment_terms": payment_terms
                    }
                    result = generate_document("msa", template_vars)
                    st.markdown(result)
                else:
                    st.warning("Please fill in all required fields.")

    with tab3:
        st.title("Legal Assistant Chat Bot")
        st.markdown("""
        This chat bot is powered by Ollama's gemma3:12b model. You can ask questions about legal documents,
        get explanations of legal terms, or discuss legal scenarios. The bot will also add statements based on lawyer recommendations.
        """)

        # Initialize chat history
        initialize_chat_history()

        # File upload section
        uploaded_file = st.file_uploader("Upload Document for Analysis (PDF or TXT)", type=['pdf', 'txt'])
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=get_file_extension(uploaded_file.name)) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                try:
                    # Process the file
                    text = extract_text_from_file(temp_path)
                    
                    if not text or text.isspace():
                        st.error("Error: The uploaded file appears to be empty or contains only whitespace.")
                    else:
                        # Create a prompt for the file content
                        file_prompt = f"Please analyze this document and provide a summary:\n\n{text}"
                        
                        # Get response from Ollama
                        response = chat(
                            'gemma3:12b',
                            messages=[*st.session_state.messages, {'role': 'user', 'content': file_prompt}]
                        )
                        
                        # Add the exchange to message history
                        st.session_state.messages.extend([
                            {'role': 'user', 'content': file_prompt},
                            {'role': 'assistant', 'content': response['message']['content']}
                        ])
                        
                        st.success("Document analyzed successfully!")
                
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

        # Display chat messages (excluding system message)
        for message in st.session_state.messages[1:]:  # Skip system message
            st.write(f"**{message['role'].title()}:** {message['content']}")

        # Chat input
        user_input = st.text_input("You:", key="chat_input")
        
        if st.button("Send"):
            if user_input.strip():
                # Get response from Ollama
                try:
                    response = chat(
                        'gemma3:12b',
                        messages=[*st.session_state.messages, {'role': 'user', 'content': "You are a legal assistant, and you are tasked with providing legal guidance to user questions about several legal matters in a concise and brief manner. Here is the question:\n\n"+ user_input}]
                    )

                    # Add the exchange to message history
                    st.session_state.messages.extend([
                        {'role': 'assistant', 'content': response['message']['content']}
                    ])
                    
                    # Clear the input
                    st.session_state.chat_input = ""
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")

        # Add a clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = [
                {
                    'role': 'system',
                    'content': 'You are a legal assistant powered by deepseek-r1:8b. You provide helpful responses about legal matters and include lawyer recommendations when appropriate.'
                }
            ]
            st.rerun()

if __name__ == "__main__":
    main() 