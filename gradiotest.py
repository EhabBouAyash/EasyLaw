import gradio as gr
import os

def process_file(file):
    try:
        if file is None:
            return "Please upload a file first."
            
        # Get file extension
        file_extension = os.path.splitext(file.name)[1].lower()
        
        # Read file contents based on type
        if file_extension == '.txt':
            # For text files, read the actual content
            with open(file.name, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"File Content:\n\n{content}"
        else:
            return f"Unsupported file type: {file_extension}. Only .txt files are supported."
            
    except Exception as e:
        return f"Error processing file: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="File Content Extractor") as demo:
    gr.Markdown("""
    # File Content Extractor
    Upload a text file to see its contents.
    """)
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload Text File")
            process_btn = gr.Button("Extract Content")
        
        with gr.Column():
            output = gr.Textbox(label="File Contents", lines=10)
    
    process_btn.click(
        fn=process_file,
        inputs=[file_input],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
