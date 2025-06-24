# EasyLaw Docker Container

This document provides instructions for running the EasyLaw Streamlit application using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (usually comes with Docker Desktop)
- At least 4GB of RAM available for the container

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Clone or navigate to the project directory:**
   ```bash
   cd EasyLaw
   ```

2. **Set your Groq API key (optional):**
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   ```
   If not set, the application will use the default key provided in the code.

3. **Build and run the container:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
   Open your browser and go to `http://localhost:8501`

### Option 2: Using Docker directly

1. **Build the Docker image:**
   ```bash
   docker build -t easylaw .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 -v $(pwd)/chroma_db:/app/chroma_db -v $(pwd)/uploads:/app/uploads easylaw
   ```

3. **Access the application:**
   Open your browser and go to `http://localhost:8501`

## Container Features

- **OCR Processing**: Tesseract OCR is pre-installed for PDF text extraction
- **Vector Database**: ChromaDB for document embeddings and RAG functionality
- **AI Integration**: Groq API for document analysis and generation
- **Document Generation**: PDF generation capabilities
- **Chat Bot**: Ollama integration for legal assistance

## Environment Variables

- `GROQ_API_KEY`: Your Groq API key (optional, has default)
- `TESSDATA_PREFIX`: Path to Tesseract data (set automatically)

## Volumes

The container uses the following volumes:
- `./chroma_db`: Persistent storage for ChromaDB vector database
- `./uploads`: Directory for uploaded files (optional)

## Stopping the Container

### Docker Compose:
```bash
docker-compose down
```

### Docker:
```bash
docker stop <container_id>
```

## Troubleshooting

### Port already in use
If port 8501 is already in use, modify the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8502:8501"  # Use port 8502 instead
```

### Memory issues
If you encounter memory issues, increase Docker's memory limit in Docker Desktop settings.

### OCR not working
Ensure the container has enough memory and that Tesseract is properly installed. The Dockerfile includes all necessary dependencies.

### API key issues
Make sure your Groq API key is valid and has sufficient credits.

## Development

To modify the application:

1. Make changes to `easylaw_streamlit.py`
2. Rebuild the container:
   ```bash
   docker-compose up --build
   ```

## Production Deployment

For production deployment, consider:

1. Using a reverse proxy (nginx)
2. Setting up SSL/TLS certificates
3. Using environment variables for sensitive data
4. Implementing proper logging
5. Setting up monitoring and health checks

## File Structure

```
EasyLaw/
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Python dependencies
├── easylaw_streamlit.py   # Main application
├── .dockerignore          # Files to exclude from build
└── README_Docker.md       # This file
```

## Support

If you encounter any issues, check the container logs:
```bash
docker-compose logs easylaw
``` 