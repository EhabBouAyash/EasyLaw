# Use Python 3.11 as base image for better compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Tesseract OCR and other required packages
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for Chroma DB
RUN mkdir -p chroma_db

# Set environment variables
ENV TESSDATA_PREFIX=/usr/share/tessdata
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose the port Streamlit runs on
EXPOSE 8501

# Create a shell script to run the Streamlit application
RUN echo '#!/bin/sh\n\
if [ -z "$GROQ_API_KEY" ]; then\n\
    echo "Warning: GROQ_API_KEY environment variable is not set"\n\
fi\n\
echo "Starting Streamlit application..."\n\
streamlit run easylaw_streamlit.py --server.port=8501 --server.address=0.0.0.0' > /app/start.sh && chmod +x /app/start.sh

# Command to run the Streamlit application
CMD ["/app/start.sh"] 
