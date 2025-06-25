# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TESSDATA_PREFIX=/usr/share/tessdata

# Set working directory
WORKDIR /app

# Install system dependencies including Tesseract OCR and ML dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the main application file
COPY easylaw_streamlit.py .

# Create directory for ChromaDB persistence
RUN mkdir -p chroma_db

# Expose port for Streamlit
EXPOSE 8501

# Set the default command to run Streamlit
CMD ["streamlit", "run", "easylaw_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"] 
