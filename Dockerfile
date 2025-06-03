# Use Python 3.9 as base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    TESSDATA_PREFIX=/usr/share/tessdata

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p chroma_db text_files contracts

# Set permissions for directories
RUN chmod -R 777 chroma_db text_files contracts

# Expose port for Cloud Run
EXPOSE 8080

# Command to run the application
CMD ["python3", "easylaw.py"] 