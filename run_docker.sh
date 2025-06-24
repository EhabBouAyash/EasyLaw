#!/bin/bash

# EasyLaw Docker Runner Script

echo "🚀 Starting EasyLaw Docker Container..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose or docker compose is available
DOCKER_COMPOSE_CMD=""
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    echo "❌ Neither docker-compose nor docker compose is available."
    echo ""
    echo "To install docker-compose:"
    echo "  macOS: brew install docker-compose"
    echo "  Ubuntu/Debian: sudo apt-get install docker-compose"
    echo "  Or download from: https://docs.docker.com/compose/install/"
    echo ""
    echo "Note: Newer Docker installations include 'docker compose' (with space) instead of 'docker-compose'"
    echo "If you have Docker Desktop, try: docker compose version"
    exit 1
fi

echo "✅ Found Docker Compose: $DOCKER_COMPOSE_CMD"

# Create necessary directories if they don't exist
mkdir -p chroma_db
mkdir -p uploads

# Check if GROQ_API_KEY is set
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  GROQ_API_KEY not set. Using default key from the application."
    echo "   To set your own key, run: export GROQ_API_KEY='your_key_here'"
fi

# Build and run the container
echo "📦 Building and starting the container..."
$DOCKER_COMPOSE_CMD up --build

echo "✅ Container stopped." 