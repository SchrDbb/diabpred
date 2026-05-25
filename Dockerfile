# DiabPred – Reproducible Docker environment
# Build:  docker build -t diabpred .
# Run:    docker run --rm -v $(pwd)/outputs:/app/outputs diabpred

FROM python:3.11-slim

LABEL maintainer="your@email.com"
LABEL description="DiabPred: AI-powered diabetes risk prediction toolkit"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .
RUN pip install --no-cache-dir -e .

# Create output directory
RUN mkdir -p outputs

# Default command: run the full experiment
CMD ["python", "scripts/run_experiment.py", "--output-dir", "outputs"]
