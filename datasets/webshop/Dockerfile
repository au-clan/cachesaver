# Start from slim Python 3.8
FROM python:3.8-slim

# Install system dependencies: Java, Git, Bash, Build Tools, Curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk \
    git \
    bash \
    curl \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone the WebShop repository
RUN git clone https://github.com/princeton-nlp/webshop.git webshop


WORKDIR /app/webshop


RUN chmod +x setup.sh

RUN pip install spacy==3.3.0 pydantic==1.8.2


RUN pip install \
    beautifulsoup4==4.11.1 \
    cleantext==1.1.4 \
    env==0.1.0 \
    Flask==2.1.2 \
    Werkzeug==2.1.2 \
    gdown \
    gradio \
    gym==0.24.0 \
    numpy==1.22.4 \
    pandas==1.4.2 \
    pyserini==0.17.0 \
    pytest \
    PyYAML==6.0 \
    rank_bm25==0.2.2 \
    requests==2.27.1 \
    requests_mock \
    rich==12.4.4 \
    scikit_learn==1.1.1 \
    selenium==4.2.0 \
    thefuzz==0.19.0 \
    torch==1.11.0 \
    tqdm==4.64.0 \
    train==0.0.5 \
    transformers==4.19.2 \
    faiss-cpu

# Download the spaCy large English model
RUN python -m spacy download en_core_web_lg

# Pre-build the search index
WORKDIR /app/webshop/search_engine
RUN mkdir -p resources resources_100 resources_1k resources_100k indexes
RUN python convert_product_file_format.py
RUN bash run_indexing.sh

# Back to main folder
WORKDIR /app/webshop

# Expose frontend port
EXPOSE 3000

# Default command: open bash so you can manually run ./run_dev.sh if you want
CMD ["bash"]