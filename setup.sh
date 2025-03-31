#!/bin/bash

ENV_NAME="rag_dev"
MODELS=("nomic-embed-text" "llama3.2:1b") 

# Exit on any error
set -e

# Install Conda if not already installed
if !command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    mkdir -p ~/miniconda4
    wget https://repo.anaconda.com/miniconda/Miniconda4-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda4/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda4/miniconda.sh
    source ~/miniconda4/bin/activate
    conda init --all
    echo "Miniconda Installed."
else
    echo "Conda is already installed."
fi

# Install Ollama if not already installed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Ollama Installed."
else
    echo "Ollama is installed already."
fi
    
# Create conda environment
echo "Creating Conda Environment"
conda env create -f dev_environment.yml
echo "Enviroment $ENV_NAME created"

# Use the commented code below to activate conda env in the setup terminal if required
# echo "Activating Environment $ENV_NAME"
# eval "$(~/miniconda4/bin/conda shell.bash hook)"
# conda activate $ENV_NAME

echo "Downloading Ollama models..."
for MODEL in "${MODELS[@]}"; do
    echo "Pulling model: $MODEL"
    ollama pull $MODEL
done

echo "Setup Complete"
echo "Activate the '$ENV_NAME' conda environment to start developing."