# Ask Notes (RAG)

A Retrieval-Augmented Generation (RAG) project designed to answer questions based on your notes or documents, leveraging local language models with Ollama.

## Prerequisites

Before you begin, ensure you have the following installed:

*   [Git](https://git-scm.com/)
*   [Conda](https://docs.conda.io/en/latest/miniconda.html) (The setup script can install Miniconda for you on Linux)
*   [Ollama](https://ollama.com/) (The setup script can install this for you on Linux)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vikrant0017/ask-notes.git
    cd ask-notes
    ```

2.  **Run the setup script:**
    This script automates the setup process, including installing Conda and Ollama if they are not found, creating the necessary Python environment, and downloading the required local models.
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```
    The script will:
    *   Check for and optionally install Conda and Ollama.
    *   Create a Conda environment named `rag_dev` using the `dev_environment.yml` file.
    *   Pull the necessary Ollama models (`nomic-embed-text`, `llama3.2:1b`).

3.  **Activate the Conda environment:**
    ```bash
    conda activate rag_dev
    ```

4.  **Create `.env` file:**
    Create a file named `.env` in the root directory of the project. This file is used to store sensitive information like API keys.

## Environment Variables (`.env`)

The project requires certain API keys for evaluation and potentially for using specific language models. Create a `.env` file in the project root directory and add the following key-value pairs:

```plaintext
# .env

# Required for logging evaluation metrics to Weights & Biases
WANDB_API_KEY=your_wandb_api_key

# Required for using Google models via LLMasJudge for evaluation
GOOGLE_API_KEY=your_google_api_key
```

**Note:** Ensure you replace `WANDB_API_KEY` and `GOOGLE_API_KEY` with your actual keys. You can obtain these keys from [Weights & Biases](https://wandb.ai/authorize) and [Google AI Studio](https://aistudio.google.com/app/apikey) respectively. Other keys are optional depending on your specific configuration and needs.

## Usage

To interact with the RAG system, you can use the provided Streamlit chat interface:

1.  Make sure the `rag_dev` Conda environment is activated:
    ```bash
    conda activate rag_dev
    ```
2.  Run the Streamlit application:
    ```bash
    streamlit run ui/app.py
    ```
    This will open the chat interface in your web browser.

## Development

(TODO: Add any specific instructions for developers, such as running tests, linting, or contributing guidelines.)

