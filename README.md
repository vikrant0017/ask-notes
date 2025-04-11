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

## Evaluation

To evaluate the performance of different components of the RAG system (like retrievers and generators), you can use the `evaluate.py` script.

1.  **Activate the Conda environment:**
    Ensure the `rag_dev` environment is active:
    ```bash
    conda activate rag_dev
    ```

2.  **Configure the evaluation:**
    Modify the configuration file located at `rag/configs/evaluate.yaml` (or create a new one). This file defines:
    *   `evaluation_name`: A name for your evaluation run.
    *   `generator`, `ingestor`, `retriever`: The specific components and their parameters you want to test.
    *   `scorers`: The metrics to calculate for the retriever and generator (e.g., `contextual_relevancy`, `faithfulness`). Make sure the corresponding scorer modules are implemented and registered.
    *   `corpus`: The path(s) to the document(s) or directory containing the source material.
    *   `dataset`: The path to the evaluation dataset (in `.jsonl` format) containing inputs (e.g., questions) and expected outputs. It also allows mapping column names.

3.  **Run the evaluation script:**
    Execute the script from the root directory of the project:

    *   **Using the default configuration (`rag/configs/evaluate.yaml`):**
        ```bash
        python rag/evaluation/evaluate.py
        ```

    *   **Using a custom configuration file:**
        Specify the path to your configuration file relative to the project root using the `-f` or `--file` flag:
        ```bash
        python rag/evaluation/evaluate.py -f path/to/your/custom_config.yaml
        ```

    *   **Enable Weights & Biases logging (Optional):**
        If you have set up your `WANDB_API_KEY` in the `.env` file, you can enable logging to W&B using the `-wb` or `--enable_wandb` flag:
        ```bash
        python rag/evaluation/evaluate.py -wb
        # or with a custom config
        python rag/evaluation/evaluate.py -f path/to/your/config.yaml -wb
        ```

4.  **View Results:**
    The evaluation script uses Weave (Weights & Biases's tracing and evaluation tool) to run and log the evaluations. The results, including scores for the specified metrics, will be printed to the console. If W&B logging is enabled, the results will also be available in your W&B project dashboard under the specified `evaluation_name`.

## Development

This section provides guidance for developers looking to extend or modify the RAG system.

### Adding Custom Components (Generators, Retrievers, Chunkers, Scorers, Ingestors)

The system is designed to be extensible, allowing you to easily add your own custom components. The general steps are:

1.  **Implement the Component:**
    *   Create a new Python file within the corresponding component directory (e.g., `rag/scorers/my_scorers.py`, `rag/retrievers/my_retriever.py`).
    *   Define your component. This can be a **function** (common for scorers) or a **class** (common for generators, retrievers, chunkers, ingestors).
    *   If using a class, it's recommended (and often necessary) to inherit from the base class for that component type (e.g., `BaseGenerator` in `rag/generators/base.py`, `BaseRetriever` in `rag/retrievers/base.py`).
    *   Implement the required methods/logic.

2.  **Register the Component:**
    *   Import the global `registry` object from `rag.common.registry`.
    *   Use the appropriate registration decorator above your function or class definition (e.g., `@registry.register_scorer("my_metric")`, `@registry.register_retriever("my_retriever")`). Choose a unique string identifier.

3.  **Use in Configuration:**
    *   Reference your component in configuration files (like `rag/configs/evaluate.yaml`) using the unique identifier you registered.

**Example Blueprints:**

*   **Custom Scorer (Function-based):**

    ```python
    # rag/scorers/my_custom_metrics.py
    from rag.common.registry import registry
    import asyncio # If async operations are needed

    @registry.register_scorer("your_scorer_identifier")
    async def custom_scorer_function(query: str, context: list[str], generated_answer: str, expected_output: str = None) -> float:
        """Implement your scoring logic here."""
        # ... calculation logic ...
        score = 0.0 # Placeholder
        print(f"Calculating your_scorer_identifier: {score}")
        return score
    ```
    *Usage in `evaluate.yaml`:*
    ```yaml
    scorers:
      generator: # Or retriever
        - your_scorer_identifier
    ```

*   **Custom Retriever (Class-based):**

    ```python
    # rag/retrievers/my_custom_retriever.py
    from rag.common.registry import registry
    from rag.retrievers.base import BaseRetriever, Document
    from rag.chunkers.base import BaseChunkingStrategy

    @registry.register_retriever("your_retriever_identifier")
    class CustomRetriever(BaseRetriever):
        def __init__(self, k: int, chunker: BaseChunkingStrategy, **kwargs):
            super().__init__(k=k, chunker=chunker)
            # ... initialize custom state, models, etc. ...
            # Access other params via kwargs.get('param_name')
            print(f"Initialized CustomRetriever")

        def add_docs(self, docs: list[Document]):
            """Chunk documents and add them to the retriever's index/storage."""
            chunked_docs = self.chunker.chunk(docs)
            # ... logic to store/index chunked_docs ...
            print(f"Added {len(chunked_docs)} chunks to CustomRetriever.")

        def query(self, input_query: str) -> list[Document]:
            """Retrieve relevant documents based on the input query."""
            # ... implement retrieval logic ...
            retrieved_docs = [] # Placeholder
            print(f"CustomRetriever processing query: '{input_query}'")
            # Return top k documents
            return retrieved_docs[:self.k]
    ```
    *Usage in `evaluate.yaml`:*
    ```yaml
    retriever:
      name: "your_retriever_identifier"
      params:
        k: 5
        # ... other custom parameters for __init__ ...
        chunker:
          name: "registered_chunker_name"
          params: { ... chunker params ... }
    ```

*   **Custom Generator (Class-based):**

    ```python
    # rag/generators/my_custom_generator.py
    from rag.common.registry import registry
    from rag.generators.base import BaseGenerator, GenerationResult

    @registry.register_generator("your_generator_identifier")
    class CustomGenerator(BaseGenerator):
        def __init__(self, model: str, **kwargs):
            super().__init__(model=model)
            # ... initialize LLM client, load prompts, etc. ...
            # Access other params via kwargs.get('param_name')
            print(f"Initialized CustomGenerator with model: {model}")

        def query(self, input_query: str, context: list[str] = None) -> GenerationResult:
            """Generate a response based on the query and optional context."""
            # ... prepare prompt using query and context ...
            # ... call LLM ...
            response_content = "Generated response" # Placeholder
            print(f"CustomGenerator generating response for: '{input_query}'")
            return GenerationResult(content=response_content)
    ```
    *Usage in `evaluate.yaml`:*
    ```yaml
    generator:
      name: "your_generator_identifier"
      params:
        model: "your_model_name_or_path"
        # ... other custom parameters for __init__ ...
    ```

**Explore Existing Components:**

For more complex and functional examples, browse the files within the component directories:
*   `rag/chunkers/`
*   `rag/scorers/`
*   `rag/generators/`
*   `rag/retrievers/`
*   `rag/ingestors/`

The system automatically discovers and registers components placed in these directories thanks to the `setup_imports` function (`rag/common/setup_imports.py`), making them available for use in your configurations.

(TODO: Add any specific instructions for developers, such as running tests, linting, or contributing guidelines.)

