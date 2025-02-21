import asyncio
from dotenv import load_dotenv

import weave
from langchain_core.documents import Document

from generator import ResponseGenerator
from scorer import factual_correctness, faithfulness

load_dotenv()

WANDB_PROJECT = "rag"

run = weave.init(WANDB_PROJECT)

sample_docs = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
]

sample_queries = [
    "Who introduced the theory of relativity?",
    # "Who was the first computer programmer?",
    # "What did Isaac Newton contribute to science?",
    # "Who won two Nobel Prizes for research on radioactivity?",
    # "What is the theory of evolution by natural selection?"
]

expected_contexts = [
    ["Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity."],
    ["Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."],
    ["Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics."],
    ["Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes."],
    ["Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."],
]

expected_responses = [
    "Albert Einstein proposed the theory of relativity",
    # "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
    # "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    # "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    # "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."
]


dataset = []

for query, reference, expected_contexts in zip(sample_queries, expected_responses, expected_contexts):
    """These key names are based on the expected argument names by our custom scorer and evaluator which are
    managed by the weave.Evaluation class. If differnt names are given we need to provide a map in the weave.Evaluation
    class. We aonly need the user_input and reference since the reponse of model is evaluated by the Evualtion process
    which teakes in both the dataset and passes through the Model and passes the reponse to the evaluator"""
    dataset.append(
        {
            "query": query,
            "reference": reference,
            "expected_contexts": expected_contexts,
            "context": "\n".join(expected_contexts) # This is expected by Generator.predict() method as custom paramter name
        }
    )

evaluation_dataset = weave.Dataset(name="raw_data", rows=dataset)

"""Initialize a simple retriver (vector store). Here simple is to loosely imply that the
chunks are just simple sentences with no overlap and retrieval is simple top k consine similarity
scoring"""

docs = [Document(page_content=doc) for doc in sample_docs]
# retriever = SimpleRetriever(model="nomic-embed-text", docs=docs)
generator = ResponseGenerator(model="qwen2.5:3b")

retrieval_evaluation = weave.Evaluation(
    name="Generator_Evaluation",
    dataset=evaluation_dataset,
    scorers=[faithfulness, factual_correctness],
)

retrieval_scores = asyncio.run(retrieval_evaluation.evaluate(generator))
