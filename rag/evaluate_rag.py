import streamlit as st
import os
import json
import random
import sys
import weave
from rag.retrievers.reranker import Reranker
from rag.scorers.deepeval_scorers import (
    faithfulness,
    contextual_precision,
    contextual_recall,
    contextual_relevancy,
    answer_relevancy,
)
import asyncio

# Temporary fix to add rag as package to sys path
# sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from rag.rag import RAG
from rag.generators.generator import ResponseGenerator
from rag.ingestors.ingestor import SimpleIngestor
from rag.retrievers.retriever import SimpleRetriever, MMR
from rag.chunkers.base import ByTitleChunking

from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = "rag"  # weave project name
weave.init(PROJECT_NAME)

def convert_to_absolute_paths(relative_paths):
    return [os.path.abspath(path) for path in relative_paths]


import os
from pathlib import Path
# TODO: Add directory logic in the ingestor itself. Also resolve the paths properly, either from
# file or from where its run as a standard
source_dir = Path(os.path.join(os.path.dirname(__file__), '../corpus/intro-to-ml-notes'))
file_paths = []
for file_path in source_dir.iterdir():
    if file_path.is_file():
        file_paths.append(os.path.normpath(file_path))
# print(file_paths)



# Initialize the RAG
# qwen2.5:3b / 1.9GB
# llama3.2:latest / 2.0GB
# phi3.5:latest / 2.2GB
# llama3.2:1b / 1.3GB
generator = ResponseGenerator(model="llama3.2:latest")
ingestor = SimpleIngestor()
# retriever = SimpleRetriever(
#     model="nomic-embed-text",
#     k = 3,
#     # The paramter values were estmated from visualing the chunks of the docuemnt provided in chunking_visualizer.ipynb notebook
#     chunker=ByTitleChunking(
#         max_characters=1500, new_after_n_chars=1000, combine_text_under_n_characters=300
#     ),
#     # similarity_threshold=0.3
# )
# retriever = MMR(
#     model="nomic-embed-text",
#     k=3,
#     fetch_k=6,
#     diversity=0.7,
#     chunker=ByTitleChunking(
#         max_characters=1500, new_after_n_chars=1000, combine_text_under_n_characters=300
#     ),
# )
retriever = Reranker(
    emedding_model="nomic-embed-text",
    cross_encoding_model='cross-encoder/ms-marco-MiniLM-L6-v2',
    fetch_k=12,
    k=3,
    chunker=ByTitleChunking(
        max_characters=1500, new_after_n_chars=1000, combine_text_under_n_characters=300
    ),
)

docs = ingestor.load_file(file_path=file_paths)
retriever.add_docs(docs)


# Read the JSONL file and convert to weave dataset
dataset_rows = []
with open("datasets/itml_mcq_10_samples.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        dataset_rows.append(data)


# Convert dataset keys to match DeepEval requirements
for row in dataset_rows:
    row["input"] = row.pop("question")
    row["expected_output"] = row.pop("answer")
    # This is for faithfulness which requirest context. This can be either manually verified, or just use a retriver to 
    # ouput context, as it checks if outputs are based on context, not if context is relevant
    row["retrieval_context"] = [doc.page_content for doc in retriever.query(row['input'])]

evaluaton_rows = dataset_rows
evaluation_dataset = weave.Dataset(name="eval_data", rows=evaluaton_rows)

# TODO: Check why is this not working, it only puts "input" and not "expected_output" to the input
# @weave.op()
# def deepeval_preprocessor(example):
#     return {
#         "input": example["question"],
#         "expected_output": example["answer"],
#     }


# for row in evaluaton_rows:
#     input = row['input']
#     print(f"Question:\n {input}")
#     output = retriever.query(input)
#     print(f"Output: (len={len(output)}) \n {output}")
#     # print(f"Output")

# print(evaluaton_rows[0:3])

# retriever_evaluation = weave.Evaluation(
#     evaluation_name="Test Simple Retriver with threshold 0.5",
#     dataset=evaluation_dataset,
#     scorers=[
#         contextual_precision,
#         contextual_recall,
#         contextual_relevancy,
#         # faithfulness,
#         # answer_relevancy,
#     ],
#     # preprocess_model_input=deepeval_preprocessor,
# )


# @weave.op()
# def evaluate_rag(input):
#     retrieved_docs = retriever.predict(input)
#     retrieval_context = [doc.page_content for doc in retrieved_docs]
#     reponse = generator.predict(input, retrieval_context)
#     return {
#         "retrieval_context": retrieval_context,
#         "actual_output": reponse.content,
#     }  # This will be the model_output parameter names for the scoreres


# # TODO Find a work around for evaluating each component of the RAG pipeline standalone (with DeepEval paramter requirement limitation)
# print("Starting evaluation")
# generator_scores = asyncio.run(
#     retriever_evaluation.evaluate(retriever)
# )  # calls the predict method of the model


generator_evaluation = weave.Evaluation(
    evaluation_name="Test Model LLama3.2:latest with GPT4o as LLMJudge",
    dataset=evaluation_dataset,
    scorers=[
        # contextual_precision,
        # contextual_recall,
        # contextual_relevancy,
        faithfulness,
        answer_relevancy,
    ],
    # preprocess_model_input=deepeval_preprocessor,
)


generator_scores = asyncio.run(
    generator_evaluation.evaluate(generator)
)  # calls the predict method of the model