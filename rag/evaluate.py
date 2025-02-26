import os
import yaml
import weave

import asyncio
from dotenv import load_dotenv

from langchain_core.documents import Document

from dataset import Dataset
from utils import get_callable_from_name, execute_callable

# Weave automatically detects and authorizes the WANDB_API_KEY
load_dotenv()

PROJECT_NAME = "rag"  # weave project name


# Usage example
if __name__ == "__main__":
    weave.init(PROJECT_NAME)
    # Read YAML file
    filepath = "rag/evaluate.yaml"
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)

    # Load the dataset from the path provided in the config
    dataset = Dataset.from_yaml(config["dataset"]["path"])
    evaluation_dataset = weave.Dataset(name="raw_data", rows=dataset.data[:2])

    # TODO: Extract the common code
    if "retriever" in config:
        print("Evalating Retriver...")
        search_dirs = [
            os.path.join(os.path.dirname(__file__), "retrievers"),
            # Add more directories if needed
        ]

        retriever_config = config["retriever"]
        callable_name = retriever_config.get("name")
        if not callable_name:
            raise ValueError("Retriever must include 'name' field")

        config_params = retriever_config.get("params", {}).copy()

        retriever = execute_callable(
            callable_name,
            search_dirs=search_dirs,
            # params from yaml config
            **config_params,
            # Extra params to callable
            docs=[
                Document(row["document"]) for row in dataset.data
            ],  # extra added paramters which are not in config
        )

        scorer_names = retriever_config["scorers"]
        scorers = []
        for name in scorer_names:
            scorer = get_callable_from_name(
                name,
                search_dirs=[
                    os.path.join(os.path.dirname(__file__), "scorers"),
                ],
            )
            if scorer is None:
                raise ValueError(f"Scorer {name} not found")
            scorers.append(scorer)

        exp_name = retriever_config.get("experiment_name")
        retrieval_evaluation = weave.Evaluation(
            name=exp_name,
            dataset=evaluation_dataset,
            scorers=scorers,
        )

        retrieval_scores = asyncio.run(retrieval_evaluation.evaluate(retriever))

    if "generator" in config:
        print("Evalating Generator...")
        search_dirs = [
            os.path.join(os.path.dirname(__file__), "generators"),
            # Add more directories if needed
        ]

        generator_config = config["generator"]
        callable_name = generator_config.get("name")
        if not callable_name:
            raise ValueError("Generator must include 'name' field")

        config_params = generator_config.get("params", {}).copy()

        generator = execute_callable(
            callable_name,
            search_dirs=search_dirs,
            # params from yaml config
            **config_params,
        )

        scorer_names = generator_config["scorers"]
        scorers = []
        for name in scorer_names:
            scorer = get_callable_from_name(
                name,
                search_dirs=[
                    os.path.join(os.path.dirname(__file__), "scorers"),
                ],
            )
            if scorer is None:
                raise ValueError(f"Scorer {name} not found")
            scorers.append(scorer)

        exp_name = generator_config.get("experiment_name")
        generator_evaluation = weave.Evaluation(
            name=exp_name,
            dataset=evaluation_dataset,
            scorers=scorers,
        )

        generator_scores = asyncio.run(generator_evaluation.evaluate(generator))
