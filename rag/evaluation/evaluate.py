import argparse
import asyncio
import dataclasses
import json
import os
import warnings
from pathlib import Path

import weave
import yaml

from rag.common.dataset import Dataset
from rag.common.registry import registry
from rag.common.setup_imports import setup_imports
from rag.evaluation.config import Config
from rag.generators.base import BaseGenerator
from rag.retrievers.base import BaseRetriever


async def evaluate(config: Config):
    # Load Dataset from the provided jsonl filepath in config
    dataset_conf = config.dataset
    dataset_path = Path(
        os.path.join(os.path.dirname(__file__), "../../datasets", dataset_conf.path)
    )
    dataset = Dataset.from_jsonl(dataset_path)

    # Rename the columns if mapping is provided. This is required if col names do not adhere to standard names
    if dataset_conf.column_mapping is not None:
        col_map = {v: k for k, v in dataset_conf.column_mapping.items()}
        dataset.rename_columns(col_map)

    retriever_config = config.retriever
    ingestor_config = config.ingestor
    generator_config = config.generator

    Retriver = registry.get_retriever(retriever_config.name)
    Ingestor = registry.get_ingestor(ingestor_config.name)
    Generator = registry.get_generator(generator_config.name)
    Chunker = registry.get_chunker(retriever_config.params.chunker.name)

    # Initialize Retriver
    ret_params = dataclasses.asdict(retriever_config.params)
    chunker_params = ret_params.pop("chunker")["params"]
    retriever: BaseRetriever = Retriver(**ret_params, chunker=Chunker(**chunker_params))

    # Initialize Ingestor
    ingestor_params = dataclasses.asdict(ingestor_config.params)
    ingestor = Ingestor(**ingestor_params)

    # Initialize Generator
    generator_params = dataclasses.asdict(generator_config.params)
    generator: BaseGenerator = Generator(**generator_params)

    corpus_conf = config.corpus
    if isinstance(corpus_conf.path, list):
        corpus_path = [
            os.path.join(os.path.dirname(__file__), "../../datasets", c_path)
            for c_path in corpus_conf.path
        ]
    else:
        corpus_path = os.path.join(
            os.path.dirname(__file__), "../../corpus", corpus_conf.path
        )
    print("Ingesting corpus...")
    if os.path.isdir(corpus_path):
        docs = ingestor.load_dir(dir_path=corpus_path)
    else:
        docs = ingestor.load_file(file_path=corpus_path)
    retriever.add_docs(docs)

    # Test
    # input = "Who is the Author and when was the article published?"
    # input = 'What is one of the reasons that advancement in AI slowed in the 1970s?'
    # docs = retriever.query(input)
    # output = generator.query(input, [doc.page_content for doc in docs])
    # print("Output:", output.content)

    scorer_config = config.scorers

    if scorer_config.retriever:
        print('Evaluating Retriver...')
        ret_eval_dataset = weave.Dataset(name="eval_data", rows=dataset.data)
        ret_scorers = [registry.get_scorer(scorer) for scorer in scorer_config.retriever]
        generator_evaluation = weave.Evaluation(
            evaluation_name=f'(Ret) {config.evaluation_name}',
            dataset=ret_eval_dataset,
            scorers=ret_scorers
        )

        await generator_evaluation.evaluate(retriever)


    if scorer_config.generator:
        print('Evaluating Generator...')
        gen_scorers = [registry.get_scorer(scorer) for scorer in scorer_config.generator]

        # Add retriveal_context by dyamcially querying the retriever
        # TODO: Make this only excute for metrics that requires retrieval_context like 'faithfulness' through config 
        for row in dataset.data:
            if 'retrieval_context' not in row:
                row['retrieval_context'] = [doc.page_content for doc in retriever.query(row['input'])]

        gen_eval_dataset = weave.Dataset(name="eval_data", rows=dataset.data)

        gen_evaluation = weave.Evaluation(
            evaluation_name=f'(Gen) {config.evaluation_name}',
            dataset=gen_eval_dataset,
            scorers=gen_scorers
        )

        await gen_evaluation.evaluate(generator)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    eval_path = Path(root_dir, "rag/configs/evaluate.yaml")
    parser = argparse.ArgumentParser(
        description="Evaluate the model with the specified configuration file."
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=False,
        help="Path to the configuration file relative to root dir",
    )

    parser.add_argument(
        "-wb",
        "--enable_wandb",
        action='store_true',
        help="A boolean flag for additional functionality."
    )
    args = parser.parse_args()

    rel_file_path = args.file
    if rel_file_path is not None:
        eval_path = Path(root_dir, rel_file_path)

    setup_imports(exclude_modules=["rag.scorers.ragas_scorers"])

    with open(eval_path) as stream:
        try:
            yaml_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    config = Config(**yaml_config)

    if args.enable_wandb:
        PROJECT_NAME = "rag"  # weave project name
        weave.init(PROJECT_NAME)

    print("Evaluation Config:")
    print(json.dumps(dataclasses.asdict(config), indent=4)) # better than pprint
    asyncio.run(evaluate(config))
