from typing import Dict, Optional

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(extra="allow"))  # allow for extra parameters
class GeneratorParams:
    model: str


@dataclass
class GeneratorConfig:
    name: str
    params: GeneratorParams


@dataclass
class IngestorParams:
    pass  # Add specific fields if needed in the future


@dataclass(config=ConfigDict(extra="allow"))  # allow for extra parameters
class ChunkerParams:
    max_characters: int
    new_after_n_chars: int
    combine_text_under_n_characters: int


@dataclass
class ChunkerConfig:
    name: str
    params: ChunkerParams


@dataclass(config=ConfigDict(extra="allow"))  # allow for extra parameters
class RetrieverParams:
    model: str
    k: int
    # chunker: dict = Field(default_factory=dict)  # Allow extra parameters
    chunker: ChunkerConfig


@dataclass
class RetrieverConfig:
    name: str
    params: RetrieverParams


@dataclass
class IngestorConfig:
    name: str
    params: IngestorParams


@dataclass
class ScorerConfig:
    retriever: list[str] = None
    generator: list[str] = None

    def __post_init__(self):
        if self.retriever is None and self.generator is None:
            raise ValueError(
                "At least one of 'retriever' or 'generator' must be a non-empty list"
            )


@dataclass
class DatasetConfig:
    path: str
    column_mapping: Optional[Dict[str, str]] = None


@dataclass
class CorpusConfig:
    path: str | list[str]


@dataclass
class Config:
    generator: GeneratorConfig
    ingestor: IngestorConfig
    retriever: RetrieverConfig
    scorers: ScorerConfig
    dataset: DatasetConfig
    corpus: CorpusConfig
    evaluation_name: Optional[str] = None
