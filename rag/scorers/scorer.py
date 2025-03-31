import os
from typing import List
import weave
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas import SingleTurnSample
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from ragas.llms import LangchainLLMWrapper
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = "gemini-2.0-flash-lite"

# This is specific to Google gemini API RPM of 30rpm.
# Every request will get queued and sent in 2sec gap
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,  # 2secs
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

eval_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL, api_key=GOOGLE_API_KEY, temperature=0, rate_limiter=rate_limiter
)

evaluator_llm = LangchainLLMWrapper(eval_llm)


@weave.op()
async def context_recall(output: Document, query: str, reference: str):
    scorer = LLMContextRecall(llm=evaluator_llm)
    sample = SingleTurnSample(
        user_input=query,
        retrieved_contexts=[doc.page_content for doc in output],
        reference=reference,
    )
    score = await scorer.single_turn_ascore(sample)
    return score


# https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/?h=faithfulness#faithfulness
"""Decomposes the generated response into individual claims and verifies if each claim can be inferred from the context.
and returns a score in the range (0,1)
$Final score = VerifiedClaims / TotalClaims$
"""


@weave.op()
async def faithfulness(output: BaseMessage, query: str, context: List[str]):
    scorer = Faithfulness(llm=evaluator_llm)
    sample = SingleTurnSample(
        user_input=query, response=output.content, retrieved_contexts=context
    )
    score = await scorer.single_turn_ascore(sample)
    return score


# https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/factual_correctness/
"""Dicomposes the claims in both generated and referecne text and computes TP, FP, FN from the claims.
This is then used to compute Precision, F1 and Accuray depending upon the 'mode' param value. Defualt is F1"""


@weave.op()
async def factual_correctness(output: BaseMessage, query, reference):
    scorer = FactualCorrectness(llm=evaluator_llm)
    sample = SingleTurnSample(
        user_input=query, response=output.content, reference=reference
    )
    score = await scorer.single_turn_ascore(sample)
    return score
