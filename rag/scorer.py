import os
import weave
from ragas.metrics import LLMContextRecall
from ragas import SingleTurnSample
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from ragas.llms import LangchainLLMWrapper

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = "gemini-2.0-flash-lite-preview-02-05"

# This is specific to Google gemini API RPM of 30rpm.
# Every request will get queued and sent in 2sec gap
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,  # 2secs
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

eval_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL, api_key=GEMINI_API_KEY, temperature=0, rate_limiter=rate_limiter
)

evaluator_llm = LangchainLLMWrapper(eval_llm)

@weave.op()
async def context_recall(output, query, reference):
    scorer = LLMContextRecall(llm=evaluator_llm)
    sample = SingleTurnSample(
        user_input=query, retrieved_contexts=output, reference=reference
    )
    score = await scorer.single_turn_ascore(sample)
    return score
