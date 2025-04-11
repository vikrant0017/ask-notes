import weave
from deepeval.metrics import (
    # Generator Metrics
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    # Retrival metrics
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

from rag.common.registry import registry
from rag.llms.gemini import GeminiChat

LLM_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
model = GeminiChat(LLM_MODEL)

# model = 'gpt-4o'

# test_case = LLMTestCase(
#     input="I'm on an F-1 visa, how long can I stay in the US after graduation?",
#     actual_output="You can stay up to 30 days after completing your degree.",
#     expected_output="You can stay up to 60 days after completing your degree.",
#     retrieval_context=[
#         """If you are in the U.S. on an F-1 visa, you are allowed to stay for 60 days after completing
#         your degree, unless you have applied for and been approved to participate in OPT."""
#     ]
# )

"""
Defining scoring functions
These are used to score each example. Each function should have a model_output and optionally, 
other inputs from your examples, and return a dictionary with the scores.

Scoring functions need to have a output keyword argument, but the other arguments 
are user defined and are taken from the dataset examples. 
It will only take the necessary keys by using a dictionary key based on the argument name.

The DeepEval metrics each have a set of paramters required which is docuemnt in the site docs
https://docs.confident-ai.com/docs/metrics-introduction
"""


"""
Generator Metrics:
"""
"""
The input and actual_output are required to create an LLMTestCase 
(and hence required by all metrics) even though they might not be used for metric calculatio
"""

@weave.op()
@registry.register_scorer('faithfulness')
async def faithfulness(
    input: str, output: str, retrieval_context: str
):  # use param name output as per new docs not model_output
    scorer = FaithfulnessMetric(model=model)
    test_case = LLMTestCase(
        input=input, actual_output=output, retrieval_context=retrieval_context
    )
    await scorer.a_measure(test_case)
    return {"score": scorer.score, "reason": scorer.reason}


@weave.op()
@registry.register_scorer('answer_relevancy')
async def answer_relevancy(input: str, output: str):
    # retrieval_context=output['retrieval_context']
    scorer = AnswerRelevancyMetric(model=model)
    test_case = LLMTestCase(input=input, actual_output=output)
    await scorer.a_measure(test_case)
    return {"score": scorer.score, "reason": scorer.reason}

"""
Retriever Metric
"""

@weave.op()
@registry.register_scorer('contextual_precision')
async def contextual_precision(input: str, output: str, expected_output: str):
    scorer = ContextualPrecisionMetric(model=model)
    test_case = LLMTestCase(
        input=input,
        actual_output='', # This filed is mandatory for LLMTestCase
        expected_output=expected_output,
        retrieval_context=output,
    )
    await scorer.a_measure(test_case)
    return {"score": scorer.score, "reason": scorer.reason}


@weave.op()
@registry.register_scorer('contextual_recall')
async def contextual_recall(input: str, output: str, expected_output):
    scorer = ContextualRecallMetric(model=model)
    test_case = LLMTestCase(
        input='',
        actual_output='',
        retrieval_context=output,
        expected_output=expected_output,
    )
    await scorer.a_measure(test_case)
    return {"score": scorer.score, "reason": scorer.reason}


@weave.op()
@registry.register_scorer('contextual_relevancy')
async def contextual_relevancy(input: str, output: str):
    scorer = ContextualRelevancyMetric(model=model)
    test_case = LLMTestCase(
        input=input, retrieval_context=output, actual_output=''
    )
    await scorer.a_measure(test_case)
    return {"score": scorer.score, "reason": scorer.reason}
