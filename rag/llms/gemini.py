import instructor
import google.generativeai as genai
from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv
from pydantic import BaseModel
from limiter import Limiter
from datetime import datetime
from tenacity import Retrying, wait_exponential, AsyncRetrying

load_dotenv()

# ref https://ai.google.dev/gemini-api/docs/rate-limits#current-rate-limits
limiter = Limiter(rate=0.15, consume=1, capacity=1)


class GeminiChat(DeepEvalBaseLLM):
    """
    Implements global rate limiting (shared across all instances of this class) using the Limiter utility. 
    Additionally, it employs exponential backoff retrying via Tenacity, which is passed as a parameter to 
    the Instructor client and managed internally by the package.
    """
    def __init__(self, model_name: str):
        super().__init__(model_name)  # sets model_name and loads model

    def load_model(self):
        return genai.GenerativeModel(model_name=self.model_name)

    @limiter
    def generate(self, prompt: str, schema: BaseModel) -> str:
        client = self.load_model()
        instructor_client = instructor.from_gemini(
            client=client, mode=instructor.Mode.GEMINI_JSON, use_async=False
        )
        resp = instructor_client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
            max_retries=Retrying(wait=wait_exponential(multiplier=1, min=4, max=120)),
        )
        return resp

    @limiter
    async def a_generate(self, prompt: str, schema: BaseModel) -> str:
        print(f"Current time: {datetime.now()}")
        client = self.load_model()
        instructor_client = instructor.from_gemini(
            client=client, mode=instructor.Mode.GEMINI_JSON, use_async=True
        )
        resp = await instructor_client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
            # ref https://python.useinstructor.com/concepts/retrying/
            # ref https://www.perplexity.ai/search/explain-the-wait-exponential-f-MdWz90FaTxWxvcf.iYIzng
            max_retries=AsyncRetrying(
                wait=wait_exponential(multiplier=1, min=4, max=120)
            ),
        )
        return resp

    def get_model_name(self) -> str:
        return self.model_name
