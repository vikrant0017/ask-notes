from typing import List
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
import weave


class ResponseGenerator(weave.Model):
    model: str
    _template: str = (
        "Use the following pieces of context to answer the question at the end. "
        "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
        "Use three sentences maximum. Keep the answer as concise as possible. \n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Helpful Answer: "
    )
    llm: ChatOllama = None
    prompt: PromptTemplate = PromptTemplate(
        input_variables=["context", "question"],
        template=_template,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm = ChatOllama(model=self.model, temperature=0.0, verbose=False)

    @weave.op()
    def predict(self, query: str, context: str | List[str]):
        if isinstance(context, List):
            context = "\n".join(context)

        augmented_query = self.prompt.invoke({"context": context, "question": query})

        return self.llm.invoke(augmented_query)
