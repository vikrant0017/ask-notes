import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter 
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

EMBEDDING_MODEL = "models/text-embedding-004"
API_KEY = os.getenv('GOOGLE_API_KEY')
CHROMA_DIR = 'docs/chroma'
LLM_MODEL = 'gemini-2.0-flash-exp'

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""

prompt = PromptTemplate(input_variables=["context", "question"],template=template,)

qa_chain = None
def initialize(dir_name):
    # Load the documents
    loader = DirectoryLoader(dir_name, glob='**/*.txt', loader_cls=TextLoader) # load all files including nested .txt
    docs = loader.load()

    # Split 
    splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=20, separator='.')
    chunks = splitter.split_documents(docs)

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=API_KEY, temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,  # embedding associated with chunks
        persist_directory=CHROMA_DIR
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    global qa_chain
    qa_chain = (
        # input
        {
            "context": vector_db.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    

# combine the contents of the documents for context

def ask(query):
    return qa_chain.invoke(query)
