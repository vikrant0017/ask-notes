import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
"""
Integrate all the components under the RAG class
"""
class RAG:
    def __init__(self, retriever, generator, ingestor):
        self.retriever = retriever
        self.generator = generator
        self.ingestor = ingestor
    
    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, query):
        contexts = self.retriever.query(query)
        response = self.generator.query(query, context=self._format_docs(contexts))
        contexts = [doc.page_content for doc in contexts]
        return response, contexts

    def ingest(self, filepath: str = None, file = None, dir: str = None, text: str = None):
        """Can provide multiple arguments or single"""
        docs = []
        if filepath:
            docs += self.ingestor.load_file(filepath=filepath)
        if file:
            docs += self.ingestor.load_file(file=file)
        if dir:
            docs += self.ingestor.load_dir(dir=dir)
        if text:
            docs += self.ingestor.load_text(text=text)
        self.retriever.add_docs(docs)
        