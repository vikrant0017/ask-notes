from abc import ABC, abstractmethod
from typing import List

class BaseGenerator(ABC):

    @abstractmethod
    def query(self, query: str, context: str | List[str]) -> str: 
        """Generate the response based on the query and the context - most probably the output of a retriver"""
    

