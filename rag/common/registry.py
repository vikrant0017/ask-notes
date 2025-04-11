class Registry:
    mapping = {
        'scorers': {},
        'retrievers': {},
        'generators': {},
        'chunkers': {},
        'ingestors': {}
    }

    @classmethod
    def register_scorer(cls, identifier: str):
        def wrap(scorer: callable):
            print(f'Registered scorer: {identifier}')
            cls.mapping['scorers'][identifier] = scorer
            return scorer # Return the original callable without modification
        return wrap

    @classmethod
    def register_ingestor(cls, identifier: str):
        def wrap(ingestor: callable):
            print(f'Registered ingestor: {identifier}')
            cls.mapping['ingestors'][identifier] = ingestor
            return ingestor
        return wrap

    @classmethod
    def register_retriever(cls, identifier: str):
        def wrap(retriever: callable):
            print(f'Registered retriever: {identifier}')
            cls.mapping['retrievers'][identifier] = retriever
            return retriever
        return wrap

    @classmethod
    def register_generator(cls, identifier: str):
        def wrap(generator: callable):
            print(f'Registered generator: {identifier}')
            cls.mapping['generators'][identifier] = generator
            return generator
        return wrap

    @classmethod
    def register_chunker(cls, identifier: str):
        def wrap(chunker: callable):
            print(f'Registered chunker: {identifier}')
            cls.mapping['chunkers'][identifier] = chunker
            return chunker
        return wrap
    
    @classmethod
    def get_retriever(cls, identifier: str):
        retriever_cls = cls.mapping['retrievers'].get(identifier)
        if retriever_cls is None:
            raise KeyError(f'Retriever with the name- "{identifier}" not found.')
        return retriever_cls

    @classmethod
    def get_generator(cls, identifier: str):
        generator_cls = cls.mapping['generators'].get(identifier)
        if generator_cls is None:
            raise KeyError(f'Generator with the name- "{identifier}" not found.')
        return generator_cls

    @classmethod
    def get_chunker(cls, identifier: str):
        chunker_cls = cls.mapping['chunkers'].get(identifier)
        if chunker_cls is None:
            raise KeyError(f'Chunker with the name- "{identifier}" not found.')
        return chunker_cls

    @classmethod
    def get_ingestor(cls, identifier: str):
        ingestor_cls = cls.mapping['ingestors'].get(identifier)
        if ingestor_cls is None:
            raise KeyError(f'Ingestor with the name- "{identifier}" not found.')
        return ingestor_cls

    @classmethod
    def get_scorer(cls, identifier: str):
        scorer_cls = cls.mapping['scorers'].get(identifier)
        if scorer_cls is None:
            raise KeyError(f'Scorer with the name- "{identifier}" not found.')
        return scorer_cls
            

registry = Registry()