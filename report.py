class Report:
    extractors = {}

    def __init__(self, pdf_path: str) -> None:
        self.pdf_path = pdf_path

    @classmethod
    def register(cls, name, extractor):
        cls.extractors[name] = extractor

    def extract(self, name):
        if name not in self.extractors:
            raise ValueError(f"Unknown extractor: {name}")
        return self.extractors[name].extract(self)
