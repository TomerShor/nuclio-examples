import enum


class SentimentMapping(enum.Enum):
    VERY_NEGATIVE = 0
    NEGATIVE = 1
    NEUTRAL = 2
    POSITIVE = 3
    VERY_POSITIVE = 4

    @classmethod
    def to_str(cls, sentiment):
        return cls(sentiment).name.lower().replace("_", " ")

    def to_lower(self):
        return self.name.lower().replace("_", " ")


WIKIPEDIA = "Wikipedia"


class DefaultModels:
    Sentiment: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    Summarization: str = "Falconsai/text_summarization"
