SENTIMENT_MAPPING = {
    0: "very negative ",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "very positive"
}
WIKIPEDIA = "Wikipedia"


class DefaultModels:
    Sentiment: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    Summarization: str = "Falconsai/text_summarization"
