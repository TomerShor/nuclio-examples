import os
import wikipedia
import common.constants
from transformers import pipeline


def init_context(context):
    # initialize the summarization model
    if getattr(context, "initialized", False):
        summarization_model_name = os.getenv("MODEL_NAME", common.constants.DefaultModels.Summarization)
        summarizer = pipeline("summarization", model=summarization_model_name)
        setattr(context.user_data, "summarizer", summarizer)
        setattr(context.user_data, "summarization_model_name", f"{summarization_model_name} model")
        setattr(context.user_data, "initialized", True)


def _get_wiki_summary_len(article_title):
    return wikipedia.summary(article_title)


def _get_model_summary_len(summarizer, article_title):
    article = wikipedia.page(article_title)
    result = summarizer(article.content)
    return len(result[0]["summary_text"])


def handler(context, event):
    article_title = event.body.decode().strip()

    wiki_summary_len = _get_wiki_summary_len(article_title)
    model_summary_len = _get_model_summary_len(context.user_data.summarizer, article_title)

    shorter = common.constants.WIKIPEDIA
    longer = context.user_data.summarization_model_name
    if model_summary_len < wiki_summary_len:
        shorter = context.user_data.summarization_model_name
        longer = common.constants.WIKIPEDIA
    message = f"For article \"{article_title}\", {shorter} summary is shorter than the {longer} summary!"

    return context.Response(
        body=message,
        content_type="text/plain",
        status_code=200,
    )
