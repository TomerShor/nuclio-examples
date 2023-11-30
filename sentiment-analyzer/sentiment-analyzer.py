"""
An example of a sentiment analysis function that uses a pre-trained model from the Hugging Face model hub, where
the model exists in a local folder.
The function receives a text input and returns the predicted sentiment of the text.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import preprocessing
from constants import Constants


def init_context(context):
    # load the model here, just once
    if not getattr(context.user_data, "initialized", False):
        model_name = os.getenv("MODEL_NAME", "nlptown/bert-base-multilingual-uncased-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = preprocessing.Tokenizer(AutoTokenizer.from_pretrained(model_name))
        setattr(context.user_data, "model", model)
        setattr(context.user_data, "tokenizer", tokenizer)
        setattr(context.user_data, "initialized", True)


def handler(context, event):
    # parse the input text from the event body
    input = event.body.decode().strip()

    context.logger.info(f"Received input text: {input}")

    # call the sentiment function
    predicted_sentiment = _predict_text_sentiment(context.user_data.tokenizer, context.user_data.model, input)

    # return the sentiment
    body = {
        "input_text": input,
        "predicted_sentiment": predicted_sentiment
    }
    return context.Response(
        body=body,
        headers={
            "Content-Type": "application/json"
        },
        content_type="text/plain",
        status_code=200,
    )


def _predict_text_sentiment(tokenizer, sentiment_model, input_text):
    inputs = tokenizer.tokenize(
        input_text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    sentiment_model.eval()
    with torch.no_grad():
        outputs = sentiment_model(**inputs)

    # Get the classification score for each token and pick the highest one
    logits = outputs.logits
    predicted_class_sentiment = torch.argmax(logits, dim=1).item()

    # Map the sentiment label to a rating
    predicted_label_sentiment = Constants.SENTIMENT_MAPPING[predicted_class_sentiment]
    return predicted_label_sentiment
