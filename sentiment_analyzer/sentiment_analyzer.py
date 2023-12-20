"""
An example of a sentiment analysis function that uses a pre-trained model from the Hugging Face model hub, where
the model exists in a local folder.
The function receives a text input and returns the predicted sentiment of the text.
"""

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sentiment_analyzer.preprocessing as preprocessing
from common.constants import SentimentMapping


def init_context(context):
    # load the model here, just once
    if not getattr(context.user_data, "initialized", False):
        # get model name from environment variable
        model_name = os.getenv("MODEL_NAME", "nlptown/bert-base-multilingual-uncased-sentiment")

        context.logger.info(f"Initializing the model. Model name: {model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = preprocessing.Tokenizer(AutoTokenizer.from_pretrained(model_name))

        # save the model and tokenizer in the context user data, to be shared between invocations
        setattr(context.user_data, "model", model)
        setattr(context.user_data, "tokenizer", tokenizer)
        setattr(context.user_data, "initialized", True)

        context.logger.info("Model initialized")


def handler(context, event):
    # parse the input text from the event body
    input_text = event.body.decode().strip()

    context.logger.info(f"Received input text: {input_text}")

    # call the sentiment prediction function
    predicted_sentiment = _predict_text_sentiment(
        context.user_data.tokenizer,
        context.user_data.model,
        input_text,
    )

    context.logger.info(f"Predicted sentiment: {predicted_sentiment}")

    # return the sentiment
    body = {
        "input_text": input_text,
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
    # Tokenize the input text
    inputs = tokenizer.tokenize(
        input_text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    # Run the text through the sentiment model
    sentiment_model.eval()
    outputs = sentiment_model(**inputs)

    # Get the classification
    predicted_class_sentiment = outputs.logits.argmax(dim=1).item()

    # Map the sentiment label to a rating
    predicted_label_sentiment = SentimentMapping.to_str(predicted_class_sentiment)
    return predicted_label_sentiment
