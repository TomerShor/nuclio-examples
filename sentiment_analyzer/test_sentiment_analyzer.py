import logging
import os
import unittest
import nuclio_sdk
import sys

import sentiment_analyzer
from common.constants import SentimentMapping


class EventMock:
    def __init__(self, body):
        self.body = body


class TestSentimentAnalyzer(unittest.TestCase):

    def setUp(self):
        # create logger
        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler(stream=sys.stdout))

        # create a context object that contains a logger
        self.context = nuclio_sdk.Context(logger=self.logger)

    def test_sentiment_analyzer(self):
        # put a model name in the environment variable
        os.environ["MODEL_NAME"] = "nlptown/bert-base-multilingual-uncased-sentiment"

        sentiment_analyzer.init_context(self.context)

        # create an event with a message
        message = "The food was awful, and I didn't enjoy this restaurant at all".encode('utf-8')
        event = EventMock(body=message)

        # call the handler function
        response = sentiment_analyzer.handler(self.context, event)

        self.assertEqual(response.status_code, 200)
        result = response.body
        self.logger.debug(f"Got Result: {result}")
        self.assertEqual(result["input_text"], message.decode('utf-8'))
        self.assertEqual(result["predicted_sentiment"], SentimentMapping.VERY_NEGATIVE.to_lower())
