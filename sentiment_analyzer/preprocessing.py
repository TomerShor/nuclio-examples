class Tokenizer(object):
    def __init__(self, tokenizer_function):
        self.tokenizer_function = tokenizer_function

    def tokenize(self, text, **kwargs):
        text = text.lower()
        return self.tokenizer_function(
            text,
            **kwargs
        )
