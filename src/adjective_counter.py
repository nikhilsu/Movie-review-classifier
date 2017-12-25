import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.feature_eng import TokenExtractor
from src.pos_tagger import PosTags


class AdjectiveCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizer, extractor):
        self.count_vectorizer = vectorizer
        self.tokenExtractor = extractor

    def __extract_adjectives(self, examples):
        adjectives = PosTags.adjectives()
        return self.tokenExtractor.extract_tokens_with_given_tag(examples, adjectives)

    # training data
    def fit(self, examples):
        adjectives_per_example = self.__extract_adjectives(examples)
        self.count_vectorizer.fit(adjectives_per_example)
        return self

    # run the test
    def transform(self, examples):
        adjectives_per_example = self.__extract_adjectives(examples)
        return self.count_vectorizer.transform(adjectives_per_example)


class NumberOfAdjectivesCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.token_extractor = TokenExtractor()

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        adjectives_per_example = self.token_extractor.extract_tokens_with_given_tag(examples, PosTags.adjectives())
        i = 0
        for adjectives in adjectives_per_example:
            features[i, 0] = len(adjectives.split(' '))
            i += 1
        return features