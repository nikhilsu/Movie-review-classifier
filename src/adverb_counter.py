from sklearn.base import BaseEstimator, TransformerMixin

from src.pos_tagger import PosTags


class AdverbAdjectiveCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizer, extractor):
        self.count_vectorizer = vectorizer
        self.token_extractor = extractor

    def __extract_adverb_adjectives(self, examples):
        adverbs_and_adjectives = PosTags.adjectives() + PosTags.adverbs()
        return self.token_extractor.extract_tokens_with_given_tag(examples, adverbs_and_adjectives)

    # training data
    def fit(self, examples):
        adverb_adjectives_per_example = self.__extract_adverb_adjectives(examples)
        self.count_vectorizer.fit(adverb_adjectives_per_example)
        return self

    # run the test
    def transform(self, examples):
        adjectives_per_example = self.__extract_adverb_adjectives(examples)
        return self.count_vectorizer.transform(adjectives_per_example)