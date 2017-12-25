import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


class PartOfSpeechCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.count_vectorizer = CountVectorizer()

    @staticmethod
    def __extract_pos_tags(examples):
        pos_tags_per_document = []
        for document in examples:
            pos_tags = []
            tokenized_words = nltk.word_tokenize(document)
            pos_tags_tuple = nltk.pos_tag(tokenized_words)
            for tag_tuple in pos_tags_tuple:
                pos_tags.append(tag_tuple[1])
            pos_tags_per_document.append(str.join(' ', pos_tags))
        return pos_tags_per_document

    def fit(self, examples):
        pos_tags = self.__extract_pos_tags(examples)
        self.count_vectorizer.fit(pos_tags)
        return self

    def transform(self, examples):
        pos_tags = self.__extract_pos_tags(examples)
        return self.count_vectorizer.transform(pos_tags)


class PosTags:
    @staticmethod
    def adjectives():
        return ['JJ', 'JJS', 'JJR']

    @staticmethod
    def adverbs():
        return ['RB', 'RBR', 'RBS']