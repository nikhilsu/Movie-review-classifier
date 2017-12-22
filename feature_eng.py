import json

import nltk
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

SEED = 1323

'''
The ItemSelector class was created by Matt Terry to help with using
Feature Unions on Heterogeneous Data Sources

All credit goes to Matt Terry for the ItemSelector class below

For more information:
http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
'''


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TextLengthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            features[i, 0] = len(ex)
            i += 1

        return features


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


class PartOfSpeechCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.count_vectorizer = CountVectorizer()

    def __extract_pos_tags(self, examples):
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


class TokenExtractor:
    def extract_tokens_with_given_tag(self, source_strings, tags_to_extract):
        extracted_tokens_per_example = []
        for example in source_strings:
            extracted_tokens = []
            tokenized_words = nltk.word_tokenize(example)
            pos_tags = nltk.pos_tag(tokenized_words)
            for tag in pos_tags:
                if tag[1] in tags_to_extract:
                    extracted_tokens.append(tag[0])
            extracted_tokens_per_example.append(str.join(' ', extracted_tokens))
        return extracted_tokens_per_example


class PosTags:
    @staticmethod
    def adjectives():
        return ['JJ', 'JJS', 'JJR']

    @staticmethod
    def adverbs():
        return ['RB', 'RBR', 'RBS']


class Featurizer:
    def __init__(self):
        self.all_features = FeatureUnion([
            ('text_stats', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('text_length', TextLengthTransformer())
            ])),
            ('pos_stats', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('pos_count', PartOfSpeechCountTransformer()),
            ])),
            # ('adjective_stats', Pipeline([
            #     ('selector', ItemSelector(key='text')),
            #     ('adjective_count', NumberOfAdjectivesCountTransformer()),
            # ])),
            ('adjective_stats', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('adjective_count', AdjectiveCountTransformer(CountVectorizer(min_df=0.15, max_df=0.8),
                                                              TokenExtractor())),
            ])),
            ('adverb_adjective_stats', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('adjective_count', AdverbAdjectiveCountTransformer(CountVectorizer(ngram_range=(1, 2),
                                                                                    min_df=0.15, max_df=0.8),
                                                                    TokenExtractor)),
            ])),
            # N_gram feature transformer using CountVectorizer
            # ('vectorize', Pipeline([
            #     ('selector', ItemSelector(key='text')),
            #     ('count_vect', CountVectorizer(ngram_range=(1, 3))),
            # ]))
        ])

    def train_feature(self, examples):
        return self.all_features.fit_transform(examples)

    def test_feature(self, examples):
        return self.all_features.transform(examples)


def fit_with_sgd():
    lr.fit(feat_train, y_train)
    y_pred = lr.predict(feat_train)
    accuracy = accuracy_score(y_pred, y_train)
    print("Accuracy on training set =", accuracy)
    y_pred = lr.predict(feat_test)
    accuracy = accuracy_score(y_pred, y_test)
    print("Accuracy on test set =", accuracy)


def fit_with_grid_search_cv():
    param_grid = {'alpha': [0.1, 0.001]}
    lr_cv = GridSearchCV(lr, param_grid, cv=3)
    lr_cv.fit(feat_train, y_train)
    train_y_prediction = lr_cv.predict(feat_train)
    accuracy = accuracy_score(train_y_prediction, y_train)
    print("Optimal Regularization parameter: {}".format(lr_cv.best_params_['alpha']))
    print("Accuracy: {}".format(accuracy))


if __name__ == "__main__":
    data_set_x = []
    data_set_y = []

    with open('../data/movie_review_data.json') as f:
        data = json.load(f)
        for d in data['data']:
            data_set_x.append(d['text'])
            data_set_y.append(d['label'])

    X_train, X_test, y_train, y_test = train_test_split(data_set_x, data_set_y, test_size=0.3, random_state=SEED)

    feat = Featurizer()

    labels = []
    for l in y_train:
        if l not in labels:
            labels.append(l)
    print("Label set: %s\n" % str(labels))

    feat_train = feat.train_feature({
        'text': [t for t in X_train]
    })

    # feat_train = normalize(feat_train)

    feat_test = feat.test_feature({
        'text': [t for t in X_test]
    })

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', alpha=0.001, max_iter=15000, shuffle=True, verbose=2)

    fit_with_sgd()
    # fit_with_grid_search_cv()
