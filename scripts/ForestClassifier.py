# Used code from <https://edmundmartin.com/text-classification-with-python-nltk/> as a template for this model.
# Replaced classifier and added word stemming.
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from random import shuffle
import pickle

class ForestClassifier:
    def __init__(self, csv_file, featureset_size=5000, test_ratio=0.1):
        self.csv_file = csv_file
        self.documents = []
        self.words = []
        self.featureset_size = featureset_size
        self.test_ratio = test_ratio
        self.feature_words = None
        self.classifier = None
 
    def _read_csv(self):
        port_stem = PorterStemmer()
        with open(self.csv_file, 'r') as input_csv:
            for item in input_csv:
                item = item.split(',')
                doc, label = re.findall('\w+', ''.join(item[:-1]).lower()), item[-1].strip()
                post_stem_arr = list()
                for word in doc:
                    self.words.append(port_stem.stem(word.lower()))
                    post_stem_arr.append(port_stem.stem(word))
                self.documents.append((post_stem_arr, label))
             
    def _generate_word_features(self):
        frequency_dist = nltk.FreqDist()
        for word in self.words:
            frequency_dist[word] += 1
        self.feature_words = list(frequency_dist)[:self.featureset_size]
 
    def __document_features(self, document):
        document_words = set(document)
        features = {}
        for word in self.feature_words:
            features['contains({})'.format(word)] = (word in document_words)
        return features
 
    def train_sklearn_classifier(self, sk_learn_classifier):
        if not self.feature_words:
            self._read_csv()
            self._generate_word_features()
        shuffle(self.documents)
        feature_sets = [(self.__document_features(d), c) for (d, c) in self.documents]
        cutoff = int(len(feature_sets) * self.test_ratio)
        train_set, test_set = feature_sets[cutoff:], feature_sets[:cutoff]
        self.classifier = SklearnClassifier(sk_learn_classifier()).train(train_set)
        print('Achieved {0:.2f}% accuracy against training set'.format(nltk.classify.accuracy(self.classifier, train_set)*100))
        print('Achieved {0:.2f}% accuracy against test set'.format(nltk.classify.accuracy(self.classifier, test_set)*100))
 
    def classify_new_sentence(self, sentence):
        port_stem = PorterStemmer()
        stemmed_sentence = list()
        for word in sentence.split():
            stemmed_sentence.append(port_stem.stem(word))
        sentence = " ".join(stemmed_sentence)
        if not self.feature_words:
            self._read_csv()
            self._generate_word_features()
        test_features = {}
        for word in self.feature_words:
            test_features['contains({})'.format(word.lower())] = (word.lower() in nltk.word_tokenize(sentence))
        return self.classifier.classify(test_features)
 
    def save_model(self, filename):
        save_classifier = open(filename, "wb")
        pickle.dump(self.classifier, save_classifier)
        save_classifier.close()
        save_vocab = open('vocab-{}'.format(filename), "wb")
        pickle.dump(self.feature_words, save_vocab)
        save_vocab.close()
 
    def load_model(self, model_filename, vocab_filename):
        classifier_f = open(model_filename, "rb")
        self.classifier = pickle.load(classifier_f)
        classifier_f.close()
        vocab_f = open(vocab_filename, "rb")
        self.feature_words = pickle.load(vocab_f)
        vocab_f.close()
 
c = ForestClassifier('../datasets/IMDB_Dataset.csv', featureset_size=5000)

c.load_model("../savedmodels/imdb3.pkl", "../savedmodels/vocab-imdb3.pkl")


# print(result)