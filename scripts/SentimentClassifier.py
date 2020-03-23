import re
import pandas as pd

from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class SentimentClassifier:
    def __init__(self, training_data_file, testing_data_file=None):
        self.training_data_file = training_data_file
        self.testing_data_file = testing_data_file
        self.processed_content = []
        self.sentiment_labels = []

    def _read_csv(self):
        csv_data = pd.read_csv(self.training_data_file)
        content = csv_data.iloc[:, 0].values
        sentiment = csv_data.iloc[:, 1].values
        
        return content, sentiment
    
    def _clean_data(self):
        content, sentiment = self._read_csv()

        port_stem = PorterStemmer()
        for c in content:
            pre_processed = re.sub(r"\b@\w+", "", c)
            pre_processed = re.sub(r'\W', ' ', str(pre_processed))
            pre_processed = re.sub(r'\s+[a-zA-Z]\s+', ' ', pre_processed)
            pre_processed = re.sub(r'\s+', ' ', pre_processed, flags=re.I)
            # remove numbers
            pre_processed = re.sub(r'\d+', '', pre_processed)
            pre_processed = pre_processed.strip()
            pre_processed = pre_processed.lower()

            pre_stemming_array = word_tokenize(pre_processed)
            stem_array = []
            for p in pre_stemming_array:
                stem_array.append(port_stem.stem(p))
            pre_processed = " ".join(stem_array)

            self.processed_content.append(pre_processed)

        self.sentiment_labels = sentiment

    def _word_vectorize(self):
        self._clean_data()
        vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
        self.processed_content = vectorizer.fit_transform(self.processed_content).toarray()

    def train(self):
        self._word_vectorize()
       
        x_train, x_test, y_train, y_test = train_test_split(self.processed_content, self.sentiment_labels, test_size=0.2, random_state=0)
        text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
        text_classifier.fit(x_train, y_train)
        predictions = text_classifier.predict(x_test)
        print(confusion_matrix(y_test,predictions))
        print(classification_report(y_test,predictions))
        print(accuracy_score(y_test, predictions))
        
s = SentimentClassifier("../pos_neg.csv")
s.train()