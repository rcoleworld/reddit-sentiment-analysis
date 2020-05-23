import praw
import ForestClassifier as ForestClassifier
from positivewords import positivewords
from negativewords import negativewords
from keys import keys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class RedditSentimentScorer:
    def __init__(self, model, vocab, dataset, post=None):
        self.classification = None
        self.score = 0
        self.classifier = ForestClassifier.ForestClassifier(dataset)
        self.results = list()
        self.classifier.load_model(model, vocab)
        self.reddit = praw.Reddit(client_id='z4_4e_6dkAmX7Q',
                     client_secret='eAjbFXhLw_5ZcL-1qbITE1FE1hA',
                     password=keys["password"],
                     user_agent='testscript by /u/sentimentanaly490c',
                     username=keys["username"])
        self.post = self.reddit.submission(id=post)

    def _score_comment(self, comment):
        result = self.classifier.classify_new_sentence(comment)
        
        text = word_tokenize(comment)
        text_set = set([t for t in text if t not in stopwords.words()])
        negative_score = len(negativewords & text_set) / len(text_set) * 100
        positive_score = len(positivewords & text_set) / len(text_set) * 100
        neutral_score = (len(text_set) - (len(negativewords & text_set) + len(positivewords & text_set))) / len(text_set) * 100
        score = {
            "positive": "{}%".format(round(positive_score, 1)), 
            "neutral": "{}%".format(round(neutral_score, 1)), 
            "negative": "{}%".format(round(negative_score, 1))
        }
        self.results.append((result, score, comment))

        return score

    def _reddit_data_stream(self):
        print(self.reddit.user.me())
    
    def get_post_score(self):
        for comment in self.post.comments:
            self._score_comment(comment.body)

    def vader_scores(self):
        analyzer = SentimentIntensityAnalyzer()
        for comment in self.post.comments:
            self.results.append((comment.body, analyzer.polarity_scores(comment.body)))
            

r = RedditSentimentScorer("../savedmodels/imdb3.pkl", "../savedmodels/vocab-imdb3.pkl","../datasets/IMDB_Dataset.csv", post='fzh63j')

r.get_post_score()
# r.vader_scores()
for result in r.results:
    print(result)
    print()