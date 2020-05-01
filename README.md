# Reddit Sentiment Analysis 

## Overview
This research was done on sentiment analysis and how a sentiment classifier can be used to determine how redditors feel about certain posts/topics. This project explores both machine learning and algorithmic approaches to this problem and considers the benefits of both. 

The algorithmic library that was explored during this project was Vader. This library is a rule based approach to this problem that can handle idioms and negations. I created a basic algorithm that merely checked for word occurrences to use along with Vader and my machine learning approach to see how a simple approach would compare to these more complex approaches.

The machine learning approach used during this project was a Sklearn random forest classifier. This model was trained on a dataset containing 50,000 verbose movie reviews and achieved ~85% accuracy on a testing set, though this accuracy can vary in practice.

## Exploration of ML Approach: Challenges
There were multiple challenges with building a model that can accurately classify sentiments. First of which, was finding an appropriate data set. Throughout the research I trained on multiple datasets. Some were very verbose akin to the movie review one that I settled on. Some from social media which included a lot of slang and misspelled words and some with multiple different classification outcomes than merely positive or negative.

The ones from social media had too many word misspellings and slang. The dataset was likely created during the infancy of social media when users used more word shortenings and abbreviations than modern users do today. This dataset is not as practical when the target users are redditors who are more verbose than social media platforms in the early 2010’s. 

The one containing multiple classification outputs had a lower accuracy of around 30%. This makes sense because it is very difficult to accurately and consistently distinguish similar sentiments such as anger vs hate. The problem of having multiple sentiments contained in the same string is a pitfall. Take the text: “I am thoroughly upset at what is going on with the world. I am scared for my life and the lives of my family members. Why did our government not warn us of this danger? I despise the situation we are in and I am so sad for those who have lost their lives. I hate this world right now”. This text can fall under multiple categories such as anger, hate, and worry. This was interesting but due to time constraints was not explored further, though would be interesting future research.

The data set chosen was a verbose set of movie reviews with a binary classification: positive or negative. This dataset was lengthy and contained text that was verbose enough to create an effective model for Reddit. 

## Exploration of ML Approach: Preprocessing
The models I used were SciKitLearn's [random forest classifier](##Exploration-of-ML:-Random-Forests-in-a-Nutshell). Though, before training the text had to be preprocessed and a feature set had to be created so the model could be trainied. 

Firstly, we must preprocess the data. 

The first step in this process is converting the words in the text to lowercase. This can be done in python like this:
```Python
for word in text:
    word = word.lower()
```
This ensures that words like "Hello", "hEllo" and "hello" are all equvalent. 

The next step in text preprocessing would be to convert all words to their stems. This will transform words like "amazingly" to "amaze". Doing this makes all words with the same root or stem equivalent so that when we create a feature set these words all treated as the same. This can be done in python using external libraries like this:
```Python
from nltk.stem import PorterStemmer
port_stem = PorterStemmer()

for word in text:
    word = port_stem.stem(word)
```

After doing these steps there are numerous techniques you can use to tweak accuracy. I did not use these because of the duration it took for my model to be trained due to the size of the dataset but these are a few more ways to further preprocess the data. 

Once of these steps is to remove stop words. A stop words is a word like "the", "a", or "of" these words occur very frequently and carry little meaning. Since our feature set will be a dictionary containing count of the k most frequnt words we would like to avoid these words taking spots in our feature set. We can accomplish this with the following code:

```Python
from ntlk.corpus import stopwords
words = 'He the best of all of them'
new_words = [word for word in words if word not in stopwords.words('english')]
>>> new_words
['He', 'best']
```

More preprocessing steps would be to remove symbols. A simple python regular expression for this is:

```Python
import re
my_string = 'hello&% ho&^&w ar*&e y%&o*u'
my_string = re.sub('[^a-zA-Z0-9 \n\.]', '', my_string)
>>> my_string
'hello how are you'
```

## Exploration of ML Approach: Building Features
Now that we have processed this data, we must now determine how we can train a machine learning algorithm to perform classifications on this data. We cant just a bunch of text to a random forest algorithm and expect to get very far. This is were feature sets come in. 

For our word set we want to find the freqencies of all of the words in the dataset after these words are preprocessed. We then want to take the k most frequent words where k is the feature set size in which we specify. This can be done using nltk as shown below:

``` Python
import nltk
size_of_feature_set = 100

frequency = nltk.FreqDist()
for word in words:
    frequency[word] += 1

feature_words = list(frequency_dist)[:size_of_feature_set]
```
Now that we have the word features of the dataset, we can  use these to build the feature sets of each row in our dataset. These features sets are dictionaries with the word as the key and a boolean representing whether or not the word is in the row for the value. We do this for all of the words contained in the word features. After we do this for every row in our dataset we can now train a forest classifier on these feature sets.

## Exploration of ML: Random Forests in a Nutshell

Lets say we have an image with just a solid color. This color is a bluish color that looks like it might also be purple. We want to know what this color is so we post a poll online so people can vote on the color. Most likely the answer with the most votes will be correct. This is an odd example but we do see this all of the time online. On chegg, for example the correct answers tend to get more upvotes. 
 
Why are we talking about colors? Well a random forest works in a similar way. A random forest is made up of many decision trees. The decision trees in a random forest essentially vote on the classification. The answer with the most votes is chosen for the classification.

These decision trees which make up a random forest are essentialy a series of questions. Each node in the tree represents a question with each of the two pointers to the next node being the answers to the question. Below is an example of this where the tree is representing the decision of what clothes to wear.

```
Is it cold outside?
      / \
     /   \
yes /     \ no
   /       \
  /         \
Jacket    T-shirt
```

This idea is scalable and can be used for more complex classifcations.

## Algorithm Approach: Naive
After training a model with the Sklearn random forest classfier I decided to come up with my own algorithmic approach to this problem. 

This github repository contains seperate files of positive and negative words: https://github.com/shekhargulati/sentiment-analysis-python/tree/master/opinion-lexicon-English

I put these words in a python set and then converted the text to a set to remove dupicates. I then took the intersection between the text and the negative words, and the text and the positve words. For the neutral words, I merely took the percentage of words that were not in the intersection of either set. Here is the script: 

```Python
negative_score = len(negativewords & text_set) / len(text_set) * 100
positive_score = len(positivewords & text_set) / len(text_set) * 100
neutral_score = (len(text_set) - (len(negativewords & text_set) + len(positivewords & text_set))) / len(text_set) * 100
```

Another variation of this approach that I experimented with involved taking length of these intersected sets as opposed to using the percentages. Take the two below texts for example:
```
"I hate this movie with a burning passion. The screenplay was awful. The music was uninpiring. The acting was a laughable. This movie was a waste of 2 hours and I would honestly rather watch grass growing for 10 hours straight than 30 seconds of this piece of utter garbage."
```


```
"I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it.I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it."
```

I believe the first one should have a higher negative score than the second one. The first one is verbose and really shows how much the author disliked the movie. The second one was low effort and showed little meaning. If we remove duplicate words by taking the set of both of these, the first text will have a longer set than the second one.
```Python
>>>len(set("I hate this movie with a burning passion. The screenplay was awful. The music was uninpiring. The acting was a laughable. This movie was a waste of 2 hours and I would honestly rather watch grass growing for 10 hours straight than 30 seconds of this piece of utter garbage.".split()))
37
```

```Python
>>> len(set("I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it.I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it.".split()))
4
```

This is a simple approach I came up with to derrive a sentiment intensity of various strings.

## Algorithm Approach: Advanced (Vader)
Vader is a robost sentiment analysis library. This library works by having dictionaries of various word categories. Each word in these dictionaries have multipliers that affect the overall sentiments in the words. The categories include negations, emojis, boosters (like "absolutely" and "very"), idioms, and special cases. 

Here is an example of this library in action:

```Python
>>> from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
>>> analyzer = SentimentIntensityAnalyzer()
>>> analyzer.polarity_scores("The movie was amazing")

{'neg': 0.0, 'neu': 0.441, 'pos': 0.559, 'compound': 0.5859}

>>> analyzer.polarity_scores("The movie was terrible")

{'neg': 0.508, 'neu': 0.492, 'pos': 0.0, 'compound': -0.4767}
```
The "compound" value in these scores are the overal sentiment scores with the sign representing the positivity or negativity(-) of the text.

## Comparing Both Approaches
To compare these approaches lets consider the following sentences:
```
1. That movie was good.
```
and

```
2. That was not good.
```
These sentences are very simlar (atleast for a computer) but have the opposite meaning. A machine learning model has to be trained to hopefully be able to detect this. Lets see if ours can:

```Python
>>> c.classify_new_sentence("That was good")
positive
>>> c.classify_new_sentence("That was not good")
negative
```
Our machine learning model was able to correctly classify these two examples but sometimes the model can be inaccurate.

These type of classifications are more accurate using an algorithmic approach such as vader. Lets take the following word:
```
good
```
This will yield a positive sentiment using vader. Now, lets add a negation to this word:

```
not good
```
This, as expected, yields a negative sentiment. Now how about a negation of a negation:

```
not not good
```
This yields a positive negation because the two negations cancel each other out. In general, when there is an odd amount of negations, the outcome is negative and when there is an even amount of negations, the outcome is positive. This can be represented by:
```
<negation>^n <word> where positive if n % 2 = 0 else negative 
```

It would be very difficult (if possible) for a machine learning model to learn this particular pattern. 

## Conclusion

## Sources
Thank you to the following articles and tutorials that made this project possible:
* https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/
* https://edmundmartin.com/text-classification-with-python-nltk/
* https://towardsdatascience.com/understanding-random-forest-58381e0602d2
* https://github.com/shekhargulati/sentiment-analysis-python/tree/master/opinion-lexicon-English
* https://github.com/cjhutto/vaderSentiment
## Contact
[Reginald Thomas](mailto:reginaldcolethomas@gmail.com)

