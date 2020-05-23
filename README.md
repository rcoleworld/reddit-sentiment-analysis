# Reddit Sentiment Analysis 

## Overview

Natural Language Processing (NLP) is an intriguing use of machine learning. This field is what spiked my interest in machine learning. This project is related to NLP. More specifically, it explores sentiment analysis and how a sentiment classifier can be used to determine how Redditors feel about certain posts/topics. Why Redditors? Well, Reddit offers an [API]( https://praw.readthedocs.io/en/latest/) that is less limited than most alternative sources. This project explores both machine learning and algorithmic approaches to classifying sentiment portrayed by text.

The algorithmic library that was explored during this project is Vader. This library is a rule-based approach to this problem that can handle idioms and even emojis. I also created a basic algorithm to see how the amount of positive and negative word occurrences relates to the overall sentiment of the text.

The machine learning model used during this project was a Sklearn random forest classifier. This model was trained on a dataset containing 50,000 verbose movie reviews and achieved ~85% accuracy on a testing set, though this accuracy can vary in practice.

## Exploration of ML Approach: Challenges
There were multiple challenges with building a model that can accurately classify sentiments. The first one was finding an appropriate data set. Throughout the research I trained on multiple datasets. Some were very verbose akin to the movie review dataset which I settled on. Some were from social media and included a lot of slang and misspelled words. One had multiple classifications other than the typical positive and negative sentiments.

The ones from social media had too many word misspellings and slang. These datasets were likely created near the infancy of social media when users used more word shortenings and abbreviations than modern users do today. These datasets were not as practical when the target users are Redditors who are more verbose than social media platforms in the early 2010s. 

The dataset containing multiple classification outputs had an accuracy of around 30%. This makes sense because it is very difficult to accurately and consistently distinguish similar sentiments such as anger vs hate. The problem of having multiple sentiments contained in the same string is a pitfall. Take the text: “I am thoroughly upset at what is going on with the world. I am scared for my life and the lives of my family members. Why did our government not warn us of this danger? I despise the situation we are in and I am so sad for those who have lost their lives. I hate this world right now”. This text can fall under multiple categories such as anger, hate, and worry. This was a very interesting problem but due to time constraints was not explored further, though it would be great for future research.

The dataset chosen was a verbose set of movie reviews with a binary classification: positive or negative. This dataset was lengthy and contained text that was verbose enough to create an effective model for Reddit. 

## Exploration of ML Approach: Preprocessing
The models I trained utilized SciKitLearn's [random forest classifier](##Exploration-of-ML:-Random-Forests-in-a-Nutshell). Before training, the text had to be preprocessed and a feature set had to be created.

First, we must preprocess the data. 

The first step in this process is to convert the words in the text to lowercase. This can be done in Python like this:
```Python
for word in text:
    word = word.lower()
```
This ensures that words like "Hello", "hEllo" and "hello" are all equivalent. 

The next step in text preprocessing would be to convert all words to their stems. This will transform words like "amazingly" to "amaze". Doing this makes all words with the same root or stem equivalent, this way when we create a feature set these words will all be treated as the same. This can be done in Python using external libraries like this:
```Python
from nltk.stem import PorterStemmer
port_stem = PorterStemmer()

for word in text:
    word = port_stem.stem(word)
```

There are numerous more preprocessing techniques that can be used to tweak accuracy. These were omitted from my finalized model due to training taking upwards of four hours. There was simply not enough time to see which combination of these techniques would be most optimal.

One of these steps is to remove stop words. A stop word is a word similar to "the", "a", or "of"; these words occur very frequently and carry little meaning. Since our feature set will be a dictionary containing a count of the k most frequent words, we would like to avoid these words cluttering our feature set with words that carry little meaning. We can accomplish this with the following code:

```Python
from ntlk.corpus import stopwords
words = 'He is the best of all of them'
new_words = [word for word in words if word not in stopwords.words('english')]
>>> new_words
['He', 'best']
```
Since our feature set was larger this did not make much of a difference for us. 

Another preprocessing step would be to remove symbols. A simple Python regular expression for this is:

```Python
import re
my_string = 'hello&% ho&^&w ar*&e y%&o*u'
my_string = re.sub('[^a-zA-Z0-9 \n\.]', '', my_string)
>>> my_string
'hello how are you'
```

## Exploration of ML Approach: Building Features
Now that we have processed this data, we must determine how we can train a machine-learning algorithm to perform classifications. We can’t just import a bunch of text into a random forest algorithm and expect to get very far. This is where feature sets come in. 

For our word set we want to find the frequencies of all of the words in the dataset after these words are preprocessed. We then want to take the k most frequent words where k is the feature set size in which we specify. This can be done using nltk as shown below:

``` Python
import nltk
size_of_feature_set = 1000

frequency = nltk.FreqDist()
for word in words:
    frequency[word] += 1

feature_words = list(frequency_dist)[:size_of_feature_set]
```
Now that we have the word features of the dataset, we can use these to build the feature sets for each row in our dataset. These feature sets are dictionaries with the word as the key and a Boolean representing whether or not the word is in the row for the value. We do this for all of the words contained in our word features. After we do this for every row in our dataset, we can now train a forest classifier on these feature sets.

## Exploration of ML: Random Forests in a Nutshell

Let’s say we have an image with just a solid color. This color is a bluish color that looks like it might also be purple. We want to know what this color is so we post a poll online so people can vote on the color. Most likely the answer with the most votes will be correct. This is an odd example, but we do see this all of the time online. On Chegg the correct answers tend to get more upvotes. On YouTube, videos that convey messages that people generally agree with typically receive more likes than dislikes. This method of making a decision on what is “correct” is called an ensemble. 
 
Why are we talking about colors? Well, a random forest works in a similar way. A random forest is made up of many decision trees. The decision trees in a random forest essentially vote on the classification. The answer with the most votes is chosen for the classification.

These decision trees are essentially comprised of a series of questions. Each node in the tree represents a question with each of the pointers to the node’s children being the answers to the question. Below is a very simplified example of this where the tree is representing the decision of which clothing item to wear outside.

```
Is it cold outside?
      / \
     /   \
yes /     \ no
   /       \
  /         \
Jacket    T-shirt
```

This idea is scalable and can be used for more complex classifications.

## Algorithm Approach: Naive
After training a model with the Sklearn random forest classifier, I decided to come up with my own simple algorithmic approach to this problem. 

This repository contains separate files of positive and negative words: https://github.com/shekhargulati/sentiment-analysis-python/tree/master/opinion-lexicon-English

I put the positive and negative words from the repository above into separate Python sets. I also converted the text that I wanted to classify to a set. I then took the intersection between the text and the negative words, and the text and the positive words. For the neutral words, I merely took the percentage of words that were not in the intersection of either set. Here is this idea in Python: 

```Python
negative_score = len(negativewords & text_set) / len(text_set) * 100
positive_score = len(positivewords & text_set) / len(text_set) * 100
neutral_score = (len(text_set) - (len(negativewords & text_set) + len(positivewords & text_set))) / len(text_set) * 100
```

Another variation of this approach that I experimented with comprised of taking the size of these intersected sets as opposed to using the percentages. Take the two below texts for example:
```
"I hate this movie with a burning passion. The screenplay was awful. The music was uninspiring. The acting was laughable. This movie was a waste of 2 hours and I would honestly rather watch grass growing for 10 hours straight than 30 seconds of this piece of utter garbage."
```


```
"I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it."
```

I believe the first one should have a higher negative score than the second one. The first one is verbose and really shows how much the author disliked the movie. The second one was low effort and showed little meaning. If we remove duplicate words by taking the set of both of these, the first text will have a larger set than the second one.
```Python
>>>len(set("I hate this movie with a burning passion. The screenplay was awful. The music was uninspiring. The acting was laughable. This movie was a waste of 2 hours and I would honestly rather watch grass growing for 10 hours straight than 30 seconds of this piece of utter garbage.".split()))
36
```

```Python
>>> len(set("I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it. I hated it.".split()))
3
```

This is a simple approach I came up with to compare with more complicated approaches.

Note: I did not preprocess the above examples. I merely used them as a simple demonstration of the algorithm.

## Algorithm Approach: Advanced (Vader)
Vader is a robust sentiment analysis library. This library works by having dictionaries corresponding with various word categories. Each word in these dictionaries has a multiplier which affects the overall sentiment of the text. The categories include negations, emojis, boosters (like "absolutely" and "very"), idioms, and special cases. 

Here is an example of this library in action:

```Python
>>> from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
>>> analyzer = SentimentIntensityAnalyzer()
>>> analyzer.polarity_scores("The movie was amazing")

{'neg': 0.0, 'neu': 0.441, 'pos': 0.559, 'compound': 0.5859}

>>> analyzer.polarity_scores("The movie was terrible")

{'neg': 0.508, 'neu': 0.492, 'pos': 0.0, 'compound': -0.4767}
```
The "compound" values in these scores are the overall sentiment scores with the sign representing the positivity (+) or negativity (-) of the text.

## Comparing Both Approaches
To compare these approaches let’s consider the following sentences:
```
1. That movie was good.
```
and

```
2. That was not good.
```
These sentences are very similar (at least for a computer) yet have opposite meanings. A machine learning model has to learn the impact that the word “not” can make on a sentence. Let’s see if ours did:

```Python
>>> c.classify_new_sentence("That was good")
positive
>>> c.classify_new_sentence("That was not good")
negative
```
Our machine learning model was able to correctly classify these two examples, but sometimes the model can be inaccurate.

These types of classifications are more accurate using an algorithmic approach such as Vader. Let’s take the following word:
```
good
```
This will yield a positive sentiment using Vader. Now, let’s add a negation to this word:

```
not good
```
This, as expected, yields a negative sentiment. Now how about a negation of this negation:

```
not not good
```
This yields a positive negation because the two negations cancel each other out. In general, the outcome is negative when there is an odd number of negations, and positive when there is an even number of negations. This can be represented by:
```
<negation>^n <word> where positive if n % 2 = 0 else negative 
```

It would be unlikely for our machine learning model to learn this particular pattern. 

## Conclusion
After all of these concepts are applied, it is simple to connect our classifier to the Reddit API to experiment with. In this case, let’s classify comments on a Computer Science subreddit.

Here are the results:
```
('positive', {'positive': '5.9%', 'neutral': '91.2%', 'negative': '2.9%'}, "Something I try to keep in mind is to never compare my chapter <insert #> to someone else's chapter <insert #>....often times in the beginning we tend to compare ourselves to someone who's been studying and practicing these concepts for years, then get discouraged when we fail and watch them succeed. We all fail, it's the best teacher.")

('negative', {'positive': '10.3%', 'neutral': '77.6%', 'negative': '12.1%'}, 'It’s really difficult to see people not seem to struggle while every finals season I feel like I’m going to fail every class and I seriously consider dropping out. \n\n\nMath/CS/physics seems to have a serious fetish for genius, which makes failing, not doing well, struggling etc etc all the more difficult. I have had so many classmates who don’t need to study, don’t need to take notes, can skip classes and get A+’s. And that’s ok, good for them. But it does make it hard to not be a genius. \n\n\nIt is important to talk about how the degree can make you cry. I’ve been there, and will be there a few more years. \n\n\nThank you for the video, and stay strong!')

('negative', {'positive': '3.6%', 'neutral': '95.5%', 'negative': '0.9%'}, "I had a data structures project due the same week as 4 midterms (midterms on M/T, project due W, midterms Th/F). I submitted something that basically didn't work and cried for 30 minutes afterwards. Got a 38/80 on it and for the rest of that semester and even after it, my confidence was shot. My friends would tell me to start personal projects for my resume, but all that I could think about was how I didn't know enough to start one -- I even considered not majoring in CS anymore. What really brought my confidence back was getting coding challenges and eventually interviews for internships, and after getting my first offer in October (which I decided to take), I realized that if I could somehow land an internship with two shitty half-done Android apps listed on my resume, then I wasn't as stupid as I thought I was. \n\nThis past semester I've been getting random bursts of inspiration here and there that compel me to open up my IDE and just start coding (I made my boyfriend wait half an hour to take me out because I had an idea for a Discord bot, lol) and I've been embracing Google for learning things I don't know and even showing me code examples that I can work off of! I hope these random motivation bursts continue through the summer so I can have something substantial on my resume when I start interviewing for next summer :D")

('negative', {'positive': '5.0%', 'neutral': '90.0%', 'negative': '5.0%'}, 'I really related to this video! I think only my family knows the amount of times my CS major has made me cry. None of my classmates/friends in CS talk about this and I feel like they are all in a competition where I’m just somewhere far away pretending to get things. I’m learning that it’s ok to fail and try again and staying away from the people who ONLY  brag about their accomplishments and don’t show me their human side.')

('positive', {'positive': '8.9%', 'neutral': '84.4%', 'negative': '6.7%'}, 'Stay strong friend!  I have failed many many many times! I am not smart but I have the willingness to succeed and I believe that is what matters most in college.  Now I am six classes away from graduating.\n\nHere are a few examples of my failures.\n\nFailed classes\n\nFailed an internship interview multiple times \n\nFailed project developments.\n\nWhat I did to overcome these failures:  by quoting batman begins lol "why do we fall Bruce?  So we can learn how to pick ourselves back up"')

('positive', {'positive': '0.0%', 'neutral': '91.7%', 'negative': '8.3%'}, "There's a reason CS programs are difficult to get into. They're even more difficult to complete.")

('negative', {'positive': '4.7%', 'neutral': '93.0%', 'negative': '2.3%'}, 'Oh, this is a great topic to discuss. My school went pass/fail for this semester AND waived grade requirements for majors (ie you can pass with any grade except an F), but I’m still struggling with my classes. I’m sitting at a 45% in Discrete Mathematics, and there’s basically no hope that I’m going to be able to pull that up. I’ve been beating myself up about it.\n\nI’m at a public school and the majority of my fellow students just seem so amazing—they’re ahead on projects, watching lectures on time, doing all their homework, etc. One of my friends turns in our homeworks the day they’re assigned; he spends like 3 hours, tops, where the same assignment typically takes me 5+ days. \n\nI’m only a sophomore, but I’ve been debating on if this is the major for me. But I’m already almost 30, and I don’t have the time or money to change my major all over again. \n\nIt’s depressing, but I just keep telling myself that everything will be okay if I work just a little harder.')

('positive', {'positive': '12.5%', 'neutral': '85.0%', 'negative': '2.5%'}, 'Wow!! As someone who is majoring in CS at a mid tier university, I always thought that the quality of teaching would be better at top universities and hence it would be easier. But now I know everyone struggles.\n\nPersonally, after my freshmen year, I just decided to try my best and not care about the grades. While I do strive to do the best I can in exams, I’ve come to the realization that those exams DEFINITELY don’t judge my skill. \n\nOne important thing that many people, especially the college education system needs to realize is that not everyone is capable at learning at the extremely quick pace that universities operate at. As a result, those who can’t make those connections in time struggle a lot. Until the education system changes, these struggles will continue to be there. \n\nThank you for sharing your struggles, and good luck!! You can definitely graduate and do well in life!!👍\n\nP.S. you’ve earned yourself a subscriber😁')

('negative', {'positive': '0.0%', 'neutral': '93.8%', 'negative': '6.2%'}, 'I failed so many classes in college. When I got a job offer, I took it. Fuck college never looking back')

('positive', {'positive': '7.4%', 'neutral': '88.9%', 'negative': '3.7%'}, 'Thank you for making this! I struggled through the major and when I didn’t pass a class, a friend at Stanford reached out and admitted she hadn’t passed a few cs classes. I was so surprised because I thought she seemed so perfect—it was very humanizing and helped inspire me to keep pursuing the major.')

('positive', {'positive': '10.3%', 'neutral': '89.7%', 'negative': '0.0%'}, "Something I learned through my Junior year was to not compare my success to other's. Everyone is going through school at a different pace and that's okay. That doesn't mean you can't see others success as motivation.\n\n\nAlso, life is sometimes not fair even if you do everything right.")

('negative', {'positive': '0.0%', 'neutral': '66.7%', 'negative': '33.3%'}, 'fuck jake wtf')

('positive', {'positive': '11.7%', 'neutral': '77.9%', 'negative': '10.4%'}, 'Do you think that this has anything to do with the psychology of social media in our generation (carefully curated photos and stories, making people look "perfect" without flaws)?\n\nI think some college environments just don\'t address failure in a positive way. It\'s like social media, where you only hear about people\'s accomplishments and not their failed projects or lost internships. But we lose out on a lot of opportunity when we don\'t talk about failure. Failure is essential because it means we are pushing ourselves, but it\'s crucial that we are supporting each other and giving encouragement when it happens. Smart people fail and keep failing until they start to win, because they are determined to accomplish a goal. Thus, being open about our failures (and how we made them) helps everyone make better decisions and get advice from other people who have made similar failures or overcame them.')

('positive', {'positive': '7.8%', 'neutral': '90.2%', 'negative': '2.0%'}, "Looks like everyone has that one phone call at least once in their CS journey. I'm not gonna pretend to be some macho dude, but I cried to my Dad after bombing my first CS exam. My parents never went to college while other kids came from the Bay and were programming since they were 8, it did not feel good to know that I was gonna be curved against these people lol. It can seem like you'll never catch up to some people, but remember that college is a super artificial environment and is not always a mirror that represents you accurately.")

('positive', {'positive': '15.8%', 'neutral': '84.2%', 'negative': '0.0%'}, 'Wow, huge props. In an age of social media where everyone posts only the very best versions of themselves, it was incredible to watch such a real video like this. Thank you.')

('positive', {'positive': '2.9%', 'neutral': '88.6%', 'negative': '8.6%'}, "I haven't watched the video yet because I know I'm gonna resonate with it and become a sad sponge  \nBut hell, it's even harder being a female in CS class because you feel like you need to prove yourself even more than others. I hope you get the hang of it soon, I'll be here trying too.\n\nThanks m8")

('positive', {'positive': '12.5%', 'neutral': '87.5%', 'negative': '0.0%'}, "I've always wondered how much harder the CS program is at a top school like Standford vs. my no name state school.")

('positive', {'positive': '0.0%', 'neutral': '88.9%', 'negative': '11.1%'}, 'Imagine all of the same stress but WITHOUT the grade inflation. \n     - the UC system')

('negative', {'positive': '6.2%', 'neutral': '93.8%', 'negative': '0.0%'}, 'I have two questions for you??\n\n1) What are you after??  What career do you want that will be the prize when you graduate \n\n2) Did/ are you passing the class or classes.')

('positive', {'positive': '12.0%', 'neutral': '82.0%', 'negative': '6.0%'}, 'I understand the feeling, but I always try to look forward even when it sucks. Yes, comparing yourself to others whom are better isn’t often ideal, but I don’t think it can be helped. It’s a competitive field, but you should definitely keep going forward. I always compare myself to my friends and my cousin who works at fb. It makes me feel so discouraged and want to completely drop out. I try to look at it as motivation as a goal that I’ll soon reach. It’s really difficult mentally to keep up, so once in a while give yourself a break. Tell yourself you’re doing great and remember the only person you should be competing with is yourself.')

('negative', {'positive': '0.0%', 'neutral': '100.0%', 'negative': '0.0%'}, 'Discrete Math is currently eating my lunch. I may have to take it again. If I do it’ll be my only class so far that I’ll have to do that.')

('positive', {'positive': '0.0%', 'neutral': '92.3%', 'negative': '7.7%'}, 'Out of curiosity, which class did you fail the midterm in? Also, how much CS experience did you have going into college?')

('positive', {'positive': '16.7%', 'neutral': '81.0%', 'negative': '2.4%'}, "I had the exam same phone call you did in this video during my first semester... It honestly hit me really hard as I said almost the exact same things you did about being focused and still coming up short. I'm glad you found a way to finish the week on a positive note! Times can get tough but hard work and perseverance will always lead you down the right path. Thanks for sharing!")

('positive', {'positive': '0.0%', 'neutral': '81.2%', 'negative': '18.8%'}, '"It is possible to commit no mistakes and still lose. That is not weakness, that is life"\n\n-Jean Luc Picard')

('negative', {'positive': '0.0%', 'neutral': '100.0%', 'negative': '0.0%'}, 'Wow that was relatable. Subbed.')

('positive', {'positive': '0.0%', 'neutral': '92.1%', 'negative': '7.9%'}, 'I’ve never touched CS until this past summer. I was in medicine prior to the bachelors in CS that I’m now pursuing. CS is hard, but it’s not near as difficult, soul sucking, or a drain of both my time and social life as medical school was. I’m only taking 3 classes each semester while working part-time though, so that probably has a bit to do with it. If I was taking 16-17 credits I probably couldn’t handle them all being CS or math.')

('positive', {'positive': '15.0%', 'neutral': '75.0%', 'negative': '10.0%'}, 'One thing that’s been helping me get through the hard times and regain my focus is that the things that are worth doing are often difficult. Hard work pays off. Loved the video!')

('positive', {'positive': '15.0%', 'neutral': '80.0%', 'negative': '5.0%'}, 'Hmmm... haven’t heard of duck syndrome before, but it describes me pretty well honestly.\n\nGreat video, hope everything works out! :)')

('negative', {'positive': '15.0%', 'neutral': '76.2%', 'negative': '8.8%'}, 'Hey man, this shit is hard. People don’t talk about negative results as much because it’s way more exciting to talk about your accomplishments. \n\n My undergrad experience was a lot of struggling, but at some point I kind of stopped caring about results and grades, and started to actually enjoy what I was doing. I think you can do really well if you find intrinsic motivation, and the field is broad enough that I’m sure most people can find something they love, or at least really vibe with. It doesn’t have to be hardcore theory, I mean you might be really good at design or UI/UX. If you are, run with it!\n\nI hate to see people be unhappy because they feel like they’re stuck doing something they don’t want to do. \n\nI had this inflection point when I took compilers and rendering, where I realized that the work I was doing was actually cool, and I didn’t care what grade I’d end up with in the class because just being able to learn the material and say “I know that” was enough to leave me feeling satisfied and accomplished.')

('positive', {'positive': '10.0%', 'neutral': '90.0%', 'negative': '0.0%'}, 'Damn this was a good video , did u end up passing the final? If u did can u share what you did differently?')

('positive', {'positive': '4.3%', 'neutral': '85.5%', 'negative': '10.1%'}, "I read the description to your video. Seems like duck syndrome is the norm at Stanford (matches up to a comment I saw about Stanford under MIT Confessions on FB). You need to act like your chill, even when you're struggling to stay afloat. At my uni, everyone complains, but somehow you need to be able to prosper. CS is difficult, no shit, I'm technically an ECE major, it's also heccin hard. I have fucked up many a prelim during a few of my spring semesters at my uni. Had one semester with 3 C's on my transcript.  But it's okay. We all reach our goals at different paces, but there will always be a way. Good luck!")

('positive', {'positive': '12.5%', 'neutral': '87.5%', 'negative': '0.0%'}, "Great video. I'm a rower as well btw! I enjoyed your vlog on Stanford Lightweight Rowing.\n\n #LightweightsRiseUp")

('negative', {'positive': '7.0%', 'neutral': '84.5%', 'negative': '8.5%'}, "A junior from NoName university here, I feel like Its important in CS to keep in mind that FANG internship/good job/winning some competition like ICPC all are extremely overrated, there's so much we can do yet everyone just seems to categorize themselves into one of these pots and start working towards it, and eventually as supply > demand, many fail and get depressed, in CS the community has created so much artificial pressure it's becoming toxic day by day, especially where I'm from, right from the start everyone is just bothered about packages/compensations/number of internships, etc, etc. No one talks about relaxing and looking back at what they have done far and what they can do with all the time they have ahead. CS is a Race with no spectators and only sadness as the winning prize, except for a few expectations of course.")

('positive', {'positive': '3.8%', 'neutral': '88.5%', 'negative': '7.7%'}, 'Yeah, I feel that. I was coding years before college, and my degree plan is still kicking my ass. Biggest pain in my ass right now is being required to learn x86 Intel Assembly. My professor doesn’t know how to teach online (which we’ve been forced into), and all the content feels tedious, especially when we’re forced to just recreate things we could do in C, but worse. It’s hard to find outside resources for learning it, so we’ll see how that goes. Good luck to all the other CS majors out here right now lol.')

('positive', {'positive': '6.9%', 'neutral': '86.2%', 'negative': '6.9%'}, "tl;dr Read bolded sentences. \n\nNot sure what year of school you're in, I think you mentioned the failed midterm was your first midterm but maybe I misinterpreted that. Anyways, you're right; CS is a difficult major but don't ever think that you need to switch majors. **In my opinion, if CS is the degree you want more than any other degree, switching would be giving up. Never give up.** \n\n**Failing exams happens to all of us. You can fail midterms and hell even finals and still get A's in the classes in some cases.** I'm pretty sure most schools have curved grading scales for CS and probably other difficult majors as well. That's probably why kids still have 4.0s lol. \n\nAlso, tests and grading for CS theory courses has been a terrible experience for me. I've answered test questions with textbook definitions verbatim and had those questions marked as incorrect even though they were correct. The case for these courses is that the professors have strict grading rubrics and TAs don't want to give you points if the answer doesn't exactly match the rubric (even if it's right) because they don't want to have to explain why they gave points out to an answer that wasn't the exact rubric answer. **My point is this: your grades don't always reflect your ability or what you know.**\n\n**My advice for dealing with kids like Jake is don't.** Consider them as gone and prepare to pick up those student's responsibilities with the help of your contributing team members. This avoids conflict (which I hate), and gives you that much more experience. Also, you can be sure that your project will be how you and your contributors want it to be because you won't have to worry about crappy code from the slacker in the group. The kid will benefit with the grade, but in my opinion, **you have to ask yourself if you'd rather deal with the conflict and force a crappy teammate to work with you and give you crappy results OR get the grade you want and deserve.** \n\n**I appreciate your video.** CS is a quietly competitive field so it's nice to see a post that isn't a braggadocious success story.")

('positive', {'positive': '9.7%', 'neutral': '82.3%', 'negative': '8.1%'}, 'One thing you need to realize about all STEM subjects is that they are tough not because you are dumb, it is because you need to change your way of thinking to more abstract principles which more often than not conflict with your perception of what makes sense to your unconditioned mind.\n\nI remember my first months with programming, it was an absolute nightmare - I understood practically nothing. I even recall struggling with the notion of loops! But since then I got my CS degree which wasn’t easy and have worked for, and still am working for, some of the best tech companies as a developer.\n\nCS demands sustained determination, and when you persist long enough you realize that it’s not so bad after all. Hang in there, it’s worth it!')

('negative', {'positive': '0.0%', 'neutral': '100.0%', 'negative': '0.0%'}, 'I’ve had to withdraw from linear algebra twice. I have no idea how I’m ever going to pass that darn class ;-;')

('positive', {'positive': '10.0%', 'neutral': '90.0%', 'negative': '0.0%'}, 'One think I want to bring out is that the more early you have had exposure to CS, the easier the classes are for you. Just think about how much easier your first sem classes would be if you had taken them as a sophomore level student. Those brilliant students are not smarter than you. They just have more experience and have spent a lot more time coding and doing math than you.')

('positive', {'positive': '10.2%', 'neutral': '81.6%', 'negative': '8.2%'}, "Thank you so much. College is hard. It's really fucking hard. I'm a sophomore right now studying computer science and Statistics. School along with jobs and money and social problems and trying to find time to do the things you love doing is So.Fucking.Hard. So thank you for putting this out there, because it's important. We'll get through this just fine. Don't worry :) Alright now I gotta finish an algorithms project lol.")

('positive', {'positive': '1.6%', 'neutral': '95.1%', 'negative': '3.3%'}, "I'm currently at City College in San Francisco finishing up my last CS courses (before I transfer to a 4-year) while working full-time as an IT Engineer at the moment and I gotta say I totally resonate with what you mention in your video. A few months ago, I encouraged myself to be more social and vocal about these matters with my classmates in these classes because I'm sure as hell we're not the only ones feeling this type of way. I've met students who are in your shoes and my shoes but we're both here because of our drive. We've busted our asses to get to this point in our lives and it's ok if our best is not the best. Success does not have an expiration date, failure does.")

('negative', {'positive': '1.9%', 'neutral': '90.7%', 'negative': '7.4%'}, 'Oh man. @OP I am aware. \n\nA friend of mine went there. I didn’t realize that people would jump from the bell tower.\n\nStanford actually, really, truly does have a problem with expectation management and the “what else / what now” questions that can come up. \n\nDouble-whammy: make it through with a PhD from there and THEN face the also-unexpected crisis of “AAAAGGGGggghhhh ... ahhh ........... ... okay, I’m done. Now what?” Suddenly having your all-consuming work end, even if it is because you’ve passed defense, is a bewildering and strange thing to deal with.')

('negative', {'positive': '2.3%', 'neutral': '86.0%', 'negative': '11.6%'}, 'In my CS program, in nearly all core classes, the norm was failure.  Tests were designed to be so difficult that most of the class failed without a curve.  It made the geniuses really stand out, which is I guess why they did it?  It was absolutely brutal and took a long time to not feel like an absolute failure.  The only way we could tell if we were actually failing the class would be comparing grades with classmates.')

('positive', {'positive': '0.0%', 'neutral': '100.0%', 'negative': '0.0%'}, "I couldn't get past differentials and so I switched majors from cs to something else.  Math kicked my butt.")

('positive', {'positive': '6.5%', 'neutral': '83.9%', 'negative': '9.7%'}, 'University physics made me cry. Not symbolically; literally it made me cry. There have been other times where I felt defeated, upset when I couldn’t do a programming assignment, exhausted etc. \n\nWe just need to keep pushing through. CS is not an easy degree. We got this :)')

('negative', {'positive': '3.0%', 'neutral': '95.0%', 'negative': '2.0%'}, "I mean why don't you just abuse TA hours? Lots of your friends probably do XD. Kids straight up turning up at hours without having even read the handout LOL. EDIT Ok you went to them in the video nice!\n\nEDIT: Watched the video saw it was about a midterm. \n\nHeres what I do to have been doing to be doing well above average for every CS class I have taken at Stanford for the past 4 years.\n\n1) Go to lectures (or watch them live in the current situation)\n\n2) Take notes on my iPad.\n\n3) That night write notes from all the days lectures aesthetically.\n\n4) Day after read aesthetic notes again.\n\n5) Rinse and repeat but also reread notes from previous lectures and highlight shit before step 3.\n\nSpaced Repetition. Spaced Repetition. Spaced Repetition.\n\nBonus: Start any assignment day it was given out.\n\nSource: Senior at Stanford.\n\nRegarding group projects in the case where nobody other than me has any experience. I basically carry them myself and assign subtasks relative to the aptitude of each of my other group members so everybody feels like they are contributing. As you have experienced even at Stanford there are people who don't try.")

('positive', {'positive': '4.8%', 'neutral': '85.7%', 'negative': '9.5%'}, "I study in a fucking unranked university(due to circumstances), and I always solve MIT and Stanford exams it's not that hard (keep in mind my education is not that great)")

('negative', {'positive': '3.4%', 'neutral': '86.2%', 'negative': '10.3%'}, 'Nobody talks about failure as most people here assume people will be successful in their job hunt, not even considering people failing at actually acquiring their degree. This is implicit in the subreddits name. It assumes that you are currently a cs major. Typically if a cs major fails they switch majors.')

('positive', {'positive': '0.0%', 'neutral': '81.8%', 'negative': '18.2%'}, 'I always hate when ppl cry on camera. Like u need attention or what?')

('positive', {'positive': '0.0%', 'neutral': '83.3%', 'negative': '16.7%'}, 'If you think college is hard, boy do I have a surprised for you. Life is fucking harder')

('negative', {'positive': '0.0%', 'neutral': '100.0%', 'negative': '0.0%'}, 'Nobody cares.')

('negative', {'positive': '0.0%', 'neutral': '100.0%', 'negative': '0.0%'}, 'AYYYY JAKES MY BOI')

```
As you can see these results are semi-accurate, though I am satisfied with them for my first machine learning project.

## Sources
Thank you to the following articles and tutorials that made this project possible:
* https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/
* https://edmundmartin.com/text-classification-with-python-nltk/
* https://towardsdatascience.com/understanding-random-forest-58381e0602d2
* https://github.com/shekhargulati/sentiment-analysis-python/tree/master/opinion-lexicon-English
* https://github.com/cjhutto/vaderSentiment
## Contact
[Reginald Thomas](mailto:reginaldcolethomas@gmail.com)

