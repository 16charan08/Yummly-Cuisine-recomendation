
## CS5293_Spring2020_Project3
The goal of the project is to create an application that take a list of ingredients from a user and attempts to predict the type of cuisine and similar meals
### Author :- Ram charan Reddy Kankanala
### Email :- Ramcharankankanala@gmail.com

### Flow
Input is passed as list of ingredients.
1) Loading JSON data into dataframe.
2) Normalizing and converting "ingredients" into vectors along with input.
3) Finding cosine similarity score with respect to input and getting top10 similar ID's.
4) Saving RandomforestClassifier using pickle.
5) Using saved model to predict cuisine of given input ingredients.


#### Text into Features:-
First, I converted my JSON file into a data frame that contains "ids","cuisine","ingredients". Then taking every ingredient from the list and joining with ",". After joining ingredients are passed into a normalization function where tokenization, removing stopwords, lemmatizing, etc will be done and joined to our original data frame for future purposes. Then using "TF-IDF vectorizer" vectoring all rows of ingredients along with input added to get feature matrix. I am choosing TF-IDF vectorizer because it finds how frequent a word is repeated in a document and assigns a perfect weight to it and it does for all features and with respect to each row(given ingredient).

#### Findind topN meals:-
For finding topN foods I am using "cosine similarity", I used cosine similarity because it gives similarity score based on how close features are in vector space after they are converted into vectors(weights based on their frequency). So we will get a perfect score with respect to the input given and we can sort those scores to get topN similar foods and their ID's.

#### Prediction of cuisine:-
For predicting cuisine based on input ingredient I am using "RandomForestClassifier" which predicts our cuisine with almost 75% accuracy.
For sending data into the model it needed to be converted into vector form (both input and given JSON data) and the model is built and saved using pickle. This saved model is used to predict cuisine, As it gives output by 75% accuracy cuisine may not be accurate.

#### N value:-
As, N value can be any value I choose it to be 0.001 percent of overall foods available which is nearly 40. So, user in the end will get top 40 similar meals based upon cosine similarity scores. 

### Functions and implementation:-
#### Packages used:-
import gensim as gensim \
import numpy as np  \
import os\
import pandas as pd  \
import json \
import gc \
import re \
import nltk \
import numpy as np \
nltk.download('punkt') \
nltk.download('stopwords') \
nltk.download('wordnet') \
from sklearn.ensemble import RandomForestClassifier \
from sklearn.metrics import accuracy_score \
from sklearn.model_selection import train_test_split \
from sklearn.preprocessing import LabelEncoder \
stop_words = nltk.corpus.stopwords.words('english') \
from nltk.stem.wordnet import WordNetLemmatizer \
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer \
from sklearn.metrics.pairwise import cosine_similarity \
from nltk import flatten \
from pickle import load \
from pickle import dump 

#### Python indentation is not followed in below code samples.

#### def normalize_document(txt):
##### Assumptions made in this step:
1) As every step is from nltk tokenization, lemmatization and stemming may be that appropriate.

*step1:-* Words are tokenized. 
> tokens = nltk.word_tokenize(txt)

*step2:-* Removing english stop words and word tokenization.
> clean_tokens = [t for t in tokens if t not in stop_words] 
    
*step3:-* Lemmatizing of words. 
>  wordnet_lem = [WordNetLemmatizer().lemmatize(w) for w in clean_tokens]

*step4:-* Snowball stemmer. 
> stems = [nltk.stem.SnowballStemmer('english').stem(w) for w in wordnet_lem]

*step5:-* Every step is done one after other finally will be joining while returning.
 > return ' '.join(wordnet_lem)


 #### def normalizedf(df_ini):
 This function simply takes our initial dataframe to normalize ingredients and return a new dataframe with normalized data attached. 
 
 Normalizing each row of ingredients.
 >for i in df_ini['ingredients']: \
      i = ' '.join(i) \
      x.append(normalize_document(i))

 
#### def tfidfvecinput(input,df):
This function takes normalized dataframe along with our input list and returns tdidf vector which contains input vector.

*step1:-* First input list is also normalized accordingly how our data is normalized.
> input_normalized.append(normalize_document(joined_input[0]))

*step2:-* Next our normalized input is inserted at first position of normalized ingredients data.
> y.insert(0,input_normalized[0])

*step3:-* Along with this inserted input tfidf vectorizer is calculated and that is returned.
>  tfidf_matrix_train = vectorizer.fit_transform(y).todense()

