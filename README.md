
## Yummly-Cusine-recomendation-system
The goal of the project is to create an application that take a list of ingredients from a user and attempts to predict the type of cuisine and similar meals
### Author :- Ram charan Reddy Kankanala
### Email :- Ramcharankankanala@gmail.com

### Flow
Input is passed as list of ingredients(changeable variable).
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

reference:- http://carrefax.com/new-blog/2017/7/4/cosine-similarity

#### def cosinesim(mat,df):
This function takes output from previous function(vectorized matrix) and normalized dataframe to calculate and append cosine similarity scores with respect to input list of ingredients.

Cosine similarity score calculation from sklearn package. 
> simi = cosine_similarity(mat[0:1], mat).flatten() {mat[0:1] is our input list} \
so with respect to input list cosine similarity scores are calculated and returned.

reference:- http://carrefax.com/new-blog/2017/7/4/cosine-similarity

#### def getcloseid(df,N):
This function takes our updated dataframe that contains scores and N(number of similar meals to input ingredients) to return topN similar meals.

This is done by sorting and returning ID and scores.
> final_df = df.sort_values(by=["score"],ascending=False) \
  result = final_df[["id","score"]].head(N)
  
#### def savemodel(df):
This function takes updated dataframe as input to save ran model that will be used for predicting cuisine.

*step1:-* Vectorizing of normalized ingredients column.
> tfidf_matrix = vectorizer.fit_transform(df["normalized_ing"]).todense()

*step2:-* Converting all cuisine values into numeric by using label encoding.
> cuisins = df["cuisine"] \
  lb = LabelEncoder() \
  Y = lb.fit_transform(df["cuisine"])
 
*step3:-* Training and Testing using RandomForestClassifier algorithm.
> X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)\
  clf = RandomForestClassifier()\
  clf.fit(X_train, y_train)\
  y_pred = clf.predict(X_test)
  
*step4:-* Calculationg accuracy.
>accuracy = accuracy_score(y_test,y_pred)*100  

*step5:-* Saving model using pickle.
> dump(clf, open('Rfcmodel.pkl', 'wb'))

#### def predictcuisine(input,model,df):
##### Assumptions made in this step:
1) For vectorizing and inverse labelencoding, original data is fitted and used.

This function takes input list, saved model , updated dataframe to predict cuisine for our input list of ingredients.

*step1:-* Input is normalized and it is transformed to vector.
> tfidf_matrix = vectorizer.fit_transform(df["normalized_ing"]).todense() \
>  ini = vectorizer.transform([joi[0]]).todense()

*step2:-* Inverse transform of label encoder is applied and cuisine will be returned.
>lb.inverse_transform(y_predin)

Thus finally we will be having both top 40 meals along with cuisine of input list of ingredients.

### Execution and output
I have attached a python notebook which takes a list of ingredients as  input to suggest top 40 meals and predict cuisine.

Execution of functions step wise.
>input= ["coriander powder","ground turmeric","red pepper flakes","japanese eggplants","plums","grated parmesan cheese","fresh parsley","tomatoes with juice"] \
df_ini = pd.read_json('yummly.json') \
norm_df = normalizedf(df_ini) \
mat = tfidfvecinput(input,norm_df) \
updated_df = cosinesim(mat,norm_df) \
N = 40 \
if not os.path.exists('Rfcmodel.pkl'): \
  accu = savemodel(updated_df) \
else: \
  print("model is already executed with",accu) \
model = load(open('Rfcmodel.pkl', 'rb')) \
print("Top10 ID's for given input") \
print(getcloseid(updated_df,N)) \
print("Cuisine for given input",predictcuisine(input,model,df_ini)) \

#### Output(Id's of most similar meals along with cosine similarity are returned): - 
model is already executed with 75.14770584538026 \
Top10 ID's for given input\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      id &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     score \
26586  39414  0.527524 \
25495  15840  0.507473 \
25454  22654  0.492945 \
34394  36213  0.484154 \
26117  24176  0.461936 \
14124  14472  0.460664 \
25018   1500  0.459376\
3214   40638  0.446175 \
30326  17469  0.433155 \
19777  46787  0.429784 \
13985  37784  0.420847 \
21633  14661  0.419719 \ 
32624  22292  0.418027 \
35776  26639  0.414957 \
9290    1131  0.411977 \
34723  35183  0.410843 \
15372  20021  0.407432 \
37774  18966  0.404435\
9351   35415  0.401782 \
32712  35985  0.399665 \
39704   1923  0.399622 \
38668  43074  0.399397 \
38808  12592  0.397190\
1444   20429  0.396928\ 
19909  18122  0.394279 \
5907   48958  0.392626 \
9479     740  0.391821 \
23027   6974  0.391534 \
39691  33890  0.391122 \
2022   40370  0.389360 \
9169   20096  0.388881 \
36142  11114  0.386660 \
16720  29324  0.381516 \ 
14816  37172  0.378523 \
32961  20125  0.377804 \
13     41995  0.376809 \
38885  21193  0.376413 \
32890  19138  0.376034 \
37048  14833  0.373698 \
28567  30675  0.373659 \
Cuisine for given input ['italian']
