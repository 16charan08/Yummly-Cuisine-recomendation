
## CS5293_Spring2020_Project3
The goal of the project is to create an application that take a list of ingredients from a user and attempts to predict the type of cuisine and similar meals
### Author :- Ram charan Reddy Kankanala
### Email :- Ramcharankankanala@gmail.com

### Flow
1) Loading JSON data into dataframe.
2) Normalizing and converting "ingredients" into vectors along with input.
3) Finding cosine similarity score with respect to input and getting top10 similar ID's.
4) Saving RandomforestClassifier using pickle.
5) Using saved model to predict cuisine of given input ingredients.


#### Text into Features
First, I converted my JSON file into a data frame that contains "ids","cuisine","ingredients". Then taking every ingredient from the list and joining with ",". After joining ingredients are passed into a normalization function where tokenization, removing stopwords, lemmatizing, etc will be done and joined to our original data frame for future purposes. Then using "TFIDFvectorizer" vectoring all rows of ingredients along with input added to get feature matrix. I am choosing TF-IDF vectorizer because it finds how frequent a word is repeated in a document and assigns a perfect weight to it and it does it for all features it identified with respect to each row(given ingredient).
