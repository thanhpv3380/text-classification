import pandas as pd
import numpy as np
import time
from nltk.tokenize import word_tokenize # tach tu
from nltk import pos_tag
from nltk.corpus import stopwords # remove stopword
from nltk.stem import WordNetLemmatizer #code, codes, ... => code
from sklearn.preprocessing import LabelEncoder # change pos - neg to 1 - 0
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer #vectorizer
from sklearn import model_selection 
from sklearn.naive_bayes import MultinomialNB # naive_bayes
from sklearn.svm import LinearSVC   #SVM
from sklearn.tree import DecisionTreeClassifier #decision tree algorithm
from sklearn.ensemble import RandomForestClassifier #random forest
from sklearn.neighbors import NearestCentroid #rocchio_classification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print('start ', time.asctime( time.localtime(time.time()) ))
np.random.seed(500)
Corpus = pd.read_csv("reviews.csv")

Corpus = Corpus.head(10000)

Corpus['review'] = [entry.lower() for entry in Corpus['review']]
Corpus['review']= [word_tokenize(entry) for entry in Corpus['review']]


tag_map = defaultdict(lambda : wn.NOUN)

tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(Corpus['review']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            #print(word,tag[0])
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            #print(word_Final)
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Corpus.loc[index,'text_final'] = str(Final_words)


# train 70 test 30
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['sentiment'],test_size=0.3)

print(Train_X)
#change pos-neg to 1-0
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
print(Train_Y)
# # use TF-IDF
# Tfidf_vect = TfidfVectorizer(max_features=5000)
# Tfidf_vect.fit(Corpus['text_final'])
# Train_X_Tfidf = Tfidf_vect.transform(Train_X)
# Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#Bag of words
count_vec = CountVectorizer()
count_train = count_vec.fit(Corpus['text_final'])
Train_X_Bow = count_vec.transform(Train_X)
Test_X_Bow = count_vec.transform(Test_X)


# fit the training dataset on the NB classifier
Naive = MultinomialNB()
Naive.fit(Train_X_Bow,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Bow)
# Use accuracy_score function to get the accuracy
print("Naive-Bayes: ")
print(classification_report(Test_Y,predictions_NB))


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
svm = LinearSVC()
svm.fit(Train_X_Bow,Train_Y)
# predict the labels on validation dataset
predictions_SVM = svm.predict(Test_X_Bow)
# Use accuracy_score function to get the accuracy
print("SVM: ")
print(classification_report(Test_Y,predictions_SVM))


# Classifier - Algorithm - Decision tree
# fit the training dataset on the classifier
DecisionTree = DecisionTreeClassifier(criterion='gini').fit(Train_X_Bow,Train_Y)
# predict the labels on validation dataset
predictions_DT = DecisionTree.predict(Test_X_Bow)
# Use accuracy_score function to get the accuracy
print("Decision Tree: ")
print(classification_report(Test_Y,predictions_DT))


#random forest
forest = RandomForestClassifier()
forest.fit(Train_X_Bow,Train_Y)
predictions_RF = forest.predict(Test_X_Bow)
print("Random Forest: ")
print(classification_report(Test_Y,predictions_RF))


#rocchio_classification
rocchio = NearestCentroid()
rocchio.fit(Train_X_Bow,Train_Y)
predictions_Rocchio = rocchio.predict(Test_X_Bow)
print("Rocchio_classification: ")
print(classification_report(Test_Y,predictions_Rocchio))

print('start ', time.asctime( time.localtime(time.time()) ))