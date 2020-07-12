#!/usr/bin/python

import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
import sys
import os
import pickle

def predict_proba(year,title,plot):

    clf = joblib.load(os.path.dirname(__file__) + '/model_logit.pkl')
    with open("vocabulary.txt", "rb") as fp:
        voc_ = pickle.load(fp)
    with open("th.txt", "rb") as fp:
        th = pickle.load(fp)


    data=pd.DataFrame({'year':[year],
        'title':[title],
        'plot':[plot]})

    data['title']=data['title'].str.lower()
    data['title']=data['title'].str.replace(',', '') 
    data['title']=data['title'].str.replace('.', '') 
    data['title']=data['title'].str.replace('-', '') 
    data['title']=data['title'].str.replace('_', '') 
    data['plot']=data['plot'].str.lower()
    data['plot']=data['plot'].str.replace(',', '') 
    data['plot']=data['plot'].str.replace('.', '') 
    data['plot']=data['plot'].str.replace('-', '') 
    data['plot']=data['plot'].str.replace('_', '')
    data['uno'] = data['title']+data['plot']
    data.drop(['title','plot'], axis=1, inplace=True)

    wordnet_lemmatizer = WordNetLemmatizer()
    def split_into_lemmas(text):
        text = text.lower()
        words = text.split()
        return [wordnet_lemmatizer.lemmatize(word, 's') for word in words]

    vect = TfidfVectorizer(analyzer=split_into_lemmas,
                            min_df=2,
                            max_features=25564,
                            sublinear_tf=True,
                            strip_accents='unicode',
                            ngram_range=(1,3),
                            stop_words='english',
                            vocabulary=voc_)
    X_dtm = vect.fit_transform(data['uno'], data['year'])


    cols = np.array(['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
        'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
        'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']).reshape((24,1))

    genres = np.transpose(clf.predict_proba(X_dtm))
    genres2 = np.copy(genres)
    for i in range(0,len(th)):
        #genres2[i] = int(genres2[i]>th[i])
        genres2[i] = int(genres2[i]>=0.5)

    data = np.append(cols, np.round(genres,2), axis=1)
    data = np.append(data, genres2, axis=1)
    res = pd.DataFrame(data, columns=['Genre','Prob','Val'])
    res = res[genres2 != 0]
    res.sort_values('Genre', ascending=True, inplace=True)
    res.drop(['Val','Prob'], axis=1, inplace=True)
    res.reset_index(drop=True, inplace=True)
    return res.values.tolist()


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add an URL')
        
    else:

        p1 = predict_proba(sys.argv[1],sys.argv[2],sys.argv[3])
        
        print(url)
        print('Movies Genres: ', p1)
        