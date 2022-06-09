from __future__                         import print_function
from lib2to3.pytree import convert
import os
from matplotlib.pyplot                  import tight_layout
import pandas as pd
import numpy as np
import tweepy
import csv
import string
import re
import pickle

from sklearn.model_selection            import train_test_split
from sklearn.feature_extraction.text    import CountVectorizer
from sklearn.naive_bayes                import MultinomialNB
from sklearn.linear_model               import LogisticRegression
from sklearn                            import metrics

from nltk.corpus                        import stopwords

from django.views                       import generic, View
from django.shortcuts                   import render, redirect

from .models                            import KPKTweet,KPKTweetTemp

# Create your views here.

class InsertNewView(View):

    def post(self, request, *args, **kwargs):
        classifier = TwitterSA()
        classifier.insert_dbkpk()

        return redirect('kpk:index')

class GraphView(generic.TemplateView):
    template_name       = 'chart.html'
    data                = KPKTweet.objects.all()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["qs"]       = self.data
        context["netral"]   = len([item for item in self.data if item.label == "netral"])
        context["positif"]  = len([item for item in self.data if item.label == "positif"])
        context["negatif"]  = len([item for item in self.data if item.label == "negatif"])
        return context

class TweetListView(generic.ListView):
    model               = KPKTweet
    template_name       = 'index.html' 
    context_object_name = "tweet_list"
    paginate_by         = 100 
    ordering = ['id']

    def get_queryset(self):
        old = KPKTweet.objects.all()
        new = KPKTweetTemp.objects.all()
        qs = new.union(old)
        return qs

# class TweetListView(generic.ListView):
#     model               = KPKTweetTemp
#     template_name       = 'index.html' 
#     context_object_name = "tweet_list2"
#     paginate_by         = 100 
#     ordering = ['-id']

class TwitterSA():

    data        = KPKTweetTemp.objects.all().union(KPKTweet.objects.all())
    module_dir  = os.path.dirname(__file__)  # get current directory

    def __init__(self) -> None:
        api_key             = 'uXdKtu3sZ8zicnzxgPuNcViPl'
        api_secret_key      = '2gbCTrCiXg1AphbKx9t9G9PkJlqoKN94CBl2pmxWYy3Yt1wKrQ'
        access_token        = '744811464041500672-Ho2l2iTyp6E3CW4fGesCvtgrz5sMVuD'
        access_token_secret = 'jeRs3muImQEK9MXRRGqYB0LIJCn6GtRMJ3uOh7JZJanGD'

        try:
            self.auth   = tweepy.OAuthHandler(api_key, api_secret_key)
            self.auth.set_access_token(access_token, access_token_secret)

            self.api    = tweepy.API(self.auth, wait_on_rate_limit=True)
            print('Connected')
        except:
            print("Authentication error")

    def insert_dbkpk(self, q="KPK exclude:retweets", lang="id", count=100):
        self.crawling(q=q, lang=lang, count=count)
        self.preprocessing()
        self.classification() # Update otomatis model_analisis
        self.predict()
        # path = os.path.join(self.module_dir, 'csv/datatest.csv') # Versi preprosesing
        path = os.path.join(self.module_dir, 'csv/dataresult.csv')
        df         = pd.read_csv(path)
        tweet = [
            """KPKTweet(
                tanggal=data.tanggal, 
                user_name=data.user_name, 
                isi=data.ISI, 
                stop_removal=data.STOP_REMOVAL
            ) for data in df.itertuples()"""
        ] # Versi preprosesing
        tweet = [
            KPKTweet(
                tanggal=data.tanggal, 
                user_name=data.user_name, 
                isi=data.isi, 
                stop_removal=data.stopword,
                label=data.label,
                polarity=data.polarity,
            ) for data in df.itertuples()
        ]
        KPKTweet.objects.bulk_create(tweet)

    def crawling(self, q="KPK exclude:retweets", lang="id", count="100"):
        path        = os.path.join(self.module_dir, 'csv/crawling.csv')
        csv_file     = open(path, 'w', encoding='utf-8')
        csv_writer   = csv.writer(csv_file)
        hasil_search = self.api.search_tweets(q=q, lang=lang, result_type="recent", count=count)
        csv_writer.writerow(["tanggal", "nama", "tweet"])

        for tweet in hasil_search:
            csv_writer.writerow([tweet.created_at, tweet.author.screen_name, tweet.text])
            """ tweet bersih = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet.text).split()) """
        
        print("Crawling completed")
    
    def preprocessing(self):
        path                = os.path.join(self.module_dir, 'csv/crawling.csv')
        df                  = pd.read_csv(path, encoding='unicode_escape')
        df['ISI']           = df['tweet'].apply(lambda x: self.remove_punct(x))
        df['TOKENIZATION']  = df['ISI'].apply(lambda x: self.tokenization(x.lower()))
        df['STOP_REMOVAL']  = df['TOKENIZATION'].apply(lambda x: self.remove_stopwords(x))
        df['isi']           = df['ISI'].str.lower()
        df['user_name']     = df['nama'].str.lower()
        df['STOP_REMOVAL']  = df['STOP_REMOVAL'].apply(lambda x: self.fit_stopwords(x))

        Final = df[['tanggal', 'user_name', 'ISI','STOP_REMOVAL']]
        Final = Final.drop_duplicates(['tanggal'])
        Final.to_csv(os.path.join(self.module_dir, 'csv/datatest.csv'))

    #Cleaning Text
    def remove_punct(self, text):
        text = re.sub(r'[^a-zA-Z0-9]', ' ', str(text))
        text = re.sub(r'\b\w{1,2}\b', '', text)  # menghilangkan 2 kata
        text = re.sub(r'\s\s+', ' ', text)
        return text

    #Tokenization
    def tokenization(self, text):
        text = re.split('\W+', text)
        return text

    def remove_stopwords(self, text):
        stopword = stopwords.words('indonesian')
        text = [word for word in text if word not in stopword]
        return text
    
    def fit_stopwords(self, text):
        text = np.array(text)
        text = ' '.join(text)
        return text

    def predict(self):
        model_analys  = os.path.join(self.module_dir, 'classification/model_analisis.pkl')
        count_vector  = os.path.join(self.module_dir, 'classification/count_vectorized.pkl')
        tfid_trans    = os.path.join(self.module_dir, 'classification/tfid_transform.pkl')
        data_test     = os.path.join(self.module_dir, 'csv/datatest.csv')
        data_result   = os.path.join(self.module_dir, 'csv/dataresult.csv')
        result_file   = open(data_result, 'w', encoding='utf-8')
        result_writer = csv.writer(result_file)
        
        with open(model_analys, 'rb') as handle:
            model = pickle.load(handle)

        with open(count_vector, 'rb') as h:
            cvec = pickle.load(h)
            
        with open(tfid_trans, 'rb') as t:
            tfid = pickle.load(t)

        result_writer.writerow(['tanggal', 'user_name', 'isi', 'stopword', 'label', 'polarity'])
        with open(data_test,'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i,line in enumerate(reader):
                tgl                     = line[0].split(',')[1]
                user                    = line[0].split(',')[2]
                isi                     = line[0].split(',')[4]
                removed_punct_isi       = self.remove_punct(isi)
                tokenization_isi        = self.tokenization(removed_punct_isi)
                remove_stopwords_isi    = self.remove_stopwords(tokenization_isi)
                fit_stopword_isi        = self.fit_stopwords(remove_stopwords_isi)
                
                transform_cvec = cvec.transform([fit_stopword_isi])
                transform_tfid = tfid.transform(transform_cvec)

                predict_result = model.predict(transform_tfid)
                label          = self.convert(predict_result)
            
                result_writer.writerow([tgl, user, isi, fit_stopword_isi, label, predict_result])

    def convert(self, polarity, mode="str"):
        if mode == "str":
            if str(polarity) == '[1]' or str(polarity) == 1:
                return "positif"
            elif str(polarity) == '[-1]' or str(polarity) == -1:
                return "negatif"
            else:
                return "netral"
        elif mode == "num":
            if str(polarity) == 'positif':
                return 1
            elif str(polarity) == 'negatif':
                return -1
            else:
                return 0
        else:
            return "something wrong"

    def classification(self):
        bow_transformer = CountVectorizer()
        X = [data.stop_removal for data in self.data]
        X_train_dtm = bow_transformer.fit_transform(X)
        X = bow_transformer.fit_transform(X)
        y = [self.convert(data.label, mode="num") for data in self.data]

        # splitting X and y into training and testing sets;
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # check which columns are expected by the model, but not exist in the inference dataframe
        not_existing_cols = [c for c in X_train_dtm.columns.tolist() if c not in X]
        # add this columns to the data frame
        X = X.reindex(X.columns.tolist() + not_existing_cols, axis=1)
        # new columns dont have values, replace null by 0
        X.fillna(0, inplace = True)
        # use the original X structure as mask for the new inference dataframe
        X = X[X_train_dtm.columns.tolist()]

        # Fit the model on training set
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # save the model to disk
        filename = os.path.join(self.module_dir, 'classification/model_analisis.pkl')
        pickle.dump(model, open(filename, 'wb'))

        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.score(X_test, y_test)
        print("Classification complete")
        print(result)


class ClassiView(generic.TemplateView):
    template_name       = 'klasifikasi.html'
    data                = KPKTweet.objects.all()
    data2               = KPKTweetTemp.objects.all()
    all_data            = data2.union(data)

    # Read **`file.csv`** into a pandas DataFrame
    # path = open(r'D:/sentiment_analysis/src/kpktweet/csv/dataresult.csv')
    path    = os.path.join(os.path.dirname(__file__), 'csv/dataresult.csv')
    df      = pd.read_csv(path, error_bad_lines=False, low_memory=False)

    # define X and y using the original DataFrame
    # X = df['stopword']
    # y = df['polarity']
    X   = [data.stop_removal for data in all_data]
    y   = [data.polarity for data in all_data]

    # splitting X and y into training and testing sets;
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # import and instantiate CountVectorizer
    vect = CountVectorizer()

    # create document-term matrices using CountVectorizer
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    # import and instantiate MultinomialNB
    nb = MultinomialNB()

    # fit a Multinomial Naive Bayes model
    nb.fit(X_train_dtm, y_train)

    # make class predictions
    y_pred_class = nb.predict(X_test_dtm)

    # generate classification report
    report = metrics.classification_report(y_test, y_pred_class, output_dict=True)
    report_df =  pd.DataFrame(report).transpose()
    # report_html = report_df.to_html()

    # print(report)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["report"]       = self.report_df

        return context