
def prepro(Corpus):
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
    from nltk.corpus import stopwords
    from nltk.corpus import wordnet as wn
    from collections import defaultdict
    from nltk.stem import WordNetLemmatizer
    Corpus['text'].dropna(inplace=True)
    #Corpus['text'] = [entry.lower() for entry in Corpus['text']]
    Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(Corpus['text']):
        Final_words = []
        word_Lemmatized = WordNetLemmatizer()
        for word, tag in pos_tag(entry):
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        Corpus.loc[index,'text_final'] = str(Final_words) 
    return Corpus

def encodvect(Corpus, Train_X, Train_Y, Test_X, Test_Y):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(Corpus['text_final'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    return Corpus, Train_Y, Test_Y, Train_X_Tfidf, Test_X_Tfidf

def main():

    import math
    import datetime
    import csv
    import pandas as pd
    import numpy as np
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from sklearn.preprocessing import LabelEncoder
    from collections import defaultdict
    from nltk.corpus import wordnet as wn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn import model_selection, naive_bayes, svm
    from sklearn.metrics import accuracy_score
    from textblob import TextBlob
    from textblob.classifiers import NaiveBayesClassifier	

    iterTimes = 100
    moveSize = 500
    dfSize = 5000
    testProportion = 0.2
    np.random.seed(500)
    #df = pd.read_csv(r"../Data/tweets.csv",encoding='latin-1')
    # Load the data from file

    with open('../Data/tweets.csv', 'r') as f:
        myread = csv.reader(f)
        df = list(myread)
    df_backup = df

    #df.columns = ['text','label']
    
    
    acc_gen = []
    acc_nb_gen = []
    acc = [[]]
    # General SVM in bigger training set
    for iter in range(0,(iterTimes-1)):
        if (moveSize*iter+dfSize)<6023:
            df = df_backup[(moveSize*iter):(moveSize*iter+dfSize)]
        else:
            df = df_backup[(moveSize*iter):]+df_backup[0:(dfSize-(6023-(moveSize*iter)))]
        # df_nb_xxxx for NaiveBayes model
        df_nb_train = df[:int(dfSize*(1-testProportion))]
        df_nb_test = df[int(dfSize*(1-testProportion)):]
        # prepare for svm        
        df = pd.DataFrame(df)
        df.columns = ['text','label']
        df = prepro(df)
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['text_final'],df['label'],test_size=testProportion)
        Corpus, Train_Y, Test_Y, Train_X_Tfidf, Test_X_Tfidf = encodvect(df, Train_X, Train_Y, Test_X, Test_Y)
        # SVM train and predict
        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        SVM.fit(Train_X_Tfidf,Train_Y)
        predictions_SVM = SVM.predict(Test_X_Tfidf)
        acc_gen.append(accuracy_score(predictions_SVM, Test_Y)*100)
        # NaiveBayes train and predict
        cl = NaiveBayesClassifier(df_nb_train)
        acc_nb_gen.append(cl.accuracy(df_nb_test)*100)		

    acc_gen_max = max(acc_gen)
    acc_gen_min = min(acc_gen)
    acc_gen_avg = sum(acc_gen)/len(acc_gen)
    acc_nb_gen_max = max(acc_nb_gen)
    acc_nb_gen_min = min(acc_nb_gen)
    acc_nb_gen_avg = sum(acc_nb_gen)/len(acc_nb_gen)  
    acc.append(['SVM','none',dfSize,acc_gen_min,acc_gen_avg,acc_gen_max])
    acc.append(['NB','none',dfSize,acc_nb_gen_min,acc_nb_gen_avg,acc_nb_gen_max])	

    print ("Data source from tweets.csv, which totally includes %d reccords. And iteration time is %d.\n"%(len(df),iterTimes))
    print ("-"*100)
    print ("Model\t\tSentiment\tsize\t\tmin_acc\t\tavg_acc\t\tmax_acc\n")
    print ("*"*100)
    print ("SVM\t\tnone\t\t%d\t%f\t%f\t%f\n" % (dfSize,acc_gen_min,acc_gen_avg,acc_gen_max))
    print ("NB\t\tnone\t\t%d\t%f\t%f\t%f\n" % (dfSize,acc_nb_gen_min,acc_nb_gen_avg,acc_nb_gen_max))
   
    # SVM in different training size data without sentiment filtering
    for size in range(30, 500, 10):
        # General SVM Classification
        acc_gen = []
        acc_nb_gen = []		
        for iter in range(iterTimes):
            if (moveSize*iter+size)< 6023:
                df = df_backup[(moveSize*iter):(moveSize*iter+size)]
            else: 
                df = df_backup[(moveSize*iter):]+df_backup[0:(size - (6023-(moveSize*iter)))]
            # df_nb_xxxx for NaiveBayes model
            df_nb_train = df[:int(len(df)*(1-testProportion))]
            df_nb_test = df[int(len(df)*(1-testProportion)):]
            # prepare for svm             
            df = pd.DataFrame(df)
            df.columns = ['text','label']
            df = prepro(df)
            Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['text_final'],df['label'],test_size=testProportion)
            Corpus, Train_Y, Test_Y, Train_X_Tfidf, Test_X_Tfidf = encodvect(df, Train_X, Train_Y, Test_X, Test_Y)
            # SVM train and predict
            SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
            SVM.fit(Train_X_Tfidf,Train_Y)
            predictions_SVM = SVM.predict(Test_X_Tfidf)
            acc_gen.append(accuracy_score(predictions_SVM, Test_Y)*100)
            # NaiveBayes train and predict
            cl = NaiveBayesClassifier(df_nb_train)
            acc_nb_gen.append(cl.accuracy(df_nb_test)*100)   			
        acc_gen_max = max(acc_gen)
        acc_gen_min = min(acc_gen)
        acc_gen_avg = sum(acc_gen)/len(acc_gen)
        acc_nb_gen_max = max(acc_nb_gen)
        acc_nb_gen_min = min(acc_nb_gen)
        acc_nb_gen_avg = sum(acc_nb_gen)/len(acc_nb_gen)
        acc.append(['SVM','none',size,acc_gen_min,acc_gen_avg,acc_gen_max])
        acc.append(['NB','none',size,acc_nb_gen_min,acc_nb_gen_avg,acc_nb_gen_max])		
        print ("SVM\t\tnone\t\t%d\t%f\t%f\t%f\n" % (size,acc_gen_min,acc_gen_avg,acc_gen_max))
        print ("NB\t\tnone\t\t%d\t%f\t%f\t%f\n" % (size,acc_nb_gen_min,acc_nb_gen_avg,acc_nb_gen_max))

    # SVM with sentiment filter 
    for sent in range(10):
        df = df_backup
	    # sentiment analysis
        for row in df:
            tw = TextBlob(row[0])
            t = tw.sentiment
            row.append(tw.sentiment.polarity)
            row.append(tw.sentiment.subjectivity)
	    # filter sentiment score
        data = [['','',0.0,0.0]]     # used for storing filtered data
        for row in df:
            if  ((row[2]<(-(sent/10)) or row[2]>sent/10) and row[3]>sent/10):
                data.append(row)
        data=data[1:]
        # delete last 2 columns for preparing training
        data = [[x[0],x[1]] for x in data]
        sizeofdf = len(data)
        dfSize = math.floor(len(data)*0.8)
        moveSize = 10 
        acc_sent = []     

        # preparation for  SVM model
        for iter in range(iterTimes):
            if (moveSize*iter+dfSize)< sizeofdf:
                df = data[(moveSize*iter):(moveSize*iter+dfSize)]
            else: 
                df = data[(moveSize*iter):]+data[0:(dfSize - (sizeofdf-(moveSize*iter)))]
            # df_nb_xxxx for NaiveBayes model
            df_nb_train = df[:int(len(df)*(1-testProportion))]
            df_nb_test = df[int(len(df)*(1-testProportion)):]
            # prepare for svm             
            df = pd.DataFrame(df)
            df.columns = ['text','label']
            df = prepro(df)
            Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['text_final'],df['label'],test_size=testProportion)
            Corpus, Train_Y, Test_Y, Train_X_Tfidf, Test_X_Tfidf = encodvect(df, Train_X, Train_Y, Test_X, Test_Y)

        # train and predict SVM model      
            SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
            SVM.fit(Train_X_Tfidf,Train_Y)
            predictions_SVM = SVM.predict(Test_X_Tfidf)
            acc_sent.append(accuracy_score(predictions_SVM, Test_Y)*100)
            # NaiveBayes train and predict
            cl = NaiveBayesClassifier(df_nb_train)
            acc_nb_gen.append(cl.accuracy(df_nb_test)*100) 			
        acc_sent_max = max(acc_sent)
        acc_sent_min = min(acc_sent)
        acc_sent_avg = sum(acc_sent)/len(acc_sent)
        acc_nb_gen_max = max(acc_nb_gen)
        acc_nb_gen_min = min(acc_nb_gen)
        acc_nb_gen_avg = sum(acc_nb_gen)/len(acc_nb_gen)
        acc.append(['SVM',sent/10,dfSize,acc_gen_min,acc_gen_avg,acc_gen_max])
        acc.append(['NB',sent/10,dfSize,acc_nb_gen_min,acc_nb_gen_avg,acc_nb_gen_max])	
        print ("svm\t\t%f\t%d\t%f\t%f\t%f\n" % (sent/10,dfSize,acc_sent_min,acc_sent_avg,acc_sent_max))
        print ("NB\t\t%f\t%d\t%f\t%f\t%f\n" % (sent/10,dfSize,acc_nb_gen_min,acc_nb_gen_avg,acc_nb_gen_max))
        
    acc_df = pd.DataFrame(acc, columns = ['Model','Sentiment','Size','min_acc','avg_acc','max_acc'])
    t = datetime.datetime.now()
    filename = '%s%s%s'%(t.year,t.month,t.day) + '_tweets_' + str(len(df_backup)) + '_' + str(iterTimes) + '.csv'
    export_csv = acc_df.to_csv(filename, index = None, header = True)

if __name__ == "__main__":
    main()
        