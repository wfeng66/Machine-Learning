# This script is used for comparison of the accuracies between general NaiveBayes 
# Classification and NaiveBayes Classification basing on sentiment filtering on 
# the privacy of tweets
# Syntext: python 20190302_senti_comp.py 
# In order to change the times to iterate, change the value of variable of n

import math
import csv
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier

def main():
    n = 5
    # Load the data from file
    with open('../Data/tweets.csv', 'r') as f:
	    myread = csv.reader(f)
	    df = list(myread)

    df_t = df

    acc_gen = []
    print ("Data source from tweets.csv, which totally includes %d reccords. And iteration time is %d.\n"%(len(df),n))
    print ("-"*100)
    print ("Sentiment\t\tsize\t\tmin_acc\t\tavg_acc\t\tmax_acc\n")
    print ("*"*100)
    for size in range(50, 5500, 500):
        # General NaiveBayes Classification
        for iterTime in range(n):
            trainingdt = df[100*(iterTime+1):(100*(iterTime+1)+size)]
            testingdt = df[100*iterTime:100*(iterTime+1)]
            cl = NaiveBayesClassifier(trainingdt)
            acc_gen.append(cl.accuracy(testingdt))
        acc_gen_max = max(acc_gen)
        acc_gen_min = min(acc_gen)
        acc_gen_avg = sum(acc_gen)/len(acc_gen)
        print ("noon\t\t%d\t%f\t%f\t%f\n" % (size,acc_gen_min,acc_gen_avg,acc_gen_max))

    acc_sent = []

    # test in different sentiment score filtering
    for sent in range(10):
        df = df_t
	    # sentiment analysis
        for row in df:
            tw = TextBlob(row[0])
            t = tw.sentiment
            row.append(tw.sentiment.polarity)
            row.append(tw.sentiment.subjectivity)
	    # filter sentiment score
        data = [['','',0.0,0.0]]
        for row in df:
            if  ((row[2]<-(sent/10) or row[2]>sent/10) and row[3]>sent/10):
                data.append(row)
        data=data[1:]
        # delete last 2 columns for preparing training
        for row in data:
	        del row[3]
	        del row[2]
        slice_len = math.floor(len(data)/n)
        for iterTime in range(n):
            trainingdt = data[slice_len*(iterTime+1):]+data[slice_len*(iterTime-1):slice_len*iterTime]
            testingdt = data[slice_len*iterTime:slice_len*(iterTime+1)]
            cl = NaiveBayesClassifier(trainingdt)
            acc_sent.append(cl.accuracy(testingdt))
        acc_sent_max = max(acc_sent)
        acc_sent_min = min(acc_sent)
        acc_sent_avg = sum(acc_sent)/len(acc_sent)
        print ("%f\t%d\t%f\t%f\t%f\n" % (sent/10,slice_len,acc_sent_min,acc_sent_avg,acc_sent_max))

if __name__ == "__main__":
    main()
        

