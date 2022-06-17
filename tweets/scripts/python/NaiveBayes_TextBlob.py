# This script use TextBlob NaiveBayesClassifier to classify
# social media post, including creating model, saving and 
# reusing the model

# Load data
import csv
with open('../Data/tweets.csv', 'r') as f:
	myread = csv.reader(f)
	df = list(myread)

# Split the data to training and testing dataset	
trainingdt = df[0:5000]
testingdt = df[5000:]

# Load the module and train the model
from textblob.classifiers import NaiveBayesClassifier
cl = NaiveBayesClassifier(trainingdt)

# Test the model
cl.accuracy(testingdt)
#>>> cl.accuracy(testingdt)
#0.7947214076246334

# save the model(classifier)
import pickle
object = cl
myfile = open('NaiveBayes_TextBlob.obj','wb')
pickle.dump(object,myfile)

# load the model(classifier)
with open('NaiveBayes_TextBlob.obj','rb') as f:
	 c = pickle.load(f)
# reuse the model
c.classify("I like this library!")
c.accuracy(testingdt)
