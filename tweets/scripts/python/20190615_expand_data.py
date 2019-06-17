# This script is used for expand train data set
# 1. use Lemmatization to replace original word, lemmatization to replacing can increase the probability 
#    of occurrence, that has positive impact on test accuracy
# 2. use synsets to replace original word, so that the training set can be expanded

def obj2str(synobj):
    syn_word = str(synobj)             # convert object to string
    if ".n." in syn_word:           # retrieve the exactly synonym word from synset object string
        syn_word = syn_word[8:syn_word.find(".n.")] 
    elif ".v." in syn_word:
        syn_word = syn_word[8:syn_word.find(".v.")] 
    elif ".s." in syn_word:
        syn_word = syn_word[8:syn_word.find(".s.")] 
    elif ".r." in syn_word:
        syn_word = syn_word[8:syn_word.find(".r.")] 
    elif ".a." in syn_word:
        syn_word = syn_word[8:syn_word.find(".a.")] 
    return syn_word


def getOldWordObj(wd):
    from textblob import Word
    result = None
    if Word(wd).synsets == []:
        return result
    else:        
        for synset in Word(wd).synsets:
            #if syn_word.lower() == (" "+ wd.lower() + " "):  
            syn_word = obj2str(synset)
            if syn_word.lower() == wd.lower():  
                result = synset 
                break
    return result


def main():
    import math
    import datetime
    import copy
    import time
    import csv
    import pandas as pd
    import numpy as np
    from textblob import TextBlob
    from textblob import Word
    from textblob.wordnet import Synset
    import re


    new_df = [[]]
    
    
    # load tweets data file
    with open('../Data/tweets.csv', 'r') as f:
        myread = csv.reader(f)
        df = list(myread)

    df_backup = copy.deepcopy(df)
    
    for row in df:                              # iterate rows in df list
        orig_tw = re.findall(r'\w+', row[0])    # separate words from each tweets, orig_tw is a list stored all the words in one tweet
        for wd in orig_tw:                      # wd is a single word in each tweet
            wd = Word(wd).lemmatize()           # each word in tweets is lemmatized
            wd_orig_obj = getOldWordObj(wd)     # wd_orig_obj is used to store the synset object of original word
            if wd_orig_obj is None:
                continue
            for syn in Word(wd).synsets:        # syn is a single synonym object in synsets list
                distance = wd_orig_obj.path_similarity(syn)     # the similarity between original word and synonyms
                syn_word = obj2str(syn)
                if syn_word.lower() == wd.lower():       # if the synonym is same as original word do nothing
                    continue
                else:                           # if the synonym is different from original word replace it
                    rgx = "\\W"+wd+"\\W"
                    text = re.sub(rgx, syn_word, row[0], flags = re.I)
                    new_df.append([text, row[1], wd, syn_word, distance])
    
    new_df = new_df[1:]                         # remove the first empty row
    new_df = pd.DataFrame(new_df, columns=['text', 'privacy', 'orig_wd', 'rplc_wd', 'dist'])  # convert 2d list to dataframe, and add columns names
    t = datetime.datetime.now()
    filename = '%s%s%s' % (t.year, t.month, t.day) + 'expanded_tweets' + str(len(new_df)) + '.csv'
    export_csv = new_df.to_csv(filename, index=None, header=True)
            
            
if __name__ == "__main__":
    main()


