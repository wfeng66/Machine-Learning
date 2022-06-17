


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
            for syn in Word(wd).synsets:        # syn is a single synonym object in synsets list
                syn_word = str(syn)             # convert object to string
                if ".n." in syn_word:           # retrieve the exactly synonym word from synset object string
                    syn_word = " "+ syn_word[8:syn_word.find(".n.")] + " "
                elif ".v." in syn_word:
                    syn_word = " "+ syn_word[8:syn_word.find(".v.")] + " "
                elif ".s." in syn_word:
                    syn_word = " "+ syn_word[8:syn_word.find(".s.")] + " "
                elif ".r." in syn_word:
                    syn_word = " "+ syn_word[8:syn_word.find(".r.")] + " "
                elif ".a." in syn_word:
                    syn_word = " "+ syn_word[8:syn_word.find(".a.")] + " "
                if syn_word.lower() == (" "+ wd.lower() + " "):      # if the synonym is same as original word do nothing
                    pass
                else:                                   # if the synonym is different from original word replace it
                    rgx = "\\W"+wd+"\\W"
                    text = re.sub(rgx, syn_word, row[0], flags = re.I)
                    new_df.append([text, row[1]])
    
    new_df = new_df[1:]                     # remove the first empty row
    new_df = pd.DataFrame(new_df, columns=['text', 'privacy'])  # convert 2d list to dataframe, and add columns names
    t = datetime.datetime.now()
    filename = '%s%s%s' % (t.year, t.month, t.day) + 'expanded_tweets' + str(len(new_df)) + '.csv'
    export_csv = new_df.to_csv(filename, index=None, header=True)
            
            
if __name__ == "__main__":
    main()


