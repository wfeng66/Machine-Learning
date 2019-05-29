# This script use to compare the proportion of private tweets in different filter factors

def pvtnel(wb, ws, df, filtering, result):      # according to calling requirement to filter and calculate percentage
    while (wb + ws) <= 1.0:                     # wb - window base; ws - window size; filter - sent,subj or both; result - the 2d list use to store result
        numel = 0
        numnel = 0
        numprv_el = 0
        numprv_nel = 0
        windows = [['', 0, 0, 0, 0.0, 0.0, 0, 0.0, 0.0]]
        if filtering == 'Sent':
            fltid = 2
        elif filtering == 'Subj':
            fltid = 3
        elif filtering == 'Both':
            for row in df:				# filter with both factors
                if ((abs(row[2]) < wb+ws and abs(row[2]) >= wb) and (
                    abs(row[3]) < wb+ws and abs(row[3]) >= wb)):  # fill data into windows
                    windows.append(row)
        if filtering == 'Sent' or filtering == 'Subj':
            for row in df:
                if (abs(row[fltid]) < wb + ws and abs(row[fltid]) >= wb):  # fill data into windows
                    windows.append(row)
        for row in windows:				# calculate the number of private tweets with E.L. words or not
            if row[4] == 1:
                numel = numel + 1
                if row[1] == "Private":
                    numprv_el = numprv_el + 1
            if row[4] == 0:
                numnel = numnel + 1
                if row[1] == "Private":
                    numprv_nel = numprv_nel + 1
        if numel == 0:
            pctofpvtwel = -1
            pctofnpvtwel = -1
        else:
            pctofpvtwel = numprv_el / numel
            pctofnpvtwel = 1 - pctofpvtwel
        if numnel == 0:
            pctofpvtwnel = -1
            pctofnpvtwnel = -1
        else:
            pctofpvtwnel = numprv_nel / numnel
            pctofnpvtwnel = 1 - pctofpvtwnel
        result.append([filtering, wb, ws, numprv_el, pctofpvtwel, pctofnpvtwel, numprv_nel, pctofpvtwnel,
                       pctofnpvtwnel])
        wb += ws


def main():
    import math
    import datetime
    import copy
    import time
    import csv
    import pandas as pd
    import numpy as np
    import random
    from textblob import TextBlob
    from textblob.classifiers import NaiveBayesClassifier
    from profanity_check import predict, predict_prob


    winbase = 0.0
    winsize = 0.2
    result = [[]]

    # load tweets data file
    with open('../Data/tweets.csv', 'r') as f:
        myread = csv.reader(f)
        df = list(myread)

    df_backup = copy.deepcopy(df)

    # create full matrix which includes sentiments score, subjectivity score and explicit language (profanity)
    for row in df:
        tw = TextBlob(row[0])
        t = tw.sentiment
        profanity = predict([row[0]])
        row.append(tw.sentiment.polarity)
        row.append(tw.sentiment.subjectivity)
        row.append(profanity[0])

    df_finish_backup = copy.deepcopy(df)

    filters = ['Sent', 'Subj', 'Both']
    for filter in filters:
        pvtnel(winbase,winsize, df_finish_backup, filter, result)

    result_df = pd.DataFrame(result, columns=['Filter', 'Windows Base', 'Windows Size', '# of PP w/ EL',
                                               '% of PP w/ EL', '% of NP /w EL', '# of PP /w NEL', '% of PP /w NEL',
                                               '% of NP /w NEL'])
    t = datetime.datetime.now()
    filename = '%s%s%s' % (t.year, t.month, t.day) + 'filters_comp' + str(len(df_backup)) + '.csv'
    export_csv = result_df.to_csv(filename, index=None, header=True)


if __name__ == "__main__":
    main()