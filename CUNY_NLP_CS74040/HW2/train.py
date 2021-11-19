from preprocess2 import process
from NB import NB
import pickle

def main(tr_dir, ts_dir, para_dir):
    proc = process('G://CUNY/NLP/Assignments/HW2/train2.txt',
                   'G://CUNY/NLP/Assignments/HW2/test2.txt')
    proc_f= open(para_dir, 'wb')
    pickle.dump(proc, proc_f)
    proc_f.close()






if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier')
    parser.add_argument('--tr_dir', metavar='path', required=True, help='training path')
    parser.add_argument('--ts_dir', metavar='path', required=True, help='test path')
    parser.add_argument('--para_dir', metavar='path', required=True, help='parameter path')
    args = parser.parse_args()
    main(tr_dir = args.tr_dir, ts_dir = args.ts_dir, para_dir = args.para_dir)