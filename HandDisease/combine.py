import os.path
import sys
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

class comb:
    def __init__(self, path):
        self.path = path
        self.methods = ['Weight', 'CM']
        self.digits = ['t', 'i', 'm', 'r', 'l']
        self.vars1 = ['Ax', 'Ay', 'Roll', 'Px', 'Py', 'Pz']
        self.vars2 = []
        self.data = [[0.0 for _ in range(7)] for _ in range(63)]
        self.label = [0 for _ in range(63)]
        self.result = np.empty((63, 7, 9988, 144))
        self.n_cohort1 = 26
        self.n_cohort2 = 37
        self.n_cohort1_cntl = 13
        self.n_cohort2_cntl = 18
        self.experments = ['Weight00', 'Weight02', 'CM01', 'CM02']
        for var in ['Fn', 'Ftan', 'Fx', 'Mx', 'My', 'Mz']:
            for d in self.digits:
                self.vars2.append(var+d)
        self.clmns = [e+v for e in self.experments for v in self.vars1+self.vars2]
        print(len(self.vars1+self.vars2), len(self.clmns))
        self.temp = []
    def createEmptyDF(self):
        return pd.DataFrame(columns=self.clmns)
    def readXls(self, filename):
        df = pd.read_csv(filename, sep='\t')
        return df.iloc[11:, :]
    def addData(self, df, filename, method, cohort, var):
        # print(filename)
        filename = filename.split('\\')[1].split('.')[0]
        # filename = os.path.basename(filename)[:-4]
        if cohort == 'cohort1':         # Cohort 1
            if 'Control' in filename:   # control group in Cohort 1
                num = 0
            else:                       # CTS group in Cohort 1
                num = self.n_cohort1_cntl
        else:                           # Cohort 2
            if 'Control' in filename:   # control group in Cohort 2
                num = self.n_cohort1
            else:                       # CTS group in Cohort 2
                num = self.n_cohort1 + self.n_cohort2_cntl
        if 'CTS' in filename:           # CTS files
            clmn = method + filename[-2:] + filename[:-7]
        else:                           # Control files
            clmn = method + filename[-2:] + filename[:-11]
        clmn = clmn[:-1] + clmn[-1].lower()
        self.temp.append(clmn)
        # clmn = method + filename[-2:] + var
        inClmns = clmn in self.clmns
        if not inClmns:
            print(var, clmn)
        # print(filename[-4:-2], filename, method, cohort, var)
        n_case = num + int(filename[-4:-2]) - 1
        if 'Control' in filename:
            self.label[n_case] = 0
        else:
            self.label[n_case] = 1
        for clmnid in range(7):         # clmnid is the number of time in experiments and the column id in original files
            # print(num, filename, int(filename[-4:-2]), n_case, clmnid, clmn)
            self.data[n_case][clmnid].loc[:, clmn] = df.iloc[:, clmnid]
    def traverse(self):
        # initialize data structure
        for i in range(63):
            for j in range(7):
                self.data[i][j] = self.createEmptyDF()
        print('empty: ', self.data[0][0].shape)
        for method in tqdm(self.methods):
            print(method)
            for cohort in tqdm(['cohort1', 'cohort2']):
                print(cohort)
                vars = ['Ax', 'Ay', 'Roll', 'Px', 'Py', 'Pz', 'Fn', 'Ftan', 'Fx', 'Mx', 'My', 'Mz']
                for var in tqdm(vars):
                    print(var)
                    print(self.data[0][0].shape)
                    p = self.path + method + '/' + method + '_' + cohort + '/' + var + '/'
                    # print(p)
                    filelist = glob(p+'*.xls')
                    # print(filelist)
                    for file in filelist:
                        if ')' in file:
                            continue
                        else:
                            self.addData(self.readXls(file), file, method, cohort, var)
                # for var in tqdm(self.vars1):
                #     print(var)
                #     print(self.data[0][0].shape)
                #     p = self.path + method + '/' + method + '_' + cohort + '/' + var + '/'
                #     # print(p)
                #     filelist = glob(p+'*.xls')
                #     # print(filelist)
                #     for file in filelist:
                #         if ')' in file:
                #             continue
                #         else:
                #             self.addData(self.readXls(file), file, method, cohort, var)
                # for var in tqdm(self.vars2):
                #     print(var)
                #     print(self.data[0][0].shape)
                #     p = self.path + method + '/' + method + '_' + cohort + '/' + var[:-1] + '/'
                #     filelist = glob(p + '*.xls')
                #     for file in filelist:
                #         if ')' in file:
                #             continue
                #         else:
                #             self.addData(self.readXls(file), file, method, cohort, var)
        print('create new columns: ', len(set(self.temp)))
        # filelist = ['C:/Study/PhD/HandDisease/Data/Weight/Weight_cohort1/Ax\AxControl0100.xls',
        #             'C:/Study/PhD/HandDisease/Data/Weight/Weight_cohort1/Ax\AxControl0102.xls',
        #             'C:/Study/PhD/HandDisease/Data/Weight/Weight_cohort1/Ax\AxControl0200.xls',
        #             'C:/Study/PhD/HandDisease/Data/Weight/Weight_cohort1/Ax\AxControl0202.xls']
        # for file in filelist:
        #     method, cohort, var = 'Weight', 'cohort1', 'Ax'
        #     self.addData(self.readXls(file), file, method, cohort, var)
    def save(self):
        #convert the dataframe to np array
        for i in range(63):
            if i == 30 or i == 40:
                continue
            for j in range(7):
                print(self.data[i][j].shape, self.result.shape)
                self.result[i, j, :, :] = self.data[i][j].to_numpy()
        self.label = np.array(self.label)
        # self.data = np.array(self.data, dtype=np.float16)
        np.save(self.path + 'combData', self.result)
        np.save(self.path + 'label', self.label)


def main(args):
    # print(len(args))
    path = args[1]
    # print(path)
    # folders = path.split('\\')
    # path = ''
    # for f in folders:
    #     path = path + f + '/'
    c = comb(path)
    c.traverse()
    c.save()


if __name__ == "__main__":
    main(sys.argv)














