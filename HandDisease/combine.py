import sys
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

class comb:
    def __init__(self, path):
        self.path = path
        self.methods = ['Weight', 'CM']
        self.vars = ['Ax', 'Ay', 'Roll', 'Fn', 'Ftan', 'Fx', 'Mx', 'My', 'Mz', 'Px', 'Py', 'Pz']
        self.data = [[0.0 for _ in range(7)] for _ in range(63)]
        self.label = [0 for _ in range(63)]
        self.result = np.empty((63, 7, 9988, 48))
        self.n_cohort1 = 26
        self.n_cohort2 = 37
        self.n_cohort1_cntl = 13
        self.n_cohort2_cntl = 18
    def createEmptyDF(self):
        experments = ['Weight00', 'Weight02', 'CM01', 'CM02']
        clmns = [e+v  for e in experments for v in self.vars]
        return pd.DataFrame(columns=clmns)
    def readXls(self, filename):
        df = pd.read_csv(filename, sep='\t')
        return df.iloc[11:, :]
    def addData(self, df, filename, method, cohort, var):
        filename = filename.split('\\')[1].split('.')[0]
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
        clmn = method + filename[-2:] + var
        # print(filename[-4:-2], filename, method, cohort, var)
        n_case = num + int(filename[-4:-2]) - 1
        if 'Control' in filename:
            self.label[n_case] = 0
        else:
            self.label[n_case] = 1
        for clmnid in range(7):
            # print(num, filename, int(filename[-4:-2]), n_case, clmnid, clmn)
            self.data[n_case][clmnid].loc[:, clmn] = df.iloc[:, clmnid]
    def traverse(self):
        # initialize data structure
        for i in range(63):
            for j in range(7):
                self.data[i][j] = self.createEmptyDF()
        for method in tqdm(self.methods):
            for cohort in ['cohort1', 'cohort2']:
                for var in self.vars:
                    p = self.path + method + '/' + method + '_' + cohort + '/' + var + '/'
                    # print(self.path, p)
                    filelist = glob(p+'*.xls')
                    for file in filelist:
                        if ')' in file:
                            continue
                        else:
                            self.addData(self.readXls(file), file, method, cohort, var)
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
                print(i,j)
                self.result[i, j, :, :] = self.data[i][j].to_numpy()
        self.label = np.array(self.label)
        # self.data = np.array(self.data, dtype=np.float16)
        np.save(self.path + 'combData', self.result)
        np.save(self.path + 'label', self.label)


def main(args):
    path = args[1]
    folders = path.split('\\')
    path = ''
    for f in folders:
        path = path + f + '/'
    c = comb(path)
    c.traverse()
    c.save()


if __name__ == "__main__":
    main(sys.argv)














