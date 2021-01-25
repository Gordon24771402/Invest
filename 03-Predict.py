import os
import requests
import json
import re
from random import randint
import pandas as pd
import numpy as np
import pickle

stockFundCode = ['002083', '290011', '005888', '161725', '009777']


def pre(method):

    for code in stockFundCode:
        # Predict
        outDir = './stockFundData-ML/{}'.format(method)
        df = pd.read_pickle(os.path.join(outDir, '{}-ML'.format(code)))
        # Capture JSON Data
        headers = {'content-type': 'application/json', 'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'}
        content = requests.get('http://fundgz.1234567.com.cn/js/{}.js?rt={}'.format(code, randint(1000000, 3000000)), headers=headers).text
        search = re.findall(r'^jsonpgz\((.*)\)', content)
        for i in search:
            data = json.loads(i)
        # Compute Current Features
        cNetRate = float(data['gszzl'])
        cNetValue = float(data['gsz'])
        Day10Avg = cNetValue - np.append(df[-9:]['NetValue'].to_numpy(), cNetValue).mean()
        Day20Avg = cNetValue - np.append(df[-19:]['NetValue'].to_numpy(), cNetValue).mean()
        Day10Min = cNetValue - np.append(df[-9:]['NetValue'].to_numpy(), cNetValue).min()
        Day20Min = cNetValue - np.append(df[-19:]['NetValue'].to_numpy(), cNetValue).min()
        Day10Max = cNetValue - np.append(df[-9:]['NetValue'].to_numpy(), cNetValue).max()
        Day20Max = cNetValue - np.append(df[-19:]['NetValue'].to_numpy(), cNetValue).max()
        # Load Model
        outDir = './ML-Model'
        if not os.path.exists(outDir):
            os.mkdir(outDir)
        outDir = './ML-Model/{}'.format(method)
        if not os.path.exists(outDir):
            os.mkdir(outDir)
        filename = os.path.join(outDir, '{}.sav'.format(code))
        model = pickle.load(open(filename, 'rb'))
        # Compute Prediction
        X_test = np.array([cNetValue, cNetRate, Day10Avg, Day20Avg, Day10Min, Day20Min, Day10Max, Day20Max])
        pre = model.predict(X_test.reshape(1, -1))
        # Info
        if pre[0] == 0:
            print('{} Increase at {}'.format(code, data['gztime']))
            if data['gztime'][-5:] == '15:00':
                with open("{}-Record.txt".format(method), "a") as file:
                    file.write('\n{} Increase at {}'.format(code, data['gztime']))
        elif pre[0] == 1:
            print('{} Decrease at {}'.format(code, data['gztime']))
            if data['gztime'][-5:] == '15:00':
                with open("{}-Record.txt".format(method), "a") as file:
                    file.write('\n{} Decrease at {}'.format(code, data['gztime']))


pre('DecisionTree')
