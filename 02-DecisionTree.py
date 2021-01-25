import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

stockFundCode = ['002083', '290011', '005888', '161725', '009777']

for code in stockFundCode:

    # Preprocess
    outDir = './stockFundData'
    df = pd.read_pickle(os.path.join(outDir, '{}'.format(code)))
    # Avg Feature
    Day10Avg = []
    Day20Avg = []
    for index, row in df.iterrows():
        if index < 10 - 1:
            Day10Avg.append(np.nan)
        else:
            dfTemp = df[index - (10 - 1):index + 1]
            RelMean = row['NetValue'] - dfTemp['NetValue'].mean()
            Day10Avg.append(RelMean)
        if index < 20 - 1:
            Day20Avg.append(np.nan)
        else:
            dfTemp = df[index - (20 - 1):index + 1]
            RelMean = row['NetValue'] - dfTemp['NetValue'].mean()
            Day20Avg.append(RelMean)
    df['Day10Avg'] = [round(x, 4) for x in Day10Avg]
    df['Day20Avg'] = [round(x, 4) for x in Day20Avg]
    # Min-Max Feature
    Day10Min = []
    Day20Min = []
    Day10Max = []
    Day20Max = []
    for index, row in df.iterrows():
        if index < 10 - 1:
            Day10Min.append(np.nan)
            Day10Max.append(np.nan)
        else:
            dfTemp = df[index - (10 - 1):index + 1]
            RelMin = row['NetValue'] - dfTemp['NetValue'].min()
            RelMax = row['NetValue'] - dfTemp['NetValue'].max()
            Day10Min.append(RelMin)
            Day10Max.append(RelMax)
        if index < 20 - 1:
            Day20Min.append(np.nan)
            Day20Max.append(np.nan)
        else:
            dfTemp = df[index - (20 - 1):index + 1]
            RelMin = row['NetValue'] - dfTemp['NetValue'].min()
            RelMax = row['NetValue'] - dfTemp['NetValue'].max()
            Day20Min.append(RelMin)
            Day20Max.append(RelMax)
    df['Day10Min'] = [round(x, 4) for x in Day10Min]
    df['Day20Min'] = [round(x, 4) for x in Day20Min]
    df['Day10Max'] = [round(x, 4) for x in Day10Max]
    df['Day20Max'] = [round(x, 4) for x in Day20Max]
    # nNetRate Label
    label = []
    for index, row in df.iterrows():
        if index == len(df) - 1:
            label.append(np.nan)
        else:
            nNetRate = df.iloc[index + 1]['NetRate']
            if nNetRate > 0:
                label.append('Increase')
            elif nNetRate < 0:
                label.append('Decrease')
            else:
                label.append(np.nan)
    df['Label'] = label
    # Drop Nan & Save Feature and Label
    df.dropna(inplace=True)
    df.drop(['Date', 'CumValue'], axis=1, inplace=True)
    # Store stockFundData-ML
    outDir = './stockFundData-ML'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    outDir = './stockFundData-ML/DecisionTree'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    df.to_pickle(os.path.join(outDir, '{}-ML'.format(code)))
    print('stockFundData-ML: {}'.format(code))

    # Machine Learning
    sd = pd.read_pickle(os.path.join(outDir, '{}-ML'.format(code)))
    # Label Extraction
    label_mapping = {'Increase': 0, 'Decrease': 1}
    sd['Label'] = sd['Label'].map(label_mapping)
    X = sd.drop(['Label'], axis=1).to_numpy()
    y = sd.pop('Label').to_numpy()
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # Create Model
    model = DecisionTreeClassifier()
    # Train Model
    model.fit(X_train, y_train)
    # Prediction
    pre = model.predict(X_test)
    # Accuracy
    acc = accuracy_score(y_test, pre)
    print('{} Accuracy:'.format(code), acc)

    # Save ML Model
    outDir = './ML-Model'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    outDir = './ML-Model/DecisionTree'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    filename = os.path.join(outDir, '{}.sav'.format(code))
    pickle.dump(model, open(filename, 'wb'))
    print('{} Model Saved\n'.format(code))
