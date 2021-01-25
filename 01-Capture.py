import requests
import re
from datetime import date
import pandas as pd
import os

stockFundCode = ['002083', '290011', '005888', '161725', '009777']

for code in stockFundCode:

    # Date
    sDate = re.findall('</span>ï¼(\S+)</td>', requests.get('http://fund.eastmoney.com/{}.html'.format(code)).text)[0]
    eDate = date.today().strftime('%Y-%m-%d')
    # Config
    stockFundAPI = 'https://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={}&page=1&sdate={}&edate={}&per=20'.format(
        code, sDate, eDate)
    pages = int(re.findall(r'pages:(\d+),curpage', requests.get(stockFundAPI).text)[0])
    df = pd.DataFrame(None)
    # Capture
    for page in range(1, pages):
        stockFundAPI = 'https://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={}&page={}&sdate={}&edate={}&per=20'.format(
            code, page, sDate, eDate)
        dfTemp = pd.read_html(stockFundAPI, encoding='utf-8')[0]
        df = pd.concat([df, dfTemp], ignore_index=True)
    # Data Munging
    df = df.iloc[:, :-3]
    df = df.dropna()
    df.columns = ['Date', 'NetValue', 'CumValue', 'NetRate']
    df['NetValue'] = df['NetValue'].astype(float)
    df['CumValue'] = df['CumValue'].astype(float)
    df['NetRate'] = df['NetRate'].apply(lambda x: float(x.strip('%')) / 100)
    df = df[::-1]
    df.reset_index(drop=True, inplace=True)
    # Store stockFundData
    outDir = './stockFundData'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    df.to_pickle(os.path.join(outDir, '{}'.format(code)))
    print('stockFundData: {}\n'.format(code))
