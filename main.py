#Python final project
#Link to the app: https://francisco0a-undervalued-stocks-mbfa-project-main-7kfsur.streamlit.app/

#Required modules for the app
import streamlit as st
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import yahooquery as yh
from yahooquery import Ticker
import matplotlib.pyplot as plt

#Configuring page
st.set_page_config(layout='wide')
st.title('Stock Screener: Find Undervalued Stocks for Different Indexes:money_with_wings::chart:')

#Adding section about information regarding the app
expander = st.expander('About this app')
expander.markdown('''This app helps investors to identify undervalued stocks from various indexes by utilizing financial ratios and real-time data from Yahoo Finance. Users can choose from different investment universes, including the S&P 500, STOXX Europe 600, Tel Aviv 125, Cotation Assistée en Continu 40, NZSX 50 Index, China CSI 300 Index, or selecting all the stocks. 
After importing the data, the app performs data cleaning and manipulation, filtering out stocks with incomplete or missing data. It retrieves essential financial ratios like PE ratio, Beta, PB ratio, EPS, PEG ratio, EVR ratio, and EV/EBITDA ratio.
The app then applies industry rankings and identifies undervalued stocks based on the selected ratio. Outliers are removed, and the undervalued stocks are sorted and displayed. 
* **Final project for the course of Data Science in Finance**
* **Université Paris 1, Pantheón-Sorbonne, PSME and MBFA** 
* **Members of the team:** SALONEN, Samuli, ALVAREZ, Francisco, & VIDAL, Laura
* **References:** [Yahoo Finance](https://finance.yahoo.com),
    [Streamlit library](https://docs.streamlit.io/library/api-reference),
    [Datacamp for finance](https://app.datacamp.com/learn/courses/bond-valuation-and-analysis-in-python),
    [Data Cleaning](https://www.kaggle.com/learn/data-cleaning)''')
st.header('Identify Undervalued Stocks for Your Portfolio')
st.caption('Disclaimer: This is not an investment advice, but it is a useful tool to make informed decisions for potentially undervalued stocks')

#Function to import yahoo data using the unofficial API
@st.cache_data
def get_stock_data(tickers):
    data = Ticker(tickers).get_modules('assetProfile price summaryDetail defaultKeyStatistics') 
    #assetProfile : industry , price : longName & regularMarketPrice , summaryDetail : twoHundredDayAverage , 
    #defaultKeyStatistics : forwardPE, beta , priceToBook , forwardEps , pegRatio , enterpriseToRevenue , enterpriseToEbitda
    return data

index_list = ['sp500', 'ta125', 'cac40', 'csi300', 'stoxx600']

#Data download from different sources and manipulation to make it a list
@st.cache_data
def get_index_constituents(index):
    if index == 'sp500':
        data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].to_list()
    elif index == 'ta125':
        data = pd.read_html('https://en.wikipedia.org/wiki/TA-125_Index#Constituents')[1]['Symbol'].to_list()
        data = [item + '.TA' for item in data]
    elif index == 'cac40':
        data = pd.read_html('https://en.wikipedia.org/wiki/CAC_40#Composition')[4]['Ticker'].to_list()
    elif index == 'csi300':
        data = pd.read_html('https://en.wikipedia.org/wiki/CSI_300_Index#Constituents')[3]['Index'].tolist()
        data = [str(num) for num in data]
        data = [item + '.SS' for item in data]
    elif index == 'stoxx600':
        data = pd.DataFrame(columns=['Ticker', 'Exchange'])
        for i in range(1, 20):
            url = 'https://www.dividendmax.com/market-index-constituents/stoxx600.html/?page=' + str(i)
            tickers = pd.read_html(url)[0][['Ticker', 'Exchange']]
            data = pd.concat([data, tickers])
        for i in range(0, len(data)):
            if any(x in data.iloc[i]['Exchange'] for x in ['Frankfurt', 'Xetra', 'Berlin', 'Luxembourg']):
                data.iloc[i] = data.iloc[i] + '.DE'
            elif 'London' in data.iloc[i]['Exchange']:
                data.iloc[i] = data.iloc[i] + '.L'
            elif 'Italian' in data.iloc[i]['Exchange']:
                data.iloc[i] = data.iloc[i] + '.MI'
            elif 'Amsterdam' in data.iloc[i]['Exchange']:
                data.iloc[i] = data.iloc[i] + '.AS'
            elif 'Stockholm' in data.iloc[i]['Exchange']:
                data.iloc[i] = data.iloc[i] + '.ST'
            elif 'Swiss' in data.iloc[i]['Exchange']:
                data.iloc[i] = data.iloc[i] + '.SW'
            elif 'Paris' in data.iloc[i]['Exchange']:
                data.iloc[i] = data.iloc[i] + '.PA'
            elif 'Brussels' in data.iloc[i]['Exchange']:
                data.iloc[i] = data.iloc[i] + '.BR'
            elif any(x in data.iloc[i]['Exchange'] for x in ['Madrid', 'Valencia', 'Barcelona']):
                data.iloc[i] = data.iloc[i] + '.MC'
            elif 'Irish' in data.iloc[i]['Exchange']:
                data.iloc[i] = data.iloc[i] + '.L'
            elif 'Oslo' in data.iloc[i]['Exchange']:
                data.iloc[i] = data.iloc[i] + '.OL'
            elif 'Copenhagen' in data.iloc[i]['Exchange']:
                data.iloc[i] = data.iloc[i] + '.CO'
            elif 'Vienna' in data.iloc[i]['Exchange']:
                data.iloc[i] = data.iloc[i] + '.VI'
            elif 'Lisbon' in data.iloc[i]['Exchange']:
                data.iloc[i] = data.iloc[i] + '.LS'
            elif 'Warsaw' in data.iloc[i]['Exchange']:
                data.iloc[i] = data.iloc[i] + '.WA'
            elif 'Athens' in data.iloc[i]['Exchange']:
                data.iloc[i] = data.iloc[i] + '.AT'
            elif 'Helsinki' in data.iloc[i]['Exchange']:
                data.iloc[i] = data.iloc[i] + '.HE'
        data = data['Ticker'].to_list()
        data = [x.replace('..', '.') for x in data]
        data = [x.replace(' ', '-') for x in data]
        data = [x.replace('.', '-', 1) if x.count('.')==2 else x for x in data]

    return data

sp500 = get_index_constituents('sp500')
ta125 = get_index_constituents('ta125')
cac40 = get_index_constituents('cac40')
csi300 = get_index_constituents('csi300')
stoxx600 = get_index_constituents('stoxx600')

#Creating Dataframe with the tickers that we have
list_of_tickers = st.selectbox('Choose investment universe.', 
                               ['Standard and Poors 500', 
                                'STOXX Europe 600', 
                                'Tel Aviv 125',
                                'Cotation Assistée en Continu 40', 
                                'China CSI 300 Index 沪深300',
                                'ALL'])

if list_of_tickers == 'Standard and Poors 500':
        list_of_tickers = sp500
elif list_of_tickers == 'STOXX Europe 600':
        list_of_tickers = stoxx600
elif list_of_tickers == 'Tel Aviv 125':
        list_of_tickers = ta125
elif list_of_tickers == 'Cotation Assistée en Continu 40':
        list_of_tickers = cac40
elif list_of_tickers == 'China CSI 300 Index 沪深300':
        list_of_tickers = csi300
elif list_of_tickers == 'ALL':
        list_of_tickers = stoxx600 + sp500 + ta125 + cac40 + csi300

#Getting stock data
dict_full = get_stock_data(list_of_tickers)

#Deleting stocks that have no data 
#sidenote: we are losing some tickers because the STOXX600 website is a bit old, but for the others should be ok.

delete = []
for i in dict_full.keys():
    if type(dict_full[i]) != dict:
        delete.append(str(i))
for i in delete:
    del dict_full[i]

#Yahoo finance data
delete = ['currency', 'forwardPE', 'beta']
for key in dict_full.keys():
    for i in delete:
        try:
            del dict_full[key]['summaryDetail'][i]
        except KeyError:
            pass
        
#Getting ratio for the analysis
ratios = ['PE', 'Beta', 'PB', 'EPS', 'PEG', 'EVR', 'EV/EBITDA']
ratio = st.selectbox('Choose wanted ratio.', ratios)
if ratio == 'PE':
    ratio = 'forwardPE'
elif ratio == 'Beta':
    ratio = 'beta'
elif ratio == 'PB':
    ratio = 'priceToBook'
elif ratio == 'EPS':
    ratio = 'forwardEps'
elif ratio == 'PEG':
    ratio = 'pegRatio'
elif ratio == 'EVR':
    ratio = 'enterpriseToRevenue'
elif ratio == 'EV/EBITDA':
    ratio = 'enterpriseToEbitda'

#Creating the data frame to start the analysis
df = pd.DataFrame.from_dict({(i,j): dict_full[i][j] 
                           for i in dict_full.keys() 
                           for j in dict_full[i].keys()},
                        orient='index')
df = df[['longName', 'industry', 'website', 'regularMarketPrice', 'twoHundredDayAverage', ratio]]
df.reset_index(level=1, drop=True, inplace=True)
df = df.stack().unstack()
df.reset_index(inplace=True, names='ticker')
df[ratio] = pd.to_numeric(df[ratio])
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=[ratio], how='all', inplace=True)
df.drop(df.index[df[ratio] < 0], inplace=True)

#Checking for outliers
def is_outlier(s):
    lower_limit = 0
    upper_limit = s.mean() + (s.std() * 2)
    return ~s.between(lower_limit, upper_limit)

df = df[~df.groupby('industry', group_keys=False)[ratio].apply(is_outlier)]

#Creating industry rankings
df['industry_' + ratio] = df[ratio].groupby(df['industry']).transform('mean')
df['disc_' + ratio] = df[ratio] / df['industry_' + ratio] - 1

#Creating dataframe for 'undervalued' stocks
df_uv = df.loc[df.groupby('industry')['disc_' + ratio].idxmin()]
df_uv.sort_values(by='disc_' + ratio, ascending=True, inplace=True)

#Printing results from the analysis
if st.button('Get the results'):
    st.subheader('Undervalued stocks from the index chosen')
    st.dataframe(df_uv)

