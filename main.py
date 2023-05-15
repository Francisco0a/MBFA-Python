#Python final project

#Required modules for the app
import streamlit as st
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from yahooquery import Ticker
import matplotlib.pyplot as plt

#Configuring page
st.set_page_config(layout='wide')
st.title('Stock Screener: Find Undervalued Stocks for Different Indexes:money_with_wings::chart:')
st.header('Identify Undervalued Stocks for Your Portfolio')

expander = st.expander('About this app')
expander.markdown('''Data Science in Finance (Python) 
* **Final project** 
* **Members:** SALONEN, Samuli, ALVAREZ, Francisco & VIDAL, Laura
* **References:** [Yahoo Finance](https://www.kaggle.com/code/omkaarlavangare/content-based-recommendation/notebook),
    [Streamlit library](https://docs.streamlit.io/library/api-reference):''')

st.subheader('Choose the index of your preference in order to get its individual undervalued stocks.')
st.caption('Disclaimer: This is not an investment advice, but it is a useful tool to make informed decisions for potentially undervalued stocks')

#Function to import yahoo data using the unofficial api

def get_stock_data(tickers):
    data = Ticker(tickers).get_modules('assetProfile price summaryDetail defaultKeyStatistics') 
    #assetProfile : industry , price : longName & regularMarketPrice , summaryDetail : twoHundredDayAverage , 
    #defaultKeyStatistics : forwardPE, beta , priceToBook , forwardEps , pegRatio , enterpriseToRevenue , enterpriseToEbitda
    return data

#Getting tickers from Wikipedia for the S&P500
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
#Python reads this as a series but we want it as a list
sp500 = sp500.to_list()

#Getting tickers from Wikipedia for the TA-125
ta125 = pd.read_html('https://en.wikipedia.org/wiki/TA-125_Index#Constituents')[1]['Symbol']
ta125 = ta125.to_list()
ta125 = [item + '.TA' for item in ta125]

#Getting tickers from Wikipedia for the CAC
cac40 = pd.read_html('https://en.wikipedia.org/wiki/CAC_40#Composition')[4]['Ticker']
cac40 = cac40.to_list()

#Getting tickers from Wikipedia for the NZX 50
nzx50 = pd.read_html('https://en.wikipedia.org/wiki/S%26P/NZX_50_Index#Constituents')[1]['Ticker symbol']
nzx50 = nzx50.to_list()

#Getting tickers from Wikipedia for the CSI 300 Index
csi300 = pd.read_html('https://en.wikipedia.org/wiki/CSI_300_Index#Constituents')
csi300 = csi300[3]  #Selecting Table 4
csi300 = csi300['Index'].tolist()
#Since the values of the list are numeric we need to change them to strings
csi300 = [str(num) for num in csi300]
csi300 = [item + '.SS' for item in csi300]

#Creating a DataFrame for the Stoxx600
stoxx600 = pd.DataFrame(columns=['Ticker', 'Exchange'])

#Getting tickets from Dividendmax
        #why is it 20
for i in range(1, 20):
    url = 'https://www.dividendmax.com/market-index-constituents/stoxx600.html/?page=' + str(i)
    data = pd.read_html(url)[0][['Ticker', 'Exchange']]
    stoxx600 = pd.concat([stoxx600, data])

for i in range(0, len(stoxx600)):
    if any(x in stoxx600.iloc[i]['Exchange'] for x in ['Frankfurt', 'Xetra', 'Berlin', 'Luxembourg']):
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.DE'
    elif 'London' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.L'
    elif 'Italian' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.MI'
    elif 'Amsterdam' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.AS'
    elif 'Stockholm' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.ST'
    elif 'Swiss' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.SW'
    elif 'Paris' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.PA'
    elif 'Brussels' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.BR'
    elif any(x in stoxx600.iloc[i]['Exchange'] for x in ['Madrid', 'Valencia', 'Barcelona']):
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.MC'
    elif 'Irish' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.L'
    elif 'Oslo' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.OL'
    elif 'Copenhagen' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.CO'
    elif 'Vienna' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.VI'
    elif 'Lisbon' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.LS'
    elif 'Warsaw' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.WA'
    elif 'Athens' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.AT'
    elif 'Helsinki' in stoxx600.iloc[i]['Exchange']:
        stoxx600.iloc[i] = stoxx600.iloc[i] + '.HE'

stoxx600 = stoxx600['Ticker'].to_list()
stoxx600 = [x.replace('..', '.') for x in stoxx600]
stoxx600 = [x.replace(' ', '-') for x in stoxx600]
stoxx600 = [x.replace('.', '-', 1) if x.count('.')==2 else x for x in stoxx600]

#Creating Dataframe with all the tickers

list_of_tickers = st.selectbox('Choose investment universe.', 
                               ['Standard and Poors 500', 
                                'STOXX Europe 600', 
                                'Tel Aviv 125',
                                'Cotation Assistée en Continu 40', 
                                'NZSX 50 Index',
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
elif list_of_tickers == 'NZSX 50 Index':
        list_of_tickers = nzx50
elif list_of_tickers == 'China CSI 300 Index 沪深300':
        list_of_tickers = csi300
elif list_of_tickers == 'ALL':
        list_of_tickers = stoxx600 + sp500 + ta125 + cac40 + nzx50 + csi300

#Getting stock data
dict_full = get_stock_data(list_of_tickers)

#Deleting stocks that have no data 
#sidenote: we are losing some tickers because the STOXX600 website is a bit old, 
#some stocks have changed exchanges etc.

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


#Creatingthe data frame to start the analysis

df = pd.DataFrame.from_dict({(i,j): dict_full[i][j] 
                           for i in dict_full.keys() 
                           for j in dict_full[i].keys()},
                        orient='index')
df = df[['longName', 'industry','website', 'currency', 'regularMarketPrice', 'twoHundredDayAverage', ratio, ]]
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

df_uv

#Creating a pie chart to show how much money would be necessary to buy all the undervalued stocks, with currency
sum_curr = df_uv.groupby('currency')['regularMarketPrice'].sum()

labels = sum_curr.index
values = sum_curr.values

fig, ax = plt.subplots()

ax.bar(labels, values)
ax.set_xlabel('Currency')
ax.set_ylabel('Total Value')
ax.set_title('Money needed to buy every stock that is undervalued')

for i, v in enumerate(values):
    ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')

#Rotate the x-axis labels if needed
plt.xticks(rotation=45)

#Display the bar chart in Streamlit app
st.pyplot(fig)

