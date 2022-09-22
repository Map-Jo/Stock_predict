import json
import pandas as pd
import streamlit as st
import pandas_datareader as pdr
import plotly.graph_objects as go
import plotly.express as px
import urllib.request
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
import datetime as dt
from bs4 import BeautifulSoup as bs
from pykrx import stock
from PIL import Image
import koreanize_matplotlib
import requests

st.set_page_config(
    page_title="반포자이까지 한걸음",
    page_icon= "chart_with_upwards_trend",
    layout="wide",
)

with st.sidebar:
    choose = option_menu("App Gallery", ["About", "Today\'s Korea Stock Market", "Today\'s US Stock Market", "Predict Korea Stocks", "Predict US Stocks", 'Caution'],
                         icons=['house','graph-up-arrow', 'graph-up', 'hurricane','hypnotize', 'exclamation-diamond-fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )
    
logo = Image.open('data/stockcode.jpg')

if choose == "About":
    col1, col2 = st.columns( [0.8,0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #000000;} 
        </style> """, unsafe_allow_html=True)
        st.title('나와 함께 반포 자이에 살아보지 않겠어요?')    
    with col2:               # To display brand log
        st.text(' ')

    st.markdown('<p class="font">Hello!\n\n저희는 **반포자이까지 한걸음** 입니다.\n\n저희는 *부족한 투자 지식*으로 인한 *투자손실*을 예방하고자 최적의 **포트폴리오**를 제공하고, 내일 예상 **주가를 예측**할 수 있는 사이트입니다.\n\n많이 부족하지만 **재미로만** 봐주시기를 부탁드립니다.</p>', unsafe_allow_html=True)

    image = Image.open('data/stockcode.jpg')
    st.image(image, width=800, caption= 'The Great GATSBY')

elif choose == "Today\'s Korea Stock Market":
    col1, col2 = st.columns( [0.8,0.2])
    with col1:              
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font"> Today\'s Korea Stock Market!</p>', unsafe_allow_html=True)    
    with col2:
        st.text(' ')

    st.title('Korea Stocks 📈')
    Stockcode = pd.read_csv('data/stockcode_pdr.csv')
    name_list = Stockcode['name'].tolist()
    name_list.insert(0, '')
    choice = st.selectbox('검색하실 주식 종목명을 입력해 주세요.',name_list)


    for i in range(len(name_list)):
        if choice == name_list[i]:
            choice_name = Stockcode.loc[Stockcode['name'] == name_list[i], 'name'].values
            choice_name_to_str =np.array2string(choice_name).strip("[]")
            Name = choice_name_to_str.strip("''")



    Stockcode.set_index('name', inplace=True)
    Code_name_list = Stockcode.index.tolist()

    with st.spinner('Wait for it...'):
        if Name in Code_name_list:
            code_num = Stockcode.at[name, 'code']
            df = pdr.get_data_yahoo(code_num)
            df['Change'] = df['Close'].tail(2)[0] - df['Close'].tail(2)[1]
            col1, col2, col3 = st.columns(3)
            col1.metric("현재 주식가격",format(df['Close'].tail(1)[0], ',')+'원', "%d원" %(df['Close'].diff().tail(1)[0]))
            col2.metric("현재 거래량", format(df['Volume'].tail(1)[0], ','),"%.2f%%" %(df['Volume'].pct_change().tail(1)[0] * 100))
            col3.metric("전일 대비 가격", "%d원" %(df['Close'].diff().tail(1)[0]), "%.2f%%" %(df['Change'].tail(1)[0] * 100))

            fig = px.line(df, y='Close', title='{} 종가 Time Series'.format(Name))

            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure(data=[go.Candlestick(x=df.index,
                        open=df['Open'],
                        high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            increasing_line_color = 'tomato',
                            decreasing_line_color = 'royalblue',
                            showlegend = False)])

            fig2.update_layout(title='{} Candlestick chart'.format(Name))
            st.plotly_chart(fig2, use_container_width=True)

        elif Name not in Code_name_list:
            st.text('검색하신 주식 종목이 없습니다. 정확하게 입력해주세요.')

elif choose == "Today\'s US Stock Market":
   
    col1, col2 = st.columns( [0.8,0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font"> Today\'s US Stock Market!</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.text(' ')    



    st.title('US Stocks 📈')


    page = urllib.request.urlopen("https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=%ED%99%98%EC%9C%A8")
    text = page.read().decode("utf8")

    where = text.find('class="grp_info"> <em>')
    start_of_time = where + 22
    end_of_time = start_of_time + 16
    prin = text[start_of_time:end_of_time]

    usdwhere = text.find('<span>미국 <em>USD</em></span></a></th> <td><span>')
    usdletter =  text[usdwhere+48] + text[usdwhere+50:usdwhere+56]

    Stockcode = pd.read_csv('data/oversea_stockcode.csv')
    Stockcode['ticker'] = Stockcode['Symbol'].copy()
        # Name = st.text_input('Code Name', placeholder='미국 주식의 ticker를 입력해주세요.').upper()
    name_list = Stockcode['Symbol'].tolist()
    name_list.insert(0, '')
    choice = st.selectbox('검색하실 미국 주식 종목의 Ticker를 입력해 주세요.',name_list)
    # with st.spinner('Predicting...'):
    for i in range(len(name_list)):
        if choice == name_list[i]:
            choice_name = Stockcode.loc[Stockcode['Symbol'] == name_list[i], 'Symbol'].values
            choice_name_to_str =np.array2string(choice_name).strip("[]")
            Name = choice_name_to_str.strip("''")

    Stockcode.set_index('Symbol', inplace=True)
    Code_name_list = Stockcode.index.tolist()
    if Name in Code_name_list:
        # code_num = Stockcode.at[Name, 'ticker']
        # data = fdr.DataReader(code_num)   

        with st.spinner('Wait for it...'):
            if Name in Code_name_list:
                code_num = Stockcode.at[Name, 'ticker']
                df = fdr.DataReader(code_num)
                money = df['Close'].tail(1)
                k_money = float(money)*float(usdletter)
                k_money = round(k_money,2)
                k_money = format(k_money, ',')

                col1, col2, col3 = st.columns(3)
                col1.metric("현재 주식가격",format(df['Close'].tail(1)[0], ',')+'$', "%s원" %k_money)
                col2.metric("현재 거래량", format(round(df['Volume'].tail(1)[0]), ','),"%.2f%%" %(df['Volume'].pct_change().tail(1)[0] * 100))
                col3.metric("전일 대비 가격", "%d$" %(df['Close'].diff().tail(1)[0]), "%.2f%%" %(df['Change'].tail(1)[0] * 100))

                fig = px.line(df, y='Close', title='{} 종가 Time Series'.format(Name))

                fig.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

                fig2 = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            increasing_line_color = 'tomato',
                            decreasing_line_color = 'royalblue',
                            showlegend = False)])

                fig2.update_layout(title='{} Candlestick chart'.format(Name))
                st.plotly_chart(fig2, use_container_width=True)

                st.text(prin +'의 KEB하나은행 환율정보 입니다.')
                st.text('현재 1$당 '+str(usdletter)+'원 입니다.')
            elif Name not in Code_name_list:
                st.text('검색하신 주식 종목이 없습니다. 정확하게 입력해주세요.')

elif choose == "Predict Korea Stocks":
    col1, col2 = st.columns( [0.8,0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font"> Predict Tomorrow\'s Korea Stocks!</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.text(' ')


    st.title('국내주식 종목의 주가를 예측해 보세요 📈')


    Stockcode = pd.read_csv('data/Stockcode.csv')

    name_list = Stockcode['Name'].tolist()
    name_list.insert(0, '')
    choice = st.selectbox('검색하실 주식 종목명을 입력해 주세요.',name_list)


    for i in range(len(name_list)):
        if choice == name_list[i]:
            choice_name = Stockcode.loc[Stockcode['Name'] == name_list[i], 'Name'].values
            choice_name_to_str =np.array2string(choice_name).strip("[]")
            Name = choice_name_to_str.strip("''")



    Stockcode.set_index('Name', inplace=True)
    Code_name_list = Stockcode.index.tolist()
    if Name in Code_name_list:
        code_num = Stockcode.at[Name, 'Symbol']
        data = fdr.DataReader(code_num)
        with st.spinner('Predicting...'):
            if data.shape[0] >= 60:
                startdate = (dt.datetime.now()-dt.timedelta(days=31)).strftime('%Y-%m-%d')
                enddate = dt.datetime.now().strftime('%Y-%m-%d')
                data_ = data.loc[startdate:enddate]
                close = data_['Close']
                base = (close - close.min()) / (close.max() - close.min())
                window_size = len(base)
                next_date = 5
                moving_cnt = len(data) - window_size - next_date - 1
                def cosine_similarity(x, y):
                    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

                sim_list = []

                for i in range(moving_cnt):
                    target = data['Close'].iloc[i:i+window_size]
                    target = (target - target.min()) / (target.max() - target.min())
                    cos_similarity = cosine_similarity(base, target)
                    sim_list.append(cos_similarity)

                top = pd.Series(sim_list).sort_values(ascending=False).head(1).index[0]

                idx=top
                target = data['Close'].iloc[idx:idx+window_size+next_date]
                target = (target - target.min()) / (target.max() - target.min())

                fig = plt.figure(figsize=(20,5))
                plt.plot(base.values, label='base', color='grey')
                plt.plot(target.values, label='target', color='orangered')
                plt.xticks(np.arange(len(target)), list(target.index.strftime('%Y-%m-%d')), rotation=45)
                plt.axvline(x=len(base)-1, c='grey', linestyle='--')
                plt.axvspan(len(base.values)-1, len(target.values)-1, facecolor='ivory', alpha=0.7)
                plt.legend()
                st.pyplot(fig)

                period=5
                preds = data['Change'][idx+window_size: idx+window_size+period]
                cos = round(float(pd.Series(sim_list).sort_values(ascending=False).head(1).values), 2)
                st.markdown(f'현재 주식 상황과 **{cos} %** 유사한 시기의 주식 상황입니다.')
                future = round(preds.mean()*100, 2)
                if future > 0:
                    st.markdown(f'위의 주식 상황을 바탕으로 앞으로 5일동안 **{Name}** 주식은 평균 **{future}%** 상승할 것으로 보입니다.')
                elif future < 0:
                    st.markdown(f'위의 주식 상황을 바탕으로 앞으로 5일동안 **{Name}** 주식은 평균 **{future}%** 하락할 것으로 보입니다.')

                pred = preds[0]
                predict = data['Close'].tail(1).values * pred #8월 17일꺼에 떨어질 확률 곱하면 0.1이면 1000원 일 때 100원으로 계산 됨. -0.1이면 -100으로 계산 됨
                yesterday_close = data['Close'].tail(1).values #8월 17일꺼


                if pred > 0:
                    plus_money = yesterday_close + predict
                    plus_money = format(int(plus_money), ',')
                    st.markdown(f'내일 **{Name}** 주식은 **{round(pred*100,2)} %** 상승할 예정이고, 주가는 **{plus_money}원**으로 예상됩니다.')
                elif pred < 0:
                    minus_money = yesterday_close + predict
                    minus_money = format(int(minus_money), ',')
                    st.markdown(f'내일 **{Name}** 주식은 **{round(pred*100,2)} %** 하락할 예정이고, 주가는 **{minus_money}원**으로 예상됩니다.')
                else:
                    st.markdown(f'내일 **{Name} 주식은 변동이 없을 것으로 예상됩니다.')
            elif data.shape[0] < 60:
                st.markdown(f'**{Name}**은 최근에 상장한 주식으로 예상됩니다.')
                st.markdown('예측할 데이터가 부족합니다.')
                st.markdown('충분한 데이터가 모일 때까지 조금만 기다려 주세요.')
                st.markdown('그때 다시 만나요~')

                image = Image.open('data/waitplease.png')
                st.image(image, width=500)
            st.success('Done!')

    elif Name not in Code_name_list:
        st.text('검색하신 주식 종목이 없습니다. 정확하게 입력해주세요.')

elif choose == "Predict US Stocks":
    col1, col2 = st.columns( [0.8,0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font"> Predict Tomorrow\'s US Stocks!</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.text(' ')

    st.title('해외주식 종목의 주가를 예측해 보세요 📈')



    page = urllib.request.urlopen("https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=%ED%99%98%EC%9C%A8")
    text = page.read().decode("utf8")

    where = text.find('class="grp_info"> <em>')
    start_of_time = where + 22
    end_of_time = start_of_time + 16
    prin = text[start_of_time:end_of_time]

    usdwhere = text.find('<span>미국 <em>USD</em></span></a></th> <td><span>')
    usdletter =  text[usdwhere+48] + text[usdwhere+50:usdwhere+56]


    Stockcode = pd.read_csv('data/oversea_stockcode.csv')
    Stockcode['ticker'] = Stockcode['Symbol'].copy()
    name_list = Stockcode['Symbol'].tolist()
    name_list.insert(0, '')
    choice = st.selectbox('검색하실 미국 주식 종목의 Ticker를 입력해 주세요.',name_list)

    with st.spinner('Predicting...'):
        for i in range(len(name_list)):
            if choice == name_list[i]:
                choice_name = Stockcode.loc[Stockcode['Symbol'] == name_list[i], 'Symbol'].values
                choice_name_to_str =np.array2string(choice_name).strip("[]")
                Name = choice_name_to_str.strip("''")

        Stockcode.set_index('Symbol', inplace=True)
        Code_name_list = Stockcode.index.tolist()
        if Name in Code_name_list:
            code_num = Stockcode.at[Name, 'ticker']
            data = fdr.DataReader(code_num)
        
            if data.shape[0] >= 60:
                startdate = (dt.datetime.now()-dt.timedelta(days=31)).strftime('%Y-%m-%d')
                enddate = dt.datetime.now().strftime('%Y-%m-%d')
                data_ = data.loc[startdate:enddate]
                close = data_['Close']
                base = (close - close.min()) / (close.max() - close.min())
                window_size = len(base)
                next_date = 5
                moving_cnt = len(data) - window_size - next_date - 1
                def cosine_similarity(x, y):
                    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

                sim_list = []

                for i in range(moving_cnt):
                    target = data['Close'].iloc[i:i+window_size]
                    target = (target - target.min()) / (target.max() - target.min())
                    cos_similarity = cosine_similarity(base, target)
                    sim_list.append(cos_similarity)

                top = pd.Series(sim_list).sort_values(ascending=False).head(1).index[0]

                idx=top
                target = data['Close'].iloc[idx:idx+window_size+next_date]
                target = (target - target.min()) / (target.max() - target.min())

                fig = plt.figure(figsize=(20,5))
                plt.plot(base.values, label='base', color='grey')
                plt.plot(target.values, label='target', color='orangered')
                plt.xticks(np.arange(len(target)), list(target.index.strftime('%Y-%m-%d')), rotation=45)
                plt.axvline(x=len(base)-1, c='grey', linestyle='--')
                plt.axvspan(len(base.values)-1, len(target.values)-1, facecolor='ivory', alpha=0.7)
                plt.legend()
                st.pyplot(fig)

                money = data['Close'].tail(1)
                k_money = float(money)*float(usdletter)
                k_money = round(k_money,2)


                period=5
                preds = data['Change'][idx+window_size: idx+window_size+period]
                cos = round(float(pd.Series(sim_list).sort_values(ascending=False).head(1).values), 2)
                st.markdown(f'현재 주식 상황과 **{cos} %** 유사한 시기의 주식 상황입니다.')
                future = round(preds.mean()*100, 2)
                if future > 0:
                    st.markdown(f'위의 주식 상황을 바탕으로 앞으로 5일동안 **{Name}** 주식은 평균 **{future}%** 상승할 것으로 보입니다.')
                elif future < 0:
                    st.markdown(f'위의 주식 상황을 바탕으로 앞으로 5일동안 **{Name}** 주식은 평균 **{future}%** 하락할 것으로 보입니다.')

                pred = preds[0]
                predict = data['Close'].tail(1).values * pred
                yesterday_close = data['Close'].tail(1).values
                k_yesterday = k_money

                if pred > 0:
                    plus_money = yesterday_close + predict
                    plus_money = format(int(plus_money), ',')
                    k_plus_money = k_yesterday + predict
                    k_plus_money = format(int(k_plus_money), ',')
                    st.markdown(f'내일 **{Name}** 주식은 **{round(pred*100,2)} %** 상승할 예정이고, 주가는 **{plus_money}$ ({k_plus_money}원)**으로 예상됩니다.')

                elif pred < 0:
                    minus_money = yesterday_close + predict
                    minus_money = format(int(minus_money), ',')
                    k_minus_money = k_yesterday + predict
                    k_minus_money = format(int(k_minus_money), ',')
                    st.markdown(f'내일 **{Name}** 주식은 **{round(pred*100,2)} %** 하락할 예정이고, 주가는 **{minus_money}$ ({k_minus_money}원)**으로 예상됩니다.')
                else:
                    st.markdown(f'내일 **{Name} 주식은 변동이 없을 것으로 예상됩니다.')
                
                st.text(prin +'의 KEB하나은행 환율정보 입니다.')
                st.text('현재 1$당 '+str(usdletter)+'원 입니다.')

            elif data.shape[0] < 60:
                st.markdown(f'**{Name}**은 최근에 상장한 주식으로 예상됩니다.')
                st.markdown('예측하기에는 데이터가 부족합니다.')
                st.markdown('충분한 데이터가 모일 때까지 조금만 기다려 주세요.')
                st.markdown('그때 다시 만나요~')

                image = Image.open('data/waitplease.png')
                st.image(image, width=500)

            st.success('Done!')

        elif Name not in Code_name_list:
            st.text('검색하신 주식 종목이 없습니다. 정확하게 입력해주세요.')
            
elif choose == 'Caution':
    st.markdown('### 증권투자시 유의사항')
    st.markdown('* 증권투자는 본인의 판단과 책임으로.')
    st.text(' - 증권투자는 반드시 자기 자신의 판단과 책임하에 하여야 하며, 자신의 여유자금으로 분산투자하는 것이 좋습니다.')
    st.text(' - 증권회사 직원 및 타인에게 매매거래를 위임하더라도 투자손익은 고객 자신에게 귀속되며, 투자원금의 보장 또는 손실보전 약속은 법률적으로 효력이 없습니다.')
    st.markdown('* 높은 수익에는 높은 위험이.')
    st.text(' - 높은 수익에는 반드시 높은 위험이 따른다는 것을 기억하고 투자시 어떤 위험이 있는지 반드시 확인하시기 바랍니다.')
    st.text(' - 주식워런트 증권(ELW), 선물·옵션 거래는 단기간에 투자금 전부를 손실 볼 수 있으며 특히 선물·옵션 거래는 투자금을 초과하여 손실 볼 수 있으므로 거래설명서를 교부받고 거래제도와 특성, 거래에 따른 위험 등을 반드시 숙지하여야 합니다.')

    st.markdown('### 간접투자상품 투자시 유의사항')
    st.markdown('* 원금이나 수익이 보장되지 않습니다.')
    st.text(' - 수익증권, 일임형 랩(Wrap) 등 간접투자상품은 원금보장이 되지 않으며 운용결과에 따라 투자원금의 손실이 발생할 수 있습니다.')
    st.text(' - 간접투자상품의 과거 운용실적이 미래의 수익을 보장하는 것은 아닙니다.')

    st.markdown('* 그 밖에 알아두어야 할 사항')
    st.text(' - 가입한 상품의 기준가격이나 손익현황을 수시로 확인하여 예상되는 시장변동에 대처하여야 합니다.')
    st.text(' - 해외의 유가증권에 투자하는 간접투자상품은 환율의 변동에 따라 가치가 변동될 수 있습니다.')
