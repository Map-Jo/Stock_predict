import pandas as pd
import streamlit as st
import FinanceDataReader as fdr
import plotly.graph_objects as go
import plotly.express as px
import FinanceDataReader as fdr
import urllib.request
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
import datetime
from PIL import Image


with st.sidebar:
    choose = option_menu("App Gallery", ["About", "Today\'s Korean Stock", "Today\'s American Stock", "Predict Korean Stock", "Predict American Stock"],
                         icons=['house', 'graph-up-arrow', 'graph-up', 'hurricane','hypnotize'],
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
        font-size:35px ; font-family: 'Cooper Black'; color: #330000;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font"> Do you want to be a rich?</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.text(' ')

    st.markdown('<p class="font">Hello!\n\n저희는 **반포자이까지 한걸음** 입니다.\n\n저희는 부족한 투자 지식으로 인한 투자손실을 막고자 최적의 포트폴리오를 제공하고, 내일 예상 주가를 예측할 수 있는 사이트입니다.\n\n많이 부족하지만 재미로만 봐주시기를 부탁드립니다.</p>', unsafe_allow_html=True)

    image = Image.open('data/stockcode.jpg')
    st.image(image, width=800, caption= 'The Great GATSBY')

elif choose == "Today\'s Korean Stock":
    col1, col2 = st.columns( [0.8,0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font"> Today\'s Korean Stock!</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.text(' ')

    st.title('Local Stocks 📈')
    Stockcode = pd.read_csv('data/Stockcode.csv')
    Stockcode.set_index('Name', inplace=True)
    Name = st.text_input('Code Name',placeholder= '종목명을 입력해 주세요.').upper()
    Code_name_list = Stockcode.index.tolist()

    with st.spinner('Wait for it...'):
        if Name in Code_name_list:
            code_num = Stockcode.at[Name, 'Symbol']
            df = fdr.DataReader(code_num)
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

elif choose == "Today\'s American Stock":
   
    col1, col2 = st.columns( [0.8,0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font"> Today\'s American Stock!</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.text(' ')    



    st.title('Overseas Stocks 📈')


    page = urllib.request.urlopen("https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=%ED%99%98%EC%9C%A8")
    text = page.read().decode("utf8")

    where = text.find('class="grp_info"> <em>')
    start_of_time = where + 22
    end_of_time = start_of_time + 16
    prin = text[start_of_time:end_of_time]

    usdwhere = text.find('<span>미국 <em>USD</em></span></a></th> <td><span>')
    usdletter =  text[usdwhere+48] + text[usdwhere+50:usdwhere+56]


    Stockcode = pd.read_csv('data/oversea_stockcode.csv')
    Stockcode.set_index('Symbol', inplace=True)
    Name = st.text_input('Code Name', placeholder='ticker를 입력해주세요.').upper()
    Code_name_list = Stockcode.index.tolist()
    Stockcode['ticker'] = Stockcode.index
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
            st.text('검색하신일거 같아 주식 종목이 없습니다. 정확하게 입력해주세요.')

elif choose == "Predict Korean Stock":
    col1, col2 = st.columns( [0.8,0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font"> Predict Tomrorow\'s Korean Stock!</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.text(' ')


    st.title('국내주식 종목의 주가를 예측해 보세요 📈')


    Stockcode = pd.read_csv('data/Stockcode.csv')

    name_list = Stockcode['Name'].tolist()
    name_list.insert(0, '<종목명을 입력해 주세요.>')
    choice = st.selectbox('Search',name_list)


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
                startdate = (datetime.datetime.now()-datetime.timedelta(days=31)).strftime('%Y-%m-%d')
                enddate = datetime.datetime.now().strftime('%Y-%m-%d')
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

elif choose == "Predict American Stock":
    col1, col2 = st.columns( [0.8,0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font"> Predict Tomrorow\'s American Stock!</p>', unsafe_allow_html=True)    
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
    # Name = st.text_input('Code Name', placeholder='미국 주식의 ticker를 입력해주세요.').upper()
    name_list = Stockcode['Symbol'].tolist()
    name_list.insert(0, '<검색하실 종목의 Ticker를 입력해주세요.>')
    choice = st.selectbox('Search',name_list)


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
        with st.spinner('Predicting...'):
            if data.shape[0] >= 60:
                startdate = (datetime.datetime.now()-datetime.timedelta(days=31)).strftime('%Y-%m-%d')
                enddate = datetime.datetime.now().strftime('%Y-%m-%d')
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
