#%%
"""
Reference:
[1] https://www.linkedin.com/pulse/question-answer-bot-using-openai-langchain-faiss-satish-srinivasan/
"""
#%%
import os
import json
import re
import torch
import pandas as pd
import numpy as np
import datetime
from tqdm.notebook import tqdm
tqdm.pandas()
from module.embedding import get_embedding
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import openai
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager, rc
import plotly.graph_objs as go
import plotly.express as px
from tenacity import retry, wait_random_exponential, stop_after_attempt
import FinanceDataReader as fdr
import datetime
import platform
#%%
def main():
    # https://huggingface.co/jhgan/ko-sroberta-multitask
    tokenizer = AutoTokenizer.from_pretrained('jhgan/ko-sroberta-multitask')
    model = AutoModel.from_pretrained('jhgan/ko-sroberta-multitask')
    
    directory = "./log"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    csv_file = "./log/input_log.csv"
    if os.path.exists(csv_file):
        input_log = pd.read_csv(csv_file, index_col=False)
    else:
        input_log = pd.DataFrame(
            {'time': np.array([]).astype('datetime64[ns]'),
            'query': np.array([]),
            'sector': np.array([]),
            'company': np.array([]),
            'start date': np.array([]),
            'end date': np.array([]),
            'topk': np.array([]),
            'feedback': np.array([]).astype(int),
            }
        )

    keywords_df = pd.read_csv('./data/trend_data/kewords.csv',encoding='utf-8-sig')
    st.header("""
    키워드 기반의 주요 문장 추출을 통한 요약 시스템
    #### 서울시립대학교 통계데이터사이언스학과 팀 VANILLA 🍦
    """)
    sectors = ['반도체', '바이오', '2차전지', '통신사', '자동차', 'IT', '지수']
    
    st.markdown("""---""")
    selected_sector = st.selectbox('##### 📈 테마를 선택하세요', sectors)
    companys = {'반도체':['삼성전자', 'SK하이닉스'],
                '바이오':['삼성바이오로직스', '셀트리온'],
                '2차전지':['삼성SDI', 'LG화학'],
                '통신사':['KT', 'SK텔레콤', "LG유플러스"],
                '자동차':['현대차', '기아'],
                'IT':['NAVER', '카카오'],
                '지수':['코스피', '니케이지수', '나스닥', '상해종합지수']}
    
    selected_company = st.selectbox('##### 📈 종목을 선택하세요', companys[selected_sector])
    
    company_dict = {
        '삼성바이오로직스': '삼성바이오',
        'SK하이닉스': 'sk하이닉스',
        '삼성SDI': '삼성sdi',
        'LG화학': 'lg화학',
        'NAVER': '네이버',
        'SK텔레콤': 'SKT',
        '니케이지수': '닛케이지수',
    }
    if selected_company in company_dict.keys():
       selected_company = company_dict.get(selected_company)
    keywords_list = keywords_df[selected_company].values.tolist()

    start_date = st.date_input("##### 🗓️ 시작 날짜를 선택하세요", value=datetime.date(2018, 11, 2))
    end_date = st.date_input("##### 🗓️ 종료 날짜를 선택하세요", value=datetime.date(2023, 10, 31))
    start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
    end_date = datetime.datetime.combine(end_date, datetime.datetime.min.time())

    # test
    # selected_sector = '바이오'
    # selected_company = '삼성바이오로직스'
    # query = '반도체'
    # start_date = datetime.datetime(2021,12,2)
    # end_date = datetime.datetime(2023,10,31)
    
    df = pd.read_csv(f"./data/trend_data/trend_news_{selected_company}.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df_new = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    start_idx = df_new.index[0]
    end_idx = df_new.index[-1]
    df_new = df_new.reset_index(drop=True)

    flag = df_new["news"].isna().sum()
    if flag > 0:
        drop_idx = list(df_new[df_new["news"].isna()].index)
        df_new = df_new.drop(index=drop_idx)
    
    file_path = f"./data/trend_data/trend_embedd_database_{selected_company}.pt"
    embeddings = torch.load(file_path)[start_idx:end_idx+1]
    
    if flag > 0:
        embeddings = [e for j, e in enumerate(embeddings) if j not in drop_idx]
    
    assert len(embeddings) == len(df_new)
    
    df_new["news"] = df_new["news"].apply(lambda x: re.sub("fn", "", x))
    
    topk = st.selectbox(
        '##### 키워드를 이용해 추출할 뉴스 기사 헤드라인의 개수', [5, 10, 15, 20, 25, 30], index=1)
    
    # 선택된 옵션 출력
    # st.write('- 선택한 종목:', selected_company)
    # st.write('- 추천 키워드:', ', '.join(keywords_list[:10]))
    st.write('##### 추천 키워드: ' + ' '.join([f"`{x.upper()}`" for x in keywords_list[:10]]))
    
    with st.form('input_form'):
        query = st.text_input(
            "",
            # """
            # 원하는 키워드를 입력하고, [Enter]를 눌러주세요.
            # """,
            label_visibility='collapsed').lower()
        st.form_submit_button('원하는 키워드를 입력하고, [Enter]를 눌러주세요.')
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.form_submit_button('원하는 키워드를 입력하고, [Enter]를 눌러주세요.')
        # with col2:
        #     query = st.text_input(
        #         "",
        #         # """
        #         # 원하는 키워드를 입력하고, [Enter]를 눌러주세요.
        #         # """,
        #         label_visibility='collapsed')

    if query:
        
        if 'summary' not in st.session_state:
            st.session_state.summary = ""
        
        tab1, tab2, tab3 = st.tabs(["Top-K 뉴스 헤드라인", "시각화", "ChatGPT 보정 결과"])
        
        keywords = query
        query_emb = get_embedding(keywords, tokenizer, model)

        score = []
        for i in range(len(df_new)):
            score.append((i, cosine_similarity(embeddings[i], query_emb)[0][0]))

        # ranking
        top_idx = [t[0] for t in sorted(score, key=lambda x: x[1], reverse=True)[:50]] # default
        top_idx = sorted(top_idx)
        
        keywords = selected_company + ' ' + query
        query_emb = get_embedding(keywords, tokenizer, model)
        
        score = []
        for i in top_idx:
            score.append((i, cosine_similarity(embeddings[i], query_emb)[0][0]))
        
        # re-ranking
        top_idx2 = [t[0] for t in sorted(score, key=lambda x: x[1], reverse=True)[:topk]]
        top_idx2 = sorted(top_idx2)
        
        output = ""
        df_new['Date'] = df_new['Date'].astype(str)            
        for i in top_idx2:
            output += "##### [" + df_new['Date'].iloc[i] + "] "
            output += df_new['news'].iloc[i]
            output += " " + f"(중요도: {abs(df_new['Predicted_Without'].iloc[i]): .3f})" + "\n\n"
                    
        # df['news'].iloc[bottom_idx]
        if len(output) == 0 : output = '결과가 없습니다.'
        with tab1:
            st.write("### 원문에서 추출된 중요 문장")
            st.write(output)

        vis_data = {
            'Date': [],
            'News': [],
            'Predicted_Without': []
        } 
        
        # 검색 결과를 vis_data에 추가합니다.
        for i in top_idx2:
            vis_data['Date'].append(df_new['Date'].iloc[i])
            vis_data['News'].append(df_new['news'].iloc[i])
            vis_data['Predicted_Without'].append(df_new['Predicted_Without'].iloc[i])

        
        if platform.system() == 'Windows':
            rc('font', family='Malgun Gothic')
        elif platform.system() == 'Darwin': # Mac
            rc('font', family='AppleGothic')
        else: #linux
            rc('font', family='NanumGothic')
        # font_path = 'C:\Windows\Fonts\MALGUNSL.ttf' 
        # font_name = font_manager.FontProperties(fname=font_path).get_name()
        # rc('font', family=font_name)
                    
        vis_df = pd.DataFrame(vis_data)
        vis_df['Date'] = pd.to_datetime(vis_df['Date'])

        # st.markdown("""---""")
        with tab2:
            st.write("### [그림] 키워드와 관련성이 높은 뉴스 헤드라인")
            
            # Plotly 그래프 생성
            fig = go.Figure()
            
            y_max = max([abs(x) + 0.5 for x in vis_df['Predicted_Without']])

            # 점들을 추가
            for index, row in vis_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['Date']],  # 날짜는 여기서 문자열로 전달합니다.
                    y=[abs(row['Predicted_Without'])],
                    mode='markers',
                    marker=dict(size=12),
                    customdata=[[  # customdata는 각 데이터 포인트에 대한 리스트의 리스트로 구성됩니다.
                        row['Date'].strftime('%Y-%m-%d'),  # 날짜는 문자열로 그대로 전달합니다.
                        row['News'],
                        abs(row['Predicted_Without'])
                    ]],
                    hovertemplate=(
                        "<b>날짜:</b> %{customdata[0]}<br>" +
                        "<b>헤드라인:</b> %{customdata[1]}<br>" +
                        "<b>중요도:</b> %{customdata[2]:.3f}<extra></extra>"  # <extra></extra>는 불필요한 trace 이름을 숨깁니다.
                    ),
                    showlegend=False  # 이 줄을 추가하여 각 trace의 범례 표시를 비활성화합니다.
                ))

            # 레이아웃 설정
            fig.update_traces(
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=18,  # 폰트 크기를 조정합니다.
                    # font_family=font_name,
                    font_color="black"
                )
            )

            fig.update_layout(
                title=dict(
                    # text="[그림] 키워드와 관련성이 높은 뉴스 헤드라인", 
                    text="",
                    font=dict(size=24), 
                    automargin=True),
                xaxis=dict(
                    # title='날짜',
                    # showgrid=True,
                    # gridwidth=1,
                    # gridcolor='LightPink',
                    # griddash='dot',
                    tickfont = dict(size=18)
                ),
                yaxis=dict(
                    title='중요도',
                    zeroline=True,
                    zerolinewidth=1,  # Zero line의 두께
                    zerolinecolor='LightPink',  # Zero line의 색상을 변경합니다.
                    range=(0, y_max),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightPink',
                    griddash='dot',
                    tickfont = dict(size=18)
                ),
                hovermode='closest',
                showlegend=False  # 이 줄을 추가하여 전체 그래프의 범례를 숨깁니다.
            )

            # Streamlit에 그래프를 표시
            st.plotly_chart(fig, use_container_width=True)

        # st.markdown("""---""")
        with tab3:
            st.write("### ChatGPT를 이용한 결과 보정")

            # 요약문을 표시할 영역
            summary_placeholder = st.empty()

            vis_df['Date'] = vis_df['Date'].astype(str)
            
            if st.button('요약하기'):

                # OpenAI 클라이언트 초기화
                client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                            
                # 뉴스 헤드라인 요약 요청 문자열 생성
                news_for_summary = "\n".join(vis_df['Date'] + ": " + vis_df['News'])
                prompt = """
                Please generate an appropriate response to the given requirements and instructions.\n
                Requirements and Instructions : \n
                1. You must transform news headlines in the form of several paragraphs in korean. \n
                2. Only one year should be included in one paragraph. 
                3. Everything about each year should be included in the corresponding paragraph.
                    For example, your response should be in the form of "In 2019, there was an event called a, and there was also an event called b." \n
                4. All given news headlines for a year must be included in the corresponding paragraph. \n
                5. All news headlines should be included in your reponse.
                6. Facts must not be changed. \n
                7. Don't mention a specific date, just mention the year. \n
                """+ "\nData and News headlines : \n" + news_for_summary

                @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(100))
                def completions_with_backoff(**kwargs):
                    return client.chat.completions.create(**kwargs)
                # @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
                # def completions_with_backoff(**kwargs):
                #     return openai.ChatCompletion.create(**kwargs)
                response = completions_with_backoff(
                    model="gpt-3.5-turbo-1106", 
                    messages=[
                        {
                            "role": "system",
                            "content": "You're a helpful assistant that transforms news headlines in the form of several paragraphs in korean.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=0.5)
                
                # 요약 결과 표시
                summary = response.choices[0].message.content.strip()
                st.session_state.summary = summary
            # st.write(summary)
            summary_placeholder.text_area("", st.session_state.summary, height=400)
                
    else:
        st.write('검색된 뉴스 기사가 없습니다.')
    
    st.markdown("""---""")
    st.write(
    """
    요약 결과를 평가해주세요. 여러분의 소중한 의견을 부탁드립니다.
    """
    )
    
    col1, col2, col3 = st.columns([0.15, 0.15, 0.7])
    with col1:
        like_button = st.button("좋아요 👍")
    with col2:
        dislike_button = st.button("싫어요 👎")
    with col3:
        st.empty()
         
    if like_button:
        timestamp = datetime.datetime.now()
        input_log.loc[len(input_log)] = [
            timestamp, query, selected_sector, selected_company, start_date, end_date, topk, 1] # like
        input_log.to_csv(csv_file, index=False)
    elif dislike_button:
        timestamp = datetime.datetime.now()
        input_log.loc[len(input_log)] = [
            timestamp, query, selected_sector, selected_company, start_date, end_date, topk, 0] # dislike
        input_log.to_csv(csv_file, index=False)
    # else:
    #     st.markdown("*로그 기록을 위해 제출을 눌러주세요!")

    st.markdown("""---""")
    st.write("###### MADE BY 안승환, 박기정, 박재성, 우경동, 임재성")

#%%
if __name__ == "__main__":
    main()


# please write above headlines in the form of several paragraphs in korean.
