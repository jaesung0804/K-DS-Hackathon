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
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
#%%
companys = {'반도체':['삼성전자', 'sk하이닉스'],
            '바이오':['삼성바이오', '셀트리온'],
            '2차전지':['삼성sdi', 'lg화학'],
            '통신사':['KT', 'SKT', "LG유플러스"],
            '자동차':['현대차', '기아'],
            'IT':['네이버', '카카오'],
            '지수':['코스피', '닛케이지수', '나스닥', '상해종합지수']}
companys = sum(list(companys.values()), [])
#%%
# test
start_date = datetime.datetime(2022,11,1)
# start_date = datetime.datetime(2022,10,31)
end_date = datetime.datetime(2023,10,31)
# selected_company = 'lg화학'

max_features = 10
topk = 50

directory = "./assets/eval"
if not os.path.exists(directory):
    os.makedirs(directory)

for selected_company in companys:
    
    with open(f"./assets/eval/topbottom_{selected_company}.txt", "w") as f:

        df = pd.read_csv(f"./data/trend_data/trend_news_{selected_company}.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        df_new = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        df_new["score"] = df_new["Predicted_Without"].apply(lambda x: abs(x))
        df_ = df_new.sort_values("score")
        df_ = df_.dropna()
        
        top_list = " ".join(df_.head(topk)["news"].to_list())
        tfidf_vectorizer = TfidfVectorizer(max_features = max_features).fit([top_list]) 
        tfidf_matrix = tfidf_vectorizer.fit_transform([top_list])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten().tolist()
        word_scores = zip(feature_names, scores)
        sorted_word_scores_top = sorted(word_scores, key = lambda x: x[1], reverse = True) 
        
        f.write("TOP\n")
        for x, y in sorted_word_scores_top:
            # f.write(f"{x} : {y:.3f}\n")
            f.write(f"{x}, ")
        
        bottom_list = " ".join(df_.tail(topk)["news"].to_list())
        tfidf_vectorizer = TfidfVectorizer(max_features = max_features).fit([bottom_list]) 
        tfidf_matrix = tfidf_vectorizer.fit_transform([bottom_list])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten().tolist()
        word_scores = zip(feature_names, scores)
        sorted_word_scores_bot = sorted(word_scores, key = lambda x: x[1], reverse = True) 
        
        f.write("\nBOTTOM\n")
        for x, y in sorted_word_scores_bot:
            # f.write(f"{x} : {y:.3f}\n")
            f.write(f"{x}, ")
#%%