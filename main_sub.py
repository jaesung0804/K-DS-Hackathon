#%%
import os
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-21\bin\server'
print('JAVA_HOME' in os.environ) # konlpy 사용을 위한 설정

import pandas as pd
import numpy as np
import tqdm
import importlib
import random
import re


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from tokenizers import BertWordPieceTokenizer
from tokenizers import SentencePieceBPETokenizer
import FinanceDataReader as fdr
from module.embedding import get_embedding
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from TF_IDF import tf_idf
import statistics
import math

#%%
'''주식 분야와 해당 회사들 설정'''
companys = {'반도체':['삼성전자', 'sk하이닉스'], 
                'IT':['네이버', '카카오'], 
                '자동차':['현대차', '기아'],
                '통신사':['SKT', 'KT', 'LG유플러스'],
                '바이오':['삼성바이오', '셀트리온'],
                '2차전지':['lg화학', '삼성sdi'],
                '지수':['코스피', '나스닥', '상해종합지수', '닛케이지수']}

sector = "IT"
stocks = companys[sector]


#%%
"""seed 부여 및 hyperparameters 설정"""
config = {
    "seed": 42,
    
    "batch_size": 64,
    "head_size": 4,
    
    "epochs": 500,
    "lr": 0.005,

    "num_keywords" : 20,
    "quantile" : 2,
    
    "output_details" : 'tmp' # output 결과 상세 설명 : 파일 디렉토리명으로 설정됨
}

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
config["cuda"] = device
# torch.manual_seed(config["seed"])
def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  
setup_seed(config["seed"])
#%%
"""data import"""

"""2018-11-01 ~ 2023-10-31"""

def load_data(sector : str, 
              stocks : list):
    data_dict = {}
        
    for stock in stocks:
        # report_path = f"./data/report_data_{sector}_{stock}.csv"
        news_path = f"./data/news_data_{sector}_{stock}.csv"
        news1_path = f"./data/news_data_{sector}_{stock}_2023.11.20.csv"
        
        # report = pd.read_csv(report_path).dropna()
        news = pd.read_csv(news_path).dropna()
        news1 = pd.read_csv(news1_path).dropna()

        # data = pd.concat([news, report['리포트']], axis = 1)
        # data['리포트'] = [f"'{x[1:-1]}'" if not pd.isnull(x) else ' ' for x in data['리포트']]

        # for i in range(len(data)):
        #     if data["리포트"][i] != ' ':
        #         data["기사제목"][i] = data["기사제목"][i][:-2] + ',' + data['리포트'][i] + data["기사제목"][i][-1]
        
        # data = data.drop(columns = ['리포트'])

        data_dict[stock] = pd.concat([news, news1], axis = 0)
        data_dict[stock]['EMA'] = data_dict[stock]['Close'].ewm(span = 20, adjust = False).mean().astype(int)

        data_dict[stock].to_csv(f"./data/data_{sector}_{stock}.csv") # concat 한 데이터 내보내서 다시 tf-idf에서 이용

    df_all = pd.concat(data_dict.values(), axis = 0)
    df_all.sort_values(by = 'Date', inplace = True)
    

    return data_dict, df_all

df_dict, df = load_data(sector, 
                        stocks)

df_samsung = df_dict["삼성전자"]
df_sk = df_dict["SK하이닉스"]


#%%
'''pre-training된 모델 불러오기'''
# https://huggingface.co/jhgan/ko-sroberta-multitask
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('jhgan/ko-sroberta-multitask')
model = AutoModel.from_pretrained('jhgan/ko-sroberta-multitask')

# finbert 이용해보기 -> 성능이 오히려 안 좋아짐
# tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert")
# model = AutoModel.from_pretrained("snunlp/KR-FinBert")

#%%
# embedded_sequences = []
# for i in tqdm.tqdm(range(len(df_samsung))):
#     tmp = [re.sub(" +", " ", x.strip("'")).strip() 
#         for x in df.iloc[i]['기사제목'].strip("[").strip("]").split(", ")]
#     embedded_sequences.append(
#         torch.cat([get_embedding(x, tokenizer, model) for x in tmp], dim=0))
# #%%
# torch.save(
#     pad_sequence(embedded_sequences, batch_first=True, padding_value=0),
#     f"./assets/news_embedded_{sector}_{stock_1}.pt"
# )

#%%
"""뉴스기사들 임베딩하여 저장"""
def make_embedded_data(df_dict : dict, 
                       sector : str, 
                       stocks : list):
    
    for stock in stocks:
        df = df_dict[stock]
        embedded_sequences = []
        for i in tqdm.tqdm(range(len(df))):
            tmp = [re.sub(" +", " ", x.strip("'")).strip() 
                for x in df.iloc[i]['기사제목'].strip("[").strip("]").split(", ")]
            embedded_sequences.append(
                torch.cat([get_embedding(x, tokenizer, model) for x in tmp], dim = 0))
        torch.save(
            pad_sequence(embedded_sequences, batch_first = True, padding_value = 0),
            f"./assets/embedding/news_embedded_{sector}_{stock}.pt"
            # f"./assets/embedding/embedding_{sector}_{stock}.pt" # 리포트 포함 embedding
        )

make_embedded_data(df_dict, 
                   sector, 
                   stocks)

#%%
"""임베딩하여 저장된 데이터 불러오기"""
def load_embedded_data(sector : str, 
                       stocks : list, 
                       n_tokens = 47):
    # n_tokens는 각 데이터의 임베딩된 토큰이 다를 경우를 방지하기 위해 설정
    # default : 50
    # 과도한 메모리 사용 방지

    embedded_data_dict = {}
        
    for stock in stocks:
        file_path = f"./assets/embedding/news_embedded_{sector}_{stock}.pt"
        # file_path = f"./assets/embedding/embedding_{sector}_{stock}.pt" # 리포트 포함 embedding
        embedded_data_dict[stock] = torch.load(file_path)[1:, :n_tokens, ...]

    return embedded_data_dict

news_embedded_dict = load_embedded_data(sector, stocks)

news_embedded_samsung = news_embedded_dict["삼성전자"]
news_embedded_sk = news_embedded_dict["SK하이닉스"]

#%%
'''경제 톱 기사와 합치기'''

# for stock in stocks:
#     df2 = pd.read_csv("./data/news_data_주요뉴스_코스피.csv").dropna()
#     df1 = df_dict[stock]
#     df2['기사제목'].iloc[0]
#     # 데이터프레임을 날짜를 기준으로 합침
#     merged_df = pd.merge(df1, df2, on='Date', suffixes=('_df1', '_df2'))

#     # 같은 날짜의 기사 제목을 하나의 리스트로 합침
#     merged_df['기사제목_합침'] = merged_df['기사제목_df1'] + merged_df['기사제목_df2']

#     # 필요한 열만 선택
#     result_df = merged_df[['Date', 'Close_df1', '기사날짜_df1','기사제목_합침']]
#     df = result_df.rename(columns={'Close_df1': 'Close', '기사날짜_df1': '기사날짜','기사제목_합침': '기사제목'})
#     df['기사제목'].iloc[:] = [re.sub("\]\[", ", ", news) for news in df['기사제목'].iloc[:]]
#     df_dict[stock] = df
#%%
"""각 주식별 수익률 계산"""
# 현재 "수익률", "로그수익률", "updown", "코스피200" 중 하나 고를 수 있음
# 앞으로 필요한 라벨을 추가할 예정
# 이동평균기반 실질수익률 사용

def make_target(df_dict : dict, 
                stocks : list, 
                target_type : str):
    
    target_dict = {}

    # regression
    # 1. 수익률 = ((다음날 종가 - 당일 종가)/당일 종가)*100
    if target_type == "수익률" :
        for stock in stocks:
            target_dict[stock] = torch.tensor(torch.from_numpy(
                (df_dict[stock]["Close"].iloc[1:].to_numpy() - df_dict[stock]["Close"].iloc[:-1].to_numpy()) / df_dict[stock]["Close"].iloc[:-1].to_numpy()
            ), dtype = torch.float32) * 100
    
    # 2. 로그수익률
    elif target_type == "로그수익률":
        for stock in stocks:
            target_dict[stock] = torch.tensor(
                torch.from_numpy(np.log(df_dict[stock]["Close"].iloc[1:].to_numpy() / df_dict[stock]["Close"].iloc[:-1].to_numpy())),
                dtype = torch.float32)
            
    # binary label
    ##### 원래 수익이 나면 1 아니면 0 : 과적합 문제 발생
    elif target_type == "updown":
        for stock in stocks:
            target_dict[stock] = torch.tensor(torch.from_numpy(np.where(
                df_dict[stock]["Close"].iloc[1:].to_numpy() - df_dict[stock]["Close"].iloc[:-1].to_numpy() >= 0,
                1,
                0
            )), dtype = torch.float32)

    ##### 코스피 기반 분류 라벨링 
    elif target_type == "코스피200_":
        df_kospi200 = pd.read_csv('data/kospi200.csv') ### 임시로 저장
        threshold = (df_kospi200["Close"].iloc[1:].to_numpy() - df_kospi200["Close"].iloc[:-1].to_numpy())/df_kospi200["Close"].iloc[:-1].to_numpy()

        for stock in stocks:
            target_dict[stock] = torch.tensor(torch.from_numpy(np.where(
                (df_dict[stock]["Close"].iloc[1:].to_numpy() - df_dict[stock]["Close"].iloc[:-1].to_numpy())/df_dict[stock]["Close"].iloc[:-1].to_numpy() >= threshold, 
                1,
                0
            )), dtype = torch.float32)
    
    ##### 코스피 기반 회귀 라벨링
    elif target_type == "kospi200":
        df_kospi200 = pd.read_csv('data/kospi200.csv') ### 임시로 저장
        threshold = torch.tensor(torch.from_numpy(
                (df_kospi200["Close"].iloc[1:].to_numpy() - df_kospi200["Close"].iloc[:-1].to_numpy()) / df_kospi200["Close"].iloc[:-1].to_numpy()
            ), dtype = torch.float32) * 100
        
        for stock in stocks:
            target_dict[stock] = torch.tensor(torch.from_numpy(
                (df_dict[stock]["Close"].iloc[1:].to_numpy() - df_dict[stock]["Close"].iloc[:-1].to_numpy()) / df_dict[stock]["Close"].iloc[:-1].to_numpy()
            ), dtype = torch.float32) * 100 - threshold

    ### 코스피 기반 실질 수익률로 계산 = ((1 + 주식등락률)/(1 + kopsi200 등락률) -1)*100
    elif target_type == "실질수익률":
        
        df_kospi200 = pd.read_csv('data/kospi200.csv') ### 임시로 저장
        kospi200_updown = torch.tensor(torch.from_numpy(
                (df_kospi200["Close"].iloc[1:].to_numpy() - df_kospi200["Close"].iloc[:-1].to_numpy()) / df_kospi200["Close"].iloc[:-1].to_numpy()
            ), dtype = torch.float32)
        
        for stock in stocks:
            stock_updown = torch.tensor(torch.from_numpy(
                (df_dict[stock]["Close"].iloc[1:].to_numpy() - df_dict[stock]["Close"].iloc[:-1].to_numpy()) / df_dict[stock]["Close"].iloc[:-1].to_numpy()
            ), dtype = torch.float32)

            target_dict[stock] = ((1 + stock_updown)/(1 + kospi200_updown + 1e-6) - 1)*100

    ### 이동평균 기반 실질 수익률로 계산 = ((1 + 주식등락률)/(1 + 이동평균 등락률) -1)*100
    elif target_type == '이동평균기반 보정':

        for stock in stocks:
            ema_updown = torch.tensor(torch.from_numpy(
                (df_dict[stock]["EMA"].iloc[1:].to_numpy() - df_dict[stock]["EMA"].iloc[:-1].to_numpy()) / df_dict[stock]["EMA"].iloc[:-1].to_numpy()
            ), dtype = torch.float32)

            stock_updown = torch.tensor(torch.from_numpy(
                (df_dict[stock]["Close"].iloc[1:].to_numpy() - df_dict[stock]["Close"].iloc[:-1].to_numpy()) / df_dict[stock]["Close"].iloc[:-1].to_numpy()
            ), dtype = torch.float32)

            target_dict[stock] = ((1 + stock_updown)/(1 + ema_updown + 1e-6) - 1)*100

    return target_dict

# target_dict = make_target(df_dict, stocks, "이동평균기반 보정")
# target_dict = make_target(df_dict, stocks, "실질수익률")
# target_dict = make_target(df_dict, stocks, "kospi200")
target_dict = make_target(df_dict, stocks, "수익률")

# target_samsung = target_dict["삼성전자"]
# target_sk = target_dict["sk하이닉스"]
#%%
"""수익률 선택에 따른 분포 변화 체크"""
# import matplotlib.pyplot as plt
# plt.hist(target_samsung)
# plt.hist(target_sk)

#%%
# """train test split"""

# def train_test_split(target_dict : dict, news_embedded_dict : dict, stocks : list, n_tests = 50):
#     # n_tests는 각 종목별 테스트 데이터 개수
#     # default = 50

#     train_embedded_dict = {}
#     test_embedded_dict = {}
#     train_target_dict = {}
#     test_target_dict = {}

#     for stock in stocks:
#         train_embedded_dict[stock] = news_embedded_dict[stock][:-n_tests, ...]
#         test_embedded_dict[stock] = news_embedded_dict[stock][-n_tests:, ...]
#         train_target_dict[stock] = target_dict[stock][:-n_tests]
#         test_target_dict[stock] = target_dict[stock][-n_tests:]

#     return train_embedded_dict, test_embedded_dict, train_target_dict, test_target_dict 

# train_embedded_dict, test_embedded_dict, train_target_dict, test_target_dict = train_test_split(target_dict, news_embedded_dict, stocks)

#%%
"""train 만 이용"""
def get_train(target_dict : dict, news_embedded_dict : dict, stocks : list):
    # n_tests는 각 종목별 테스트 데이터 개수
    # default = 50

    train_embedded_dict = {}
    train_target_dict = {}

    for stock in stocks:
        train_embedded_dict[stock] = news_embedded_dict[stock]
        train_target_dict[stock] = target_dict[stock]

    return train_embedded_dict, train_target_dict

train_embedded_dict, train_target_dict = get_train(target_dict, news_embedded_dict, stocks)

#%%
"""모델 인풋 획득(embedded, target dict의 stock별로 저장된 값들을 합침)"""

# def get_model_input(train_embedded_dict: dict,
#                     test_embedded_dict : dict,
#                     train_target_dict : dict,
#                     test_target_dict : dict):
    
#     train_embedded = torch.cat(list(train_embedded_dict.values()), dim = 0)
#     test_embedded = torch.cat(list(test_embedded_dict.values()), dim = 0)
#     train_target = torch.cat(list(train_target_dict.values()), dim = 0)
#     test_target = torch.cat(list(test_target_dict.values()), dim = 0)

#     return train_embedded, test_embedded, train_target, test_target 

# train_embedded, test_embedded, train_target, test_target = get_model_input(train_embedded_dict, test_embedded_dict, train_target_dict, test_target_dict)

#%%
"""모델 인풋 획득(embedded, target dict의 stock별로 저장된 값들을 합침)"""

def get_model_input(train_embedded_dict: dict,
                    train_target_dict : dict):
    
    train_embedded = torch.cat(list(train_embedded_dict.values()), dim = 0)
    train_target = torch.cat(list(train_target_dict.values()), dim = 0)

    return train_embedded, train_target 

train_embedded, train_target = get_model_input(train_embedded_dict, train_target_dict)

#%%
"""model 정의"""

class Regressor(nn.Module):
    def __init__(self, head_size, embedding):
        super().__init__()
        self.key = nn.Linear(embedding, head_size, bias=False)
        self.query = nn.Linear(embedding, head_size, bias=False)
        self.value = nn.Linear(embedding, head_size, bias=False)
        # 회귀 문제를 위해 마지막 레이어의 출력을 1로 유지합니다.
        self.fc = nn.Linear(head_size, 1)

    def forward(self, x):
        C = x.size(-1)
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        v = self.value(x) # (B,T,C)
        
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2 ,-1) * C ** -0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        #wei = F.softmax(wei, dim=-1) # (B, T, T)
        # 회귀를 위해 활성화 함수는 사용하지 않습니다.
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        out = out.mean(dim=1)

        return self.fc(out)#.squeeze(-1)

#%%
"""모델 수정해보는 과정"""

# class Regressor(nn.Module):
#     def __init__(self, embedding_dim, head_size):
#         super().__init__()
#         # 첫 번째 어텐션 레이어
#         self.key1 = nn.Linear(embedding_dim, head_size)
#         self.query1 = nn.Linear(embedding_dim, head_size)
#         self.value1 = nn.Linear(embedding_dim, head_size)

#         # 두 번째 어텐션 레이어
#         self.key2 = nn.Linear(head_size, head_size)
#         self.query2 = nn.Linear(head_size, head_size)
#         self.value2 = nn.Linear(head_size, head_size)

#         # 최종 출력을 위한 선형 레이어
#         self.fc = nn.Linear(head_size, 1)

#     def attention(self, query, key, value, use_softmax=True):
#         d_k = query.size(-1)
#         scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#         if use_softmax:
#             weights = F.softmax(scores, dim=-1)
#         else:
#             weights = scores
#         return torch.matmul(weights, value)

#     def forward(self, x):
#         # 첫 번째 어텐션 레이어 (활성화 함수 사용)
#         query1 = self.query1(x)
#         key1 = self.key1(x)
#         value1 = self.value1(x)
#         x = self.attention(query1, key1, value1, use_softmax=True)

#         # 두 번째 어텐션 레이어 (활성화 함수 미사용)
#         query2 = self.query2(x)
#         key2 = self.key2(x)
#         value2 = self.value2(x)
#         x = self.attention(query2, key2, value2, use_softmax=False)

#         # 평균을 계산하고 최종 출력을 위한 선형 레이어를 통과
#         x = x.mean(dim = 1)
#         return self.fc(x)

#%%
# class Regressor(nn.Module):
#     def __init__(self, embedding_dim, head_size):
#         super().__init__()
#         # 첫 번째 어텐션 레이어
#         self.key1 = nn.Linear(embedding_dim, head_size)
#         self.query1 = nn.Linear(embedding_dim, head_size)
#         self.value1 = nn.Linear(embedding_dim, head_size)

#         # 두 번째 어텐션 레이어
#         self.key2 = nn.Linear(head_size, head_size)
#         self.query2 = nn.Linear(head_size, head_size)
#         self.value2 = nn.Linear(head_size, head_size)

#         # 최종 출력을 위한 선형 레이어
#         self.fc = nn.Linear(head_size, 1)

#     def attention(self, query, key, value):
#         d_k = query.size(-1)
#         scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#         return torch.matmul(scores, value)

#     def forward(self, x):
#         # 첫 번째 어텐션 레이어 (활성화 함수 미사용)
#         query1 = self.query1(x)
#         key1 = self.key1(x)
#         value1 = self.value1(x)
#         x = self.attention(query1, key1, value1)

#         # 두 번째 어텐션 레이어 (활성화 함수 미사용)
#         query2 = self.query2(x)
#         key2 = self.key2(x)
#         value2 = self.value2(x)
#         x = self.attention(query2, key2, value2)

#         # 평균을 계산하고 최종 출력을 위한 선형 레이어를 통과
#         x = x.mean(dim = 1)
#         return self.fc(x)


#%%
"""모델, loss, optmizer 선언"""
regressor = Regressor(768, 3).to(device)
regressor
regressor.train()

criterion = nn.MSELoss()
# criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    regressor.parameters(), lr = config["lr"]
)

#%%
"""모델 크기 check"""
count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad) # numel = num of element
num_params = count_parameters(regressor)
print(f"Number of Parameters: {num_params / 1000:.1f}K")

#%%
"""SAM optimizer 해보기"""
# class SAM(Optimizer):
#     def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
#         if rho < 0.0:
#             raise ValueError(f"Invalid rho, should be non-negative: {rho}")

#         defaults = dict(rho=rho, **kwargs)
#         super(SAM, self).__init__(params, defaults)

#         self.base_optimizer = base_optimizer(params, **kwargs)
#         self.param_groups = self.base_optimizer.param_groups

#     @torch.no_grad()
#     def first_step(self, zero_grad=False):
#         grad_norm = self._grad_norm()
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 e_w = (torch.pow(p, 2) if group.get('weight_decay', 0) else 0.) + p.grad * group['lr']
#                 p.sam_e_w = e_w * self.rho / (grad_norm + 1e-12)  # 저장
#                 p.add_(p.sam_e_w, alpha=1.0)

#         self.base_optimizer.step()

#         if zero_grad:
#             self.zero_grad()

#     @torch.no_grad()
#     def second_step(self, zero_grad=False):
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 p.sub_(p.sam_e_w)  # 저장된 값을 사용하여 되돌림

#         self.base_optimizer.step()


#         if zero_grad:
#             self.zero_grad()

#     def _grad_norm(self):
#         norm = torch.norm(
#                     torch.stack([
#                         p.grad.norm(p=2)
#                         for group in self.param_groups
#                         for p in group['params']
#                         if p.grad is not None
#                     ]),
#                     p=2
#                )
#         return norm

# optimizer = SAM(regressor.parameters(),
#                  torch.optim.SGD, 
#                  lr=config['lr'],
#                  momentum = 0.9)

#%%
"""model train"""
train_module = importlib.import_module("module.train")
importlib.reload(train_module)

for epoch in range(config['epochs']):
    logs = train_module.train_function(train_embedded, train_target, regressor, config, optimizer, device)

    if (epoch + 1) % 10 == 0:
        print_input = f"Epoch [{epoch+1:03d}/{config['epochs']}]"
        print_input += "".join(
            [", {}: {:.6f}".format(x, np.mean(y)) for x, y in logs.items()]
        )
        print(print_input)

regressor.eval()
#%%
"""학습 데이터에 대한 예측 및 평가"""
train_pred = regressor(train_embedded).detach().cpu().numpy().squeeze()
smape = np.abs(train_pred - train_target.cpu().numpy()) 
smape /= (np.abs(train_pred) / 2 + np.abs(train_target.cpu().numpy()) / 2)
smape = smape.mean()
print(f"Train SMAPE: {smape:.3f}")

#%%g09
"""테스트 데이터에 대한 예측 및 평가"""
# test_pred = regressor(test_embedded).detach().cpu().numpy()
# smape = np.abs(test_pred - test_target.cpu().numpy()) 
# smape /= (np.abs(test_pred) / 2 + np.abs(test_target.cpu().numpy()) / 2)
# smape = smape.mean()
# print(f"Test SMAPE: {smape:.3f}")

#%%
"""tf_idf를 통한 키워드 추출"""
def get_kewords(stocks : list):
    kewords_dict = {}
    for stock in stocks:
        news_path = f"./data/data_{sector}_{stock}.csv" # 뉴스기사만 이용
        get_keyword = tf_idf(news_path, "기사제목", config["num_keywords"])
        kewords_dict[stock] = get_keyword.fit()
    
    return kewords_dict

keywords_dict = get_kewords(stocks)
# keywords_dict['삼성전자']
# keywords_dict['SK하이닉스']

#%%
"""미리 데이터 베이스를 만들기"""
trend_news_dict = {}
trend_value_dict = {}
trend_date_dict = {}
trend_embed_dict = {}
for stock in stocks:
    """Trend in AlldataSet"""
    trend_news_list = []
    trend_value_list = []
    trend_date_list = []
    trend_embed_list = []
    for idx in range(len(target_dict[stock])):
        # 실제 값과 예측 값을 비교하는 코드를 삭제하거나 회귀 문제에 맞게 조정합니다.
        nonzero_idx = torch.nonzero(news_embedded_dict[stock][idx].sum(1))[:, 0]
        label = target_dict[stock][idx]
        pred = regressor(news_embedded_dict[stock][[idx]]).detach()  # 활성화 함수 제거

        without_pred = []  # 회귀 문제에서는 확률 간의 차이가 아닌, 값의 차이를 계산합니다.
        for i in nonzero_idx:
            without_idx = list(range(news_embedded_dict[stock].size(1)))
            without_idx.remove(i)
            # 활성화 함수를 사용하지 않고, 예측 값의 차이를 계산합니다.
            without_pred.append(
                regressor(news_embedded_dict[stock][[[idx]], without_idx, ...]).detach().item())

        # 회귀 문제에서는 가장 큰 긍정 혹은 부정적 영향을 미치는 뉴스를 찾는 방식을 조정해야 합니다.
        most_idx = np.argmax(np.abs(without_pred))  # 가장 큰 영향을 미치는 뉴스의 인덱스
        tmp = [re.sub(" +", " ", x.strip("'")).strip() 
            for x in df_dict[stock].iloc[1:].iloc[idx]['기사제목'].strip("[").strip("]").split(", ")]
        trend_news_list.append(tmp[most_idx])
        trend_value_list.append([label.item(), without_pred[most_idx], pred.item()])
        trend_date_list.append(df_dict[stock].iloc[1:].iloc[idx]['Date'])
        trend_embed_list.append(news_embedded_dict[stock][[idx], most_idx])

    trend_news_dict[stock] = trend_news_list
    trend_value_dict[stock] = trend_value_list
    trend_date_dict[stock] = trend_date_list
    trend_embed_dict[stock] = trend_embed_list

for stock in stocks:
    torch.save(trend_embed_dict[stock],
                f"./data/trend_data/trend_embedd_database_{stock}.pt"
            )
    result = pd.DataFrame(trend_news_dict[stock], columns = ['news'])
    value = pd.DataFrame(trend_value_dict[stock], columns = ['True_value', 'Predicted_Without','Predicted_With'])
    date = pd.DataFrame(trend_date_dict[stock], columns = ['Date'])

    df = pd.concat([date, result, value], axis = 1)
    df.to_csv('./data/trend_data/trend_news_{}.csv'.format(stock), mode = 'w', encoding = 'utf-8-sig', index = False)

#%%
# def save_trend_output(keywords_dict : dict,
#                       df_dict : dict,
#                       train_embedded_dict : dict,
#                       test_embedded_dict : dict,
#                       train_target_dict : dict,
#                       test_target_dict : dict,                      
#                       stocks : list):
    
#     train_news_dict = {}
#     test_news_dict = {}
#     train_keword_news_dict = {}
#     test_keword_news_dict = {}

#     for stock in stocks:
#         for keyword in keywords_dict[stock]:
            
#             """Trend in TrainSet"""
#             train_keyword_news_list = []
#             with open(f"./assets/output6/train_{sector}_{stock}_{keyword[0]}.txt", "w") as f:
#                 for idx in range(len(train_target_dict[stock])):
                    
#                     # 실제 값과 예측 값을 비교하는 코드를 삭제하거나 회귀 문제에 맞게 조정합니다.
#                     nonzero_idx = torch.nonzero(train_embedded_dict[stock][idx].sum(1))[:, 0]
#                     label = train_target_dict[stock][idx]
#                     pred = regressor(train_embedded_dict[stock][[idx]]).detach()  # 활성화 함수 제거

#                     without_pred = []  # 회귀 문제에서는 값 자체의 차이를 계산합니다.
#                     for i in nonzero_idx:
#                         without_idx = list(range(train_embedded_dict[stock].size(1)))
#                         without_idx.remove(i)
                        
#                         # 활성화 함수를 사용하지 않고, 예측 값의 차이를 계산합니다.
#                         without_pred.append(
#                             pred.item() - regressor(train_embedded_dict[stock][[[idx]], without_idx, ...]).detach().item())

#                     # 회귀 문제에서는 가장 큰 긍정 혹은 부정적 영향을 미치는 뉴스를 찾는 방식을 조정해야 합니다.
#                     most_idx = np.argmax(np.abs(without_pred))  # 가장 큰 영향을 미치는 뉴스의 인덱스
#                     tmp = [re.sub(" +", " ", x.strip("'")).strip() 
#                         for x in df_dict[stock].iloc[1:].iloc[:-50].iloc[idx]['기사제목'].strip("[").strip("]").split(", ")]
                    
#                     if keyword[0] in tmp[most_idx]:
#                         f.write(f"Date: {df_dict[stock].iloc[1:].iloc[:-50].iloc[idx]['Date']}\n")
#                         f.write(f"Most Influencial News: {tmp[most_idx]}\n")
#                         f.write(f"True Value: {label.item():.3f}\n")
#                         f.write(f"Predicted Without: {without_pred[most_idx]:.3f}\n")
#                         f.write(f"Predicted With: {pred.item():.3f}\n")
#                         f.write("\n\n")
#                         train_keyword_news_list.append(tmp[most_idx])

#             train_keword_news_dict[keyword[0]] = train_keyword_news_list
#             train_news_dict[stock] = train_keword_news_dict


#             """Trend in TestSet"""
#             test_keyword_news_list = []
#             with open(f"./assets/output6/test_{sector}_{stock}_{keyword[0]}.txt", "w") as f:
#                 for idx in range(len(test_embedded_dict[stock])):
                    
#                     # 예측이 맞았는지의 여부를 확인하는 코드는 회귀 문제에서는 필요 없으므로 제거합니다.
#                     nonzero_idx = torch.nonzero(test_embedded_dict[stock][idx].sum(1))[:, 0]
#                     label = test_target_dict[stock][idx]
#                     pred = regressor(test_embedded_dict[stock][[idx]]).detach()  # 활성화 함수 제거

#                     without_pred = []  # 값의 차이를 계산합니다.
#                     for i in nonzero_idx:
#                         without_idx = list(range(test_embedded_dict[stock].size(1)))
#                         without_idx.remove(i)
                        
#                         # 활성화 함수를 사용하지 않고, 예측 값의 차이를 계산합니다.
#                         without_pred.append(
#                             pred.item() - regressor(test_embedded_dict[stock][[[idx]], without_idx, ...]).detach().item())

#                     most_idx = np.argmax(np.abs(without_pred))  # 가장 큰 영향을 미치는 뉴스의 인덱스
#                     tmp = [re.sub(" +", " ", x.strip("'")).strip() 
#                         for x in df_dict[stock].iloc[1:].iloc[-50:].iloc[idx]['기사제목'].strip("[").strip("]").split(", ")]
                    
#                     if keyword[0] in tmp[most_idx]:
#                         f.write(f"Date: {df_dict[stock].iloc[1:].iloc[-50:].iloc[idx]['Date']}\n")
#                         f.write(f"Most Influencial News: {tmp[most_idx]}\n")
#                         f.write(f"True Value: {label.item():.3f}\n")
#                         f.write(f"Predicted Without: {without_pred[most_idx]:.3f}\n")
#                         f.write(f"Predicted With: {pred.item():.3f}\n")
#                         f.write("\n\n")
#                         test_keyword_news_list.append(tmp[most_idx])

#             test_keword_news_dict[keyword[0]] = test_keyword_news_list
#             test_news_dict[stock] = test_keword_news_dict

#     return train_news_dict, test_news_dict


# train_news_dict, test_news_dict = save_trend_output(keywords_dict,
#                                                     df_dict,
#                                                     train_embedded_dict,
#                                                     test_embedded_dict,
#                                                     train_target_dict,
#                                                     test_target_dict,     
#                                                     stocks) 

#%%
def save_trend_output(keywords_dict : dict,
                      df_dict : dict,
                      train_embedded_dict : dict,
                      train_target_dict : dict,                      
                      stocks : list):
    
    train_news_dict = {}
    train_keword_news_dict = {}

    output_path = f'./assets/output_{config["output_details"]}'

    # 해당 경로에 폴더가 없다면 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for stock in stocks:
        for keyword in keywords_dict[stock]:
            
            """Trend in TrainSet"""
            train_keyword_news_list = []
            with open(f"{output_path}/train_{sector}_{stock}_{keyword[0]}.txt", "w") as f:
                for idx in range(len(train_target_dict[stock])):
                    
                    nonzero_idx = torch.nonzero(train_embedded_dict[stock][idx].sum(1))[:, 0]
                    label = train_target_dict[stock][idx]
                    pred = regressor(train_embedded_dict[stock][[idx]]).detach()  # 활성화 함수 제거

                    without_pred = []  
                    for i in nonzero_idx:
                        without_idx = list(range(train_embedded_dict[stock].size(1)))
                        without_idx.remove(i)
                        without_pred.append(
                            pred.item() - regressor(train_embedded_dict[stock][[[idx]], without_idx, ...]).detach().item())

                    most_idx = np.argmax(np.abs(without_pred)) 
                    tmp = [re.sub(" +", " ", x.strip("'")).strip() 
                        for x in df_dict[stock].iloc[1:].iloc[idx]['기사제목'].strip("[").strip("]").split(", ")]
                    
                    if keyword[0] in tmp[most_idx]:
                        f.write(f"Date: {df_dict[stock].iloc[1:].iloc[idx]['Date']}\n")
                        f.write(f"Most Influencial News: {tmp[most_idx]}\n")
                        f.write(f"True Value: {label.item():.3f}\n")
                        f.write(f"Predicted Without: {without_pred[most_idx]:.3f}\n")
                        f.write(f"Predicted With: {pred.item():.3f}\n")
                        f.write("\n\n")
                        train_keyword_news_list.append(tmp[most_idx])

            train_keword_news_dict[keyword[0]] = train_keyword_news_list
            train_news_dict[stock] = train_keword_news_dict

    return train_news_dict

train_news_dict = save_trend_output(keywords_dict,
                                    df_dict,
                                    train_embedded_dict,
                                    train_target_dict,     
                                    stocks)
#%%
def get_threshold(news_dict : dict,
                  keywords_dict : dict,
                  stocks : list,
                  q = 2):
    
    score_dict = {}
    all_score_list = []

    for stock in stocks:
        keyword_score_dict = {}
        stock_kewords_vector = ""
        for keyword in keywords_dict[stock]:
            stock_kewords_vector += keyword[0] + ' '
        stock_kewords_vector = get_embedding(stock_kewords_vector, tokenizer, model)  

        for keyword in keywords_dict[stock]:
            score_list = []
            for news in news_dict[stock][keyword[0]]:
                embedded_news = get_embedding(news, tokenizer, model)
                score = cosine_similarity(embedded_news, stock_kewords_vector)
                score_list.append(score)
                all_score_list.append(score)
            keyword_score_dict[keyword[0]] = score_list
            score_dict[stock] = keyword_score_dict

    threshold = statistics.quantiles(all_score_list)[q] # 필요에 따라서 변경

    return threshold, score_dict

train_threshold, train_score_dict = get_threshold(train_news_dict, 
                                                  keywords_dict, 
                                                  stocks,
                                                  q = config['quantile'])

# test_threshold, test_score_dict = get_threshold(test_news_dict,
#                                                 keywords_dict, 
#                                                 stocks,
#                                                 q = config['quantile'])

# %%
def select_news(news_dict : dict,
                keywords_dict : dict,
                score_dict : dict,
                threshold : np.array,
                stocks : list):
    
    filtered_news_dict = {}
    for stock in stocks:
        keyword_score_dict = {}
        for keyword in keywords_dict[stock]:
            idx = score_dict[stock][keyword[0]] >= threshold
            idx = idx.flatten()
            filtered_values = [value for value, flag in zip(news_dict[stock][keyword[0]], idx) if flag]
            keyword_score_dict[keyword[0]] = filtered_values
        filtered_news_dict[stock] = keyword_score_dict

    return filtered_news_dict

train_filtered_news_dict = select_news(train_news_dict,
                                       keywords_dict, 
                                       train_score_dict, 
                                       train_threshold, 
                                       stocks)

#%%      
def parse_file(file_path):
    data_list = []
    current_data = {}

    with open(file_path, 'r', encoding = 'utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if 'Date' in line:
                if current_data:
                    data_list.append(current_data)
                    current_data = {}
                current_data['Date'] = line.split(': ')[1].strip()
            elif 'Most Influencial News' in line:
                current_data['Most Influencial News'] = line.split(': ')[1].strip()
            elif 'True Value' in line:
                current_data['True Value'] = float(line.split(': ')[1].strip())
            elif 'Predicted Without' in line:
                current_data['Predicted Without'] = float(line.split(': ')[1].strip())
            elif 'Predicted With' in line:
                current_data['Predicted With'] = float(line.split(': ')[1].strip())
        if current_data: 
            data_list.append(current_data)

    return data_list

#%%
def text2dict(keywords_dict: dict, 
              sector: str, 
              stocks: list, 
              is_train=True):
    
    text_dict = {}
    output_path = f'./assets/output_{config["output_details"]}'

    for stock in stocks:
        stock_text_dict = {}
        for keyword in keywords_dict[stock]:
            file_suffix = 'train' if is_train else 'test'
            file_path = f"{output_path}/{file_suffix}_{sector}_{stock}_{keyword[0]}.txt"
            stock_text_dict[keyword[0]] = parse_file(file_path)
        text_dict[stock] = stock_text_dict

    return text_dict


train_output_dict = text2dict(keywords_dict, 
                              sector, 
                              stocks)
#%%
def save_filtered_outputs(filtered_news_dict : dict,
                          keywords_dict : dict,
                          output_dict : dict,
                          stocks : list,
                          sector : str,
                          is_train = True):
    
    file_suffix = 'train' if is_train else 'test'
    real_output_path = f'./assets/real_output_{config["output_details"]}'

    # 해당 경로에 폴더가 없다면 생성
    if not os.path.exists(real_output_path):
        os.makedirs(real_output_path)
    
    for stock in stocks:
        for keyword in keywords_dict[stock]:
            file_path = f"{real_output_path}/{file_suffix}_{sector}_{stock}_{keyword[0]}.txt"
            with open(file_path, "w") as f:
                for dictionary in output_dict[stock][keyword[0]]:
                    if dictionary['Most Influencial News'] in filtered_news_dict[stock][keyword[0]]:
                        f.write(f"Date: {dictionary['Date']}\n")
                        f.write(f"Most Influencial News: {dictionary['Most Influencial News']}\n")
                        f.write(f"True Value: {dictionary['True Value']:}\n")
                        f.write(f"Predicted Without: {dictionary['Predicted Without']}\n")
                        f.write(f"Predicted With: {dictionary['Predicted With']}\n")
                        f.write("\n\n")

save_filtered_outputs(train_filtered_news_dict,
                      keywords_dict, 
                      train_output_dict, 
                      stocks, 
                      sector)

# %%
