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
from torch.nn.utils.rnn import pad_sequence
from module.embedding import get_embedding
from TF_IDF import tf_idf

#%%
'''주식 분야와 해당 회사들 설정'''

companys = {'반도체':['삼성전자', 'sk하이닉스'], 
            'IT':['네이버', '카카오'], 
            '자동차':['현대차', '기아'],
            '통신사':['SKT', 'KT', 'LG유플러스'],
            '바이오':['삼성바이오', '셀트리온'],
            '2차전지':['lg화학', '삼성sdi'],
            '지수':['코스피', '나스닥', '상해종합지수', '닛케이지수']}

sector = "반도체"
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
}

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
config["cuda"] = device

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
        news_path = f"./data/news_data_{sector}_{stock}.csv"
        data_dict[stock] = pd.read_csv(news_path).dropna()
        data_dict[stock]['EMA'] = data_dict[stock]['Close'].ewm(span = 20, adjust = False).mean().astype(int)

    df_all = pd.concat(data_dict.values(), axis = 0)
    df_all.sort_values(by = 'Date', inplace = True)

    return data_dict, df_all

df_dict, df = load_data(sector, 
                        stocks)

#%%
'''pre-training된 모델 불러오기'''
from transformers import AutoTokenizer, AutoModel
# https://huggingface.co/jhgan/ko-sroberta-multitask

tokenizer = AutoTokenizer.from_pretrained('jhgan/ko-sroberta-multitask')
model = AutoModel.from_pretrained('jhgan/ko-sroberta-multitask')

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
            f"./assets/embedded_data/news_embedded_{sector}_{stock}.pt")

make_embedded_data(df_dict, 
                   sector, 
                   stocks)

#%%
"""임베딩하여 저장된 데이터 불러오기"""
def load_embedded_data(sector : str, 
                       stocks : list, 
                       n_tokens = 47):
    # n_tokens는 각 데이터의 임베딩된 토큰이 다를 경우를 방지하기 위해 설정
    # 과도한 메모리 사용 방지

    embedded_data_dict = {}
        
    for stock in stocks:
        file_path = f"./assets/embedded_data/news_embedded_{sector}_{stock}.pt"
        embedded_data_dict[stock] = torch.load(file_path)[1:, :n_tokens, ...]

    return embedded_data_dict

#%%
"""각 주식별 수익률 계산"""
def make_target(df_dict : dict, 
                stocks : list, 
                target_type : str):
    
    target_dict = {}

    # regression
    # 1. 수익률 = ((다음날 종가 - 당일 종가)/당일 종가)*100
    if target_type == "수익률" :
        for stock in stocks:
            target_dict[stock] = torch.tensor(torch.from_numpy(
                (df_dict[stock]["Close"].iloc[1:].to_numpy() - df_dict[stock]["Close"].iloc[:-1].to_numpy()) / df_dict[stock]["Close"].iloc[:-1].to_numpy()),
                dtype = torch.float32) * 100
    
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
                df_dict[stock]["Close"].iloc[1:].to_numpy() - df_dict[stock]["Close"].iloc[:-1].to_numpy() >= 0, 1, 0)), 
                dtype = torch.float32)

    ##### 코스피 기반 분류 라벨링 
    elif target_type == "코스피200_":
        df_kospi200 = pd.read_csv('data/kospi200.csv') ### 임시로 저장
        threshold = (df_kospi200["Close"].iloc[1:].to_numpy() - df_kospi200["Close"].iloc[:-1].to_numpy())/df_kospi200["Close"].iloc[:-1].to_numpy()

        for stock in stocks:
            target_dict[stock] = torch.tensor(torch.from_numpy(np.where(
                (df_dict[stock]["Close"].iloc[1:].to_numpy() - df_dict[stock]["Close"].iloc[:-1].to_numpy())/df_dict[stock]["Close"].iloc[:-1].to_numpy() >= threshold, 1, 0)), 
                dtype = torch.float32)
    
    ##### 코스피 기반 회귀 라벨링
    elif target_type == "kospi200":
        df_kospi200 = pd.read_csv('data/kospi200.csv') ### 임시로 저장
        threshold = torch.tensor(torch.from_numpy(
                (df_kospi200["Close"].iloc[1:].to_numpy() - df_kospi200["Close"].iloc[:-1].to_numpy()) / df_kospi200["Close"].iloc[:-1].to_numpy()), 
                dtype = torch.float32) * 100
        
        for stock in stocks:
            target_dict[stock] = torch.tensor(torch.from_numpy(
                (df_dict[stock]["Close"].iloc[1:].to_numpy() - df_dict[stock]["Close"].iloc[:-1].to_numpy()) / df_dict[stock]["Close"].iloc[:-1].to_numpy()), 
                dtype = torch.float32) * 100 - threshold

    ### 코스피 기반 실질 수익률로 계산 = ((1 + 주식등락률)/(1 + kopsi200 등락률) -1)*100
    elif target_type == "실질수익률":
        
        df_kospi200 = pd.read_csv('data/kospi200.csv') ### 임시로 저장
        kospi200_updown = torch.tensor(torch.from_numpy(
                (df_kospi200["Close"].iloc[1:].to_numpy() - df_kospi200["Close"].iloc[:-1].to_numpy()) / df_kospi200["Close"].iloc[:-1].to_numpy()), 
                dtype = torch.float32)
        
        for stock in stocks:
            stock_updown = torch.tensor(torch.from_numpy(
                (df_dict[stock]["Close"].iloc[1:].to_numpy() - df_dict[stock]["Close"].iloc[:-1].to_numpy()) / df_dict[stock]["Close"].iloc[:-1].to_numpy()), 
                dtype = torch.float32)

            target_dict[stock] = ((1 + stock_updown)/(1 + kospi200_updown + 1e-6) - 1)*100

    ### 이동평균 기반 실질 수익률로 계산 = ((1 + 주식등락률)/(1 + 이동평균 등락률) -1)*100
    elif target_type == '이동평균기반 보정':

        for stock in stocks:
            ema_updown = torch.tensor(torch.from_numpy(
                (df_dict[stock]["EMA"].iloc[1:].to_numpy() - df_dict[stock]["EMA"].iloc[:-1].to_numpy()) / df_dict[stock]["EMA"].iloc[:-1].to_numpy()), 
                dtype = torch.float32)

            stock_updown = torch.tensor(torch.from_numpy(
                (df_dict[stock]["Close"].iloc[1:].to_numpy() - df_dict[stock]["Close"].iloc[:-1].to_numpy()) / df_dict[stock]["Close"].iloc[:-1].to_numpy()), 
                dtype = torch.float32)

            target_dict[stock] = ((1 + stock_updown)/(1 + ema_updown + 1e-6) - 1)*100

    return target_dict

#%%
"""train 만 이용"""
def get_train(target_dict : dict, 
              news_embedded_dict : dict, 
              stocks : list):

    train_embedded_dict = {}
    train_target_dict = {}

    for stock in stocks:
        train_embedded_dict[stock] = news_embedded_dict[stock]
        train_target_dict[stock] = target_dict[stock]

    return train_embedded_dict, train_target_dict

#%%
"""모델 인풋 획득(embedded, target dict의 stock별로 저장된 값들을 합침)"""

def get_model_input(train_embedded_dict: dict,
                    train_target_dict : dict):
    
    train_embedded = torch.cat(list(train_embedded_dict.values()), dim = 0)
    train_target = torch.cat(list(train_target_dict.values()), dim = 0)

    return train_embedded, train_target 

#%%
"""model 정의"""
class Regressor(nn.Module):
    def __init__(self, head_size, embedding):
        super().__init__()
        self.key = nn.Linear(embedding, head_size, bias=False)
        self.query = nn.Linear(embedding, head_size, bias=False)
        self.value = nn.Linear(embedding, head_size, bias=False)

        self.fc = nn.Linear(head_size, 1)

    def forward(self, x):
        C = x.size(-1)
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        v = self.value(x) # (B,T,C)
        
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2 ,-1) * C ** -0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        out = out.mean(dim = 1)

        return self.fc(out)#.squeeze(-1)

#%%
"""모델 크기 check"""
regressor = Regressor(config['head_size'], 768).to(device)
count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
num_params = count_parameters(regressor)
print(f"Number of Parameters: {num_params / 1000:.1f}K")

#%%
"""sector별 모델 학습 및 데이터 베이스 구축"""

def database(sector, stocks, token = 47):
    # 데이터 로드
    df_dict, df = load_data(sector, stocks)
    news_embedded_dict = load_embedded_data(sector, stocks, n_tokens = token)
    target_dict = make_target(df_dict, stocks, "수익률")
    train_embedded, train_target = get_model_input(news_embedded_dict, target_dict)
    
    # 모형 설정
    regressor = Regressor(config["head_size"], 768).to(device)
    regressor.train()
    optimizer = torch.optim.Adam(regressor.parameters(), lr = config["lr"])

    # 학습
    train_module = importlib.import_module("module.train")
    importlib.reload(train_module)
    
    for epoch in range(config["epochs"]):
        logs = train_module.train_function(train_embedded, train_target, regressor, config, optimizer, device)

        if epoch % 50 == 0:
            print_input = f"Epoch [{epoch + 1:03d}/{config['epochs']}]"
            print_input += "".join([", {}: {:.6f}".format(x, np.mean(y)) for x, y in logs.items()])
            print(print_input)
    
    model.eval()
    # train smape
    train_pred = regressor(train_embedded).detach().cpu().numpy().squeeze()
    smape = np.abs(train_pred - train_target.cpu().numpy()) 
    smape /= (np.abs(train_pred) / 2 + np.abs(train_target.cpu().numpy()) / 2)
    smape = smape.mean()
    print(f"Train SMAPE: {smape:.3f}")
    

    # 데이터베이스 생성
    trend_news_dict = {}
    trend_value_dict = {}
    trend_date_dict = {}
    trend_embed_dict = {}

    for stock in tqdm.tqdm(stocks):
        trend_news_list = []
        trend_value_list = []
        trend_date_list = []
        trend_embed_list = []
        
        for idx in range(len(target_dict[stock])):
            nonzero_idx = torch.nonzero(news_embedded_dict[stock][idx].sum(1))[:, 0]
            label = target_dict[stock][idx]
            pred = regressor(news_embedded_dict[stock][[idx]]).detach()  
            without_pred = []  

            for i in nonzero_idx:
                without_idx = list(range(news_embedded_dict[stock].size(1)))
                without_idx.remove(i)
                without_pred.append(
                    pred.item() - regressor(news_embedded_dict[stock][[[idx]], without_idx, ...]).detach().item())

            most_idx = np.argmax(np.abs(without_pred))  
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

        torch.save(trend_embed_dict[stock], f"./data/trend_data/trend_embedd_database_{stock}.pt")

        result = pd.DataFrame(trend_news_dict[stock], columns=['news'])
        value = pd.DataFrame(trend_value_dict[stock], columns=['True_value', 'Predicted_Without','Predicted_With'])
        date = pd.DataFrame(trend_date_dict[stock], columns=['Date'])

        df = pd.concat([date, result, value], axis = 1)
        df.to_csv('./data/trend_data/trend_news_{}.csv'.format(stock), mode = 'w', encoding =' utf-8-sig', index = False)

for sector in companys.keys():
    stocks = companys[sector]
    token = 47
    if sector == '지수': token = 20 # 지수에 대해서는 token 20으로 설정해야함
    database(sector, stocks, token)

#%%
"""tf_idf를 통한 키워드 추출 : 추천키워드"""
keywords_dict = {}

for sector in companys.keys():
    for stock in companys[sector]:
        news_path = f"./data/news_data_{sector}_{stock}.csv"
        get_keyword = tf_idf(news_path, "기사제목", config['num_keywords'])
        keywords_dict[stock] = [keyword[0] for keyword in get_keyword.fit()]

df = pd.DataFrame(keywords_dict)
df.to_csv('./data/trend_data/kewords.csv', encoding = 'utf-8-sig', index = False)
     