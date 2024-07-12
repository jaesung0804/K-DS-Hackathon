#%%
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup as bs
from datetime import datetime, timedelta
import FinanceDataReader as fdr
from tqdm import tqdm
import numpy as np
import time
#%%
stock_name = '삼성전자'
code = '005930'
start_date = '2018.11.01'
end_date = '2023.10.31'
delay_time = 1.5

header = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}

#%%
## 뉴스 데이터 수집
price=fdr.DataReader(symbol=code, start=start_date, end=end_date).reset_index()[['Date','Close']]

def news_collector(search, media, make_date1, make_date2):
    part1='https://search.naver.com/search.naver?where=news&sm=tab_pge&query='
    part2='&sort=0&photo=0&field=0&pd=3&ds='
    
    date_1=f'{make_date1}&de={make_date1}'
    part3="&mynews=1&office_type=1&office_section_code=1&news_office_checked="
    part4="&nso=so:r,p:from"
    
    date_2=f'{make_date2}to{make_date2}'
    part5=",a:all&start="
    start_news_cnt='1'
    news_list=[]
    
    while start_news_cnt!='0':
        
        url=part1+search+part2+date_1+part3+media+part4+date_2+part5+start_news_cnt
        html = requests.get(url, headers=header).text
        delay = np.random.exponential(scale=delay_time) 
        time.sleep(delay)
        soup=bs(html, 'html.parser')
        nb=soup.select('div.sc_page > a')
        check = soup.select('h3')
        if len(check) != 0: # 밴 먹은 경우
            print(check)
            result = pd.DataFrame(data_dict)
            df = pd.concat([price, result], axis=1)
            df.to_csv('Trunc_data_{}.csv'.format(stock_name), mode='w', encoding='utf-8-sig', index=False)
            delay = np.random.exponential(scale=300) 
            time.sleep(delay)
            continue
                
        if len(nb) ==0 : # 아예 검색결과가 없는 경우        
            time.sleep(3)
            break 
        
        nb=nb[1] # 다음 페이지로 가는 버튼 (이전 페이지로 가는 버튼은 [0])
        if nb.attrs['aria-disabled']=='false': # 다음페이지로 넘어갈 수 있으면
            start_news_cnt=str(int(start_news_cnt)+10)
        else:
            start_news_cnt='0'

        news=soup.select('div.group_news> ul.list_news > li div.news_area > div.news_info > div.info_group > a')
        
        for i, article in enumerate(news):
            site_news=article.attrs['href']
            if ('n.news' in site_news):
                news_html=requests.get(site_news, headers=header).text
                soup=bs(news_html, 'html.parser')
                content=str(soup.select('title'))
                content=content.replace("[<title>", '')
                content=content.replace("</title>]", '')
                if 'browse_title' in content: continue
                content= re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]',' ', content)
                content=content.replace("↓", '하락')
                content=content.replace("↑", '상승')
                news_list.append(content) # 네이버 뉴스 본문 담기

        
    return news_list

data_dict = {"기사날짜": [], "기사제목": []}
for date in tqdm(price['Date']):
    date = date - timedelta(days=1)
    # 년월일 생성
    year=str(date.year)
    m=date.month
    d=date.day
    
    mm_=lambda m: str(m) if m>=10 else '0'+str(m)
    dd_=lambda m: str(d) if m>=10 else '0'+str(m)
    month=mm_(m)
    day=dd_(d)
    make_date1=year+'.'+month+'.'+day
    make_date2=year+month+day
    
    # 매일경제: 1009
    # 파이넨셜: 1014
    # 한국경제: 1015
    # 연합뉴스: 1001
    # 머니투데이: 1008
    
    Maeil_economy=news_collector(stock_name, "1009", make_date1, make_date2)
    Financial=news_collector(stock_name, "1014", make_date1, make_date2)
    Korea_economy=news_collector(stock_name, "1015", make_date1, make_date2)
    Moneytoday=news_collector(stock_name, "1008", make_date1, make_date2)
    Yeonhap = news_collector(stock_name, "1018", make_date1, make_date2)
    Maeil_economy.extend(Financial)
    Maeil_economy.extend(Korea_economy)
    Maeil_economy.extend(Moneytoday)
    Maeil_economy.extend(Yeonhap)
    if len(Maeil_economy) == 0: print("Empty!!")
    data_dict['기사날짜'].append(make_date1)
    data_dict['기사제목'].append(Maeil_economy)
      
result = pd.DataFrame(data_dict)

df = pd.concat([price, result], axis=1)
df.to_csv('news_data_{}_{}.csv'.format(stock_name, end_date), mode='w', encoding='utf-8-sig', index=False)

#%%    
## 리포트 데이터 수집
report_dict = {"Date": [], "리포트": []}
page = 1
while True:
    url = 'https://finance.naver.com/research/company_list.naver?keyword=&brokerCode=&writeFromDate=&writeToDate=&searchType=itemCode&itemName=%BB%EF%BC%BA%C0%FC%C0%DA&itemCode=' + str(code) + '&page=' + str(page) 
    source_code = requests.get(url).text
    html = bs(source_code, 'html.parser')

    news=html.select('td > a')
    article = news[1]
    for article in news:
        site_news=article.attrs['href']
        if 'company_read' in site_news:
            news_html=requests.get('https://finance.naver.com/research/' + site_news).text
            soup=bs(news_html, 'html.parser')
            date = str(soup.select('tr > th.view_sbj > .source'))
            date = date.split('|')[1].strip()
            date = re.sub('<.*?>', '', date)
            news_date = datetime.strptime(date, '%Y.%m.%d')
            if news_date < datetime.strptime(start_date, '%Y.%m.%d'):
                break

            if news_date > datetime.strptime(end_date, '%Y.%m.%d'):
                continue

            content=str(soup.select('tr > td.view_cnt'))
            content=content.replace("</strong>", ' ')
            content = re.sub('<.*?>', '', content)
            content=content.replace("\n", '')
            content=content.replace("\r", '')
            content = re.sub(r'[▶◎]', '', content)
            report_dict['리포트'].append(content)
            report_dict['Date'].append(news_date + timedelta(days=1))
    if news_date < datetime.strptime(start_date, '%Y.%m.%d'):
                break
    page += 1

report = pd.DataFrame(report_dict)
report = report.groupby('Date').agg({'리포트':'sum'}).reset_index()

df2 = pd.merge(price, report, on='Date', how='outer')
df2 = df2.sort_values(by='Date', ascending=True).reset_index()

for index, row in df2.iterrows():
    if pd.isna(row['Close']):
        if pd.isna(df2.at[index + 1, '리포트']) : df2.at[index + 1, '리포트'] = row['리포트']
        else : df2.at[index + 1, '리포트'] = df2.at[index + 1, '리포트'] + ', ' + row['리포트']
df2 = df2.dropna(subset=['Close'])
df2 = df2.drop(columns=['index'])
df2.to_csv('report_data_{}_{}.csv'.format(stock_name, end_date), mode='w', encoding='utf-8-sig', index=False)

#%%
