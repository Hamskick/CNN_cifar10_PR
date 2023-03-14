#2023-02-05

#Bianry logistic regression(sigmoid)

import pandas as pd
import numpy as np

#사이킷런의 로지스틱 회귀 라이브러리
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#타이타닉 데이터 불러오기
data = pd.read_csv('C:\\Users\\user\\Desktop\\DEEP_LEARNING_PR\\Titanic.csv')

#맨 앞의 데이터 10개 출력
print(data.head(10))
