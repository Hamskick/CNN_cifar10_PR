# #2023-01-30
#
#
# import pandas as pd
# col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label'] #데이터 열의 이름 부여
# pima = pd.read_csv("C:\\Users\\user\\Desktop\\DEEP_LEARNING_PR\\diabetes.csv", names=col_names)#데이터 불러오기, 각열에 이름 부여
#
# #split dataset in features and target variable
# feature_cols = ['pregnant','glucose', 'bp','skin','insulin', 'bmi', 'pedigree','age']#특징열 저장
# X = pima[feature_cols]  #데이터에서 X에 특징들 저장 # Features
# y = pima.label # Target variable        #라벨값 y에 저장
#
#
# from sklearn.model_selection import train_test_split      #데이터 분할 라이브러리
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)       #random seed 지정, 훈련,테스트용 데이터 8:2분할
#
#
# from sklearn.linear_model import LogisticRegression     #로지스틱 회귀 모델 사용
#
#
# logreg = LogisticRegression(random_state=16, max_iter=1000) #max_iter : 계산에 사용할 작업 수, 로지스틱 모델 사용, parameter: max_iter(계산에 사용할 작업수)
#
#
# logreg.fit(X_train, y_train)        #훈련용 데이터 학습
#
# y_pred = logreg.predict(X_test)     #학습한 모델에 새로운 데이터 넣어서 예측
#
# from sklearn import metrics     #혼동행렬
# cnf_matrix = metrics.confusion_matrix(y_test, y_pred)       #예측한 모델과 테스팅 라벨 비교
# print(cnf_matrix)       #혼동행렬출력
#
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns       #seaborn 시각화 라이브러리, matplotlib도 같이 불러야함
# class_names=[0,1] #레이블의 클레스
# sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
# plt.title('Confusion matrix')
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
# plt.show()      #혼동행렬 출력
#
# #모델의 성능평가
# from sklearn.metrics import classification_report
# target_names = ['without diabetes', 'with diabetes']
# print(classification_report(y_test, y_pred, target_names=target_names))     #모델의 성능지표 출력

#########################################################################################################

# #image dataset 실습
#
#
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import matplotlib.pyplot as plt
# import cv2
# import os
# from random import shuffle
# from PIL import Image
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings('ignore')
# import os
#
# train_messy = "C:\\Users\\user\\Desktop\\DEEP_LEARNING_PR\\images\\images\\train\\messy"
# train_clean= "C:\\Users\\user\\Desktop\\DEEP_LEARNING_PR\\images\\images\\train\\clean"
# test_messy= "C:\\Users\\user\\Desktop\\DEEP_LEARNING_PR\\images\\images\\val\\messy"
# test_clean= "C:\\Users\\user\\Desktop\\DEEP_LEARNING_PR\\images\\images\\train\\clean"
# image_size = 128
#
# image = cv2.imread("C:\\Users\\user\\Desktop\\DEEP_LEARNING_PR\\images\\images\\train\\messy\\73.png", cv2.IMREAD_ANYCOLOR)
# cv2.imshow("73.png.messy", image)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# image = cv2.imread("C:\\Users\\user\\Desktop\\DEEP_LEARNING_PR\\images\\images\\train\\clean\\73.png", cv2.IMREAD_ANYCOLOR)
# cv2.imshow("73.png.clean", image)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# for image in tqdm(os.listdir(train_messy)):
#     path = os.path.join(train_messy, image)
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (image_size, image_size)).flatten()
#     np_img=np.asarray(img)
#
# for image2 in tqdm(os.listdir(train_clean)):
#     path = os.path.join(train_clean, image2)
#     img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     img2 = cv2.resize(img2, (image_size, image_size)).flatten()
#     np_img2=np.asarray(img2)
#
# plt.figure(figsize=(10,10))
# plt.subplot(1, 2, 1)
# plt.imshow(np_img.reshape(image_size, image_size))
# plt.axis('off')
# plt.subplot(1, 2, 2)
# plt.imshow(np_img2.reshape(image_size, image_size))
# plt.axis('off')
# plt.title("Messy and Clean Rooms in GrayScale")
# plt.show()
#
#
# def train_data():
#     train_data_messy = []
#     train_data_clean = []
#     for image1 in tqdm(os.listdir(train_messy)):
#         path = os.path.join(train_messy, image)
#         img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         img1 = cv2.resize(img1, (image_size, image_size))
#         train_data_messy.append(img1)
#     for image2 in tqdm(os.listdir(train_clean)):
#         path = os.path.join(train_clean, image)
#         img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         img2 = cv2.resize(img2, (image_size, image_size))
#         train_data_clean.append(img2)
#
#     train_data = np.concatenate((np.asarray(train_data_messy), np.asarray(train_data_clean)), axis=0)
#     return train_data
#
#
# def test_data():
#     test_data_messy = []
#     test_data_clean = []
#     for image1 in tqdm(os.listdir(test_messy)):
#         path = os.path.join(test_messy, image1)
#         img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         img1 = cv2.resize(img1, (image_size, image_size))
#         test_data_messy.append(img1)
#     for image2 in tqdm(os.listdir(test_clean)):
#         path = os.path.join(test_clean, image2)
#         img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         img2 = cv2.resize(img2, (image_size, image_size))
#         test_data_clean.append(img2)
#
#     test_data = np.concatenate((np.asarray(test_data_messy), np.asarray(test_data_clean)), axis=0)
#     return test_data
#
# train_data = train_data()
# test_data = test_data()
#
# x_data=np.concatenate((train_data,test_data),axis=0)
# x_data = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#
# z1 = np.zeros(96)
# o1 = np.ones(96)
# Y_train = np.concatenate((o1, z1), axis=0)
# z = np.zeros(10)
# o = np.ones(10)
# Y_test = np.concatenate((o, z), axis=0)
#
# y_data=np.concatenate((Y_train,Y_test),axis=0).reshape(x_data.shape[0],1)
#
# print("X shape: " , x_data.shape)
# print("Y shape: " , y_data.shape)
#
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=42)
# number_of_train = x_train.shape[0]
# number_of_test = x_test.shape[0]
#
# x_train_flatten = x_train.reshape(number_of_train,x_train.shape[1]*x_train.shape[2])
# x_test_flatten = x_test .reshape(number_of_test,x_test.shape[1]*x_test.shape[2])
# print("X train flatten",x_train_flatten.shape)
# print("X test flatten",x_test_flatten.shape)

##########################################################################################################


# #이미지 분류
#
# import pandas as pd
# import numpy as np
#
# df= pd.read_excel('C:\\Users\\user\\Desktop\\DEEP_LEARNING_PR\\ex3d1 (1).xlsx', 'X', header=None)
#
#
# import matplotlib.pyplot as plt
# # plt.imshow(np.array(df.iloc[500, :]).reshape(20,20))       #행의 500번데이터,이미지 출력 위해 1차원 데이터를 2차원 20x20으로 변환하여 출력
# # plt.show()
# #
# # plt.imshow(np.array(df.iloc[1750, :]).reshape(20,20))
# # plt.show()
#
# #print(len(df))      #5000개의 행
#
# df_y= pd.read_excel('C:\\Users\\user\\Desktop\\DEEP_LEARNING_PR\\ex3d1 (1).xlsx', 'y', header=None)     #2차원 데이터,같은 엑셀 파일이나 다른 시트에 레이블존재 및 df_y에 파일 저장
#
# y= df_y[0]      #2차원 데이터의 0번방 값들 y에 저장=>1차원
#
#
# for i in range(len(y)):     #범위:0~4999
#      if y[i]!=1:     #모델이 숫자 1만 식별,따라서 레이블이 1아닌경우 레이블 모두 0으로
#         y[i]=0
#
# y = pd.DataFrame(y)     #레이블값을 바꾼것을 dataframe형태로 y에 저장=>2차원
#
# x_train=df.iloc[0:4000].T   #훈련용(feature)데이터:0~4000개,(400x4000)    #5000개의 데이터들중 4000개는 훈련용, 1000개 데이터는 테스트용으로 나눔
# y_train=y.iloc[0:4000].T    #훈련용(label)데이터:0~4000개,(1x4000)
# x_test = df.iloc[4000:].T   #테스트용(feature)data:4000~끝까지, (400x1000)
# y_test = y.iloc[4000:].T    #테스트용(label):4000~끝까지, (1x1000)
#
# x_train=np.array(x_train)       #x_train을 배열형태로 x_train에 저장=>계산을 쉽게 하기위해
# y_train = np.array(y_train)
# x_test = np.array(x_test)
# y_test = np.array(y_test)
#
# def sigmoid(z):     #시그모이드 함수 정의
#     s = 1/(1 + np.exp(-z))
#     return s
#
# def initialize_with_zeros(dim): #w,b에 대한 가중치,바이어스 값 0으로 초기화 함수
#     w = np.zeros(shape=(dim, 1))
#     b = 0
#     return w, b
#
# def propagate(w, b, X, Y):
#     # 훈련 데이터의 수를 구합니다.
#     m = X.shape[1]
#     # 예측된 출력을 계산합니다
#     A = sigmoid(np.dot(w.T, X) + b) #단순선형회귀에 시그모이드 함수 씌운것을 A에 저장
#
#     cost = -1/m * np.sum(Y*np.log(A) + (1-Y) * np.log(1-A))
#
#     dw = 1/m * np.dot(X, (A-Y).T)   # 그래디언트계산,w값
#     db = 1/m * np.sum(A-Y)
#     grads = {"dw": dw, "db": db}    #딕셔너리형태로 dw, db값 저장
#     return grads, cost
#
#
# def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
#     costs = []
#     for i in range(num_iterations):
#         grads, cost = propagate(w, b, X, Y)
#         dw = grads["dw"]
#         db = grads["db"]
#         w = w - learning_rate * dw  #경사하강법을 통해 w값 업데이트
#         b = b - learning_rate * db  #경사하강법을 통해 b값 업데이트
#
#         if i % 100 == 0:
#             costs.append(cost)
#
#             params = {"w": w,"b": b}
#             grads = {"dw": dw,"db": db}
#
#     return params, grads, costs
#
# def predict(w, b, X):
#     m = X.shape[1]
#     Y_prediction = np.zeros((1, m))
#     w = w.reshape(X.shape[0], 1)
#     A = sigmoid(np.dot(w.T, X) + b)
#     for i in range(A.shape[1]):
#         Y_prediction[:, i] = (A[:, i] > 0.5) * 1
#     return Y_prediction         #0.5보다 크면 1예측, 그렇지않으면 0예측
#
#
# def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5):
#     w, b = initialize_with_zeros(X_train.shape[0])
#     parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate,print_cost=False)
#     w = parameters["w"]
#     b = parameters["b"]
#     Y_prediction_test = predict(w, b, X_test)
#     Y_prediction_train = predict(w, b, X_train)
#     print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
#     print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
#     d = {"costs": costs, "Y_prediction_test": Y_prediction_test, "Y_prediction_train" : Y_prediction_train,"w" : w,"b" : b,"learning_rate" : learning_rate,"num_iterations": num_iterations}
#     return d
#
# d = model(x_train,y_train, x_test,y_test, num_iterations = 2500, learning_rate = 0.25)
#
#
#
# plt.figure(figsize=(7,5))
# plt.scatter(x = range(len(d['costs'])), y = d['costs'], color='black')
# plt.title('Scatter Plot of Cost Functions', fontsize=18)
# plt.ylabel('Costs', fontsize=12)
# plt.show()