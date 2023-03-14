# #2023-01-12, Iris(붓꽃) 데이터 머신러닝 p.123
# from sklearn import datasets
#
# iris=datasets.load_iris()
# print(iris)
#
# from sklearn.model_selection import  train_test_split
#
# X=iris.data             #데이타(4개의 붓꽃 특징 사용)_150개의 샘플
# y=iris.target           #레이블(붓꽃의 종류 3가지를->0,1,2 레이블로 지정)_150개의 샘플
#
# print(X.shape)          #(150, 4) 150x4 metrics
# print(y.shape)          #(150, )
#
# #(80:20)으로 분할
# X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=4)  # 70% 훈련데이터,30%테스트 데이터로 분할
#                                                                                         # random_state:불규칙한 난수속에 규칙을 만들어 줌,random_state의 숫자는 none일경우 규칙 x
#                                                                                         # 숫자 지정시 난수 생성에 규칙이 존재=>동일한 결과 나옴
#
# print(X_train.shape)    #훈련용 데이터, 붓꽃4가지 특징 (105, 4) 105x4 metrics
# print(X_test.shape)     #테스트용 데이터 (45, 4) 45x4 metrics
# print(y_train.shape)    #훈련용 데이터 (105, ) 레이블
#
# from sklearn.neighbors import KNeighborsClassifier  #sklearn 라이브러리(전통적인 머신러닝), knn모델사용(hyperparameter: k, distance) =>사이킷런에서 knn알고리즘 불러오기
#
# k_i=0
# k_scores=0
#
# for i in range(1,75):
#     knn=KNeighborsClassifier(n_neighbors=i)     #k=6으로 지정(hyperparameter), 새로운 데이터를 기준으로 거리순으로 sorting 후 이웃한 6개중 갯수가 많은쪽에 속한다.k범위 X_train 갯수까지
#     knn.fit(X_train, y_train)                   #훈련데이터 학습! 훈련데이터중 X_data와 y(레이블)을 훈련=>학습시킨다
#     y_pred=knn.predict(X_test)          #모델에 새로운 값(테스트용 데이터)을 넣어 예측
# # print(y_pred)                   #[2 0 2 2 2 1 2 0 0 2 0 0 0 1 2 0 1 0 0 2 0 2 1 0 0 0 0 0 0 2] 레이블 값
# # print(y_pred.shape)             #(30, )
#
#
#     from sklearn import metrics
#     scores=metrics.accuracy_score(y_test, y_pred)   #X_test의 레이블 값인 y_test를 모델을 통해 예측한 것과 비교 한뒤 정확도를 나타냄
#
#
#     if scores>k_scores:
#         k_scores=scores
#         k_i=i
#     print(i,scores)
#
# print(k_i, k_scores)
#
#
#
# #혼동행렬 그림으로 출력
# import matplotlib.pyplot as plt
# confusion_matrix = metrics.confusion_matrix(y_test, y_pred)     #테스트 결과값과, 예측값을 confusion_matrix에 입력
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1,2])  #레이블은 0,1,2이기때문에 cm_display에 저장후 그림으로 출력
# cm_display.plot()
# plt.show()      #그림으로 출력


# #완전히 새로운값을 넣어 예측해보기
#
# classes={0:'setosa', 1:'versicolor', 2:'virginica'}
#
# x_new=[[3,4,5,2],[5,4,2,2]]     #새로운 test데이터 설정
#
# y_predict=knn.predict(x_new)    #새로운 test데이터를 모델에 넣어 예측
#
# print(classes[y_predict[0]])
# print(classes[y_predict[1]])


###################################################################################################
#MNIST 실습 p.126

import matplotlib.pyplot as plt     #그래프 나타내기위한 라이브러리
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

digits=datasets.load_digits()   #MNIST 안에 DATA셋 업로드
# plt.imshow(digits.images[0],cmap=plt.cm.gray_r, interpolation='nearest')    #데이터의 0번째 숫자, 그레이컬러, 보간법: 최근점 보간법
# plt.show() #이미지 출력

n_samples=len(digits.images)
data=digits.images.reshape((n_samples, -1))     #1796개의 샘플, (8,8)->(64, ) 2차원에서 1차원으로 크기 변환

from sklearn.neighbors import KNeighborsClassifier

score_i=0
k_i=0
for i in range(30,450):
    knn=KNeighborsClassifier(n_neighbors=i)     #knn모델 지정,k=i

    X_train, X_test, y_train, y_test=train_test_split(data, digits.target, test_size=0.2)   #훈련용데이터와 테스트용 데이터 8:2분할, 1차원으로 변환된 data사용 해야함!

    knn.fit(X_train,y_train)        #knn모델에 학습용 데이터 학습

    y_pred=knn.predict(X_test)      #X_test용 데이터를 모델을 통해 예측

    scores=metrics.accuracy_score(y_test, y_pred)   #예측한 것을 테스트(20%로 빼놓은)용 레이블과 비교하여 정확도 측정

    if scores>score_i:
       score_i=scores
       k_i=i

    print(i, score_i)

print(k_i,score_i)


# plt.imshow(X_test[0].reshape(8,8),cmap=plt.cm.gray_r, interpolation='nearest')     #이미지 출력시 1차원->2차원으로 변경후 출력
# plt.show()
# y_pred=knn.predict([X_test[10]])
# print(y_pred)


#혼동행렬 그림으로 출력
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)     #
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1,2,3,4,5,6,7,8,9])
cm_display.plot()
plt.show()

##############################################
#혼동행렬 사용 p.131

# import matplotlib.pyplot as plt
#
# from sklearn import datasets, metrics
# from sklearn.model_selection import train_test_split
#
# digits=datasets.load_digits()       #data 업로드
# n_samples=len(digits.images)
# data=digits.images.reshape((n_samples, -1))     #1차원으로 변경, 일반적인 머신러닝 알고리즘은 특징들을 2차원->1차원
#
# from sklearn.neighbors import KNeighborsClassifier
# knn=KNeighborsClassifier(n_neighbors=6)     #knn모델 사용
#
# X_train,X_test, y_train,y_test=train_test_split(data, digits.target, test_size=0.2)     #훈련,평가용 데이터 분할 8:2
#
# knn.fit(X_train, y_train)   #knn모델을 통한 훈련용 데이터 학습,
# y_pred=knn.predict(X_test)  #새로운 테스트용 데이터를 통해 모델에 넣어 예측
#
# #혼동행렬 그림으로 출력
# confusion_matrix = metrics.confusion_matrix(y_test, y_pred)     #테스트 결과값과, 예측값을 confusion_matrix에 입력
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1,2,3,4,5,6,7,8,9])
# cm_display.plot()
# plt.show()

##########################################################################################
# #책 이외의 dataset 사용
#
# #유방암 데이터 p.120
# from sklearn import datasets
#
# breast=datasets.load_breast_cancer()
# print(breast)
#
# from sklearn.model_selection import train_test_split
#
# X=breast.data             #데이타(4개의 붓꽃 특징 사용)_150개의 샘플
# y=breast.target           #레이블(붓꽃의 종류 3가지를->0,1, 레이블로 지정)_150개의 샘플
#
# print(X.shape)          #(569, 30) 569x30 metrics
# print(y.shape)          #(569, )
#
# #(80:20)으로 분할
# X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=4)  # 80% 훈련데이터,20%테스트 데이터로 분할
#                                                                                         # random_state:불규칙한 난수속에 규칙을 만들어 줌,random_state의 숫자는 none일경우 규칙 x
#                                                                                         # 숫자 지정시 난수 생성에 규칙이 존재=>동일한 결과 나옴
#
# print(X_train.shape)    #훈련용 데이터, 붓꽃4가지 특징 (455, 30) 455x30 metrics
# print(X_test.shape)     #테스트용 데이터 (114, 30) 114x30 metrics
# print(y_train.shape)    #훈련용 데이터 (455, ) 레이블
# print(y_test.shape)     #테스트용 데이터 (114, ) 레이블
#
# from sklearn.neighbors import KNeighborsClassifier  #sklearn 라이브러리(전통적인 머신러닝), knn모델사용(hyperparameter: k, distance) =>사이킷런에서 knn알고리즘 불러오기
#
# score_i=0
# k_i=0
# for i in range(25,159):
#     knn=KNeighborsClassifier(n_neighbors=i)     #k=? 지정(hyperparameter), 새로운 데이터를 기준으로 거리순으로 sorting 후 이웃한 갯수가 많은쪽에 속한다.
#     knn.fit(X_train, y_train)                   #훈련데이터 학습! 훈련데이터중 X_data와 y(레이블)을 훈련=>학습시킨다
#
#     y_pred=knn.predict(X_test)          #모델에 새로운 값(테스트용 데이터)을 넣어 예측
# # print(y_pred)                   #[2 0 2 2 2 1 2 0 0 2 0 0 0 1 2 0 1 0 0 2 0 2 1 0 0 0 0 0 0 2] 레이블 값
# # print(y_pred.shape)             #(30, )
#
#     from sklearn import metrics
#     scores=metrics.accuracy_score(y_test, y_pred)   #X_test의 레이블 값인 y_test를 모델을 통해 예측한 것과 비교 한뒤 정확도를 나타냄
#     print(scores)
#     if scores>score_i:
#         score_i=scores
#         k_i=i
#
#         print(i, score_i)
#
# print(k_i,score_i)
#
# #혼동행렬 그림으로 출력
# import matplotlib.pyplot as plt
# confusion_matrix = metrics.confusion_matrix(y_test, y_pred)     #테스트 결과값과, 예측값을 confusion_matrix에 입력
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [True,False])  #레이블은 0,1=>진실 or 거짓이기때문에 cm_display에 저장후 그림으로 출력
# cm_display.plot()
# plt.show()      #그림으로 출력


