# #당뇨병 실습
#
# import matplotlib.pyplot as plt     #그래프 그리는 라이브러리
# import numpy as np
#
# from sklearn import linear_model
# from sklearn import datasets
#
# diabetes_X, diabetes_y=datasets.load_diabetes(return_X_y=True)      #당뇨병 데이터 업로드, diabetes_X, diabetes_y 두개에다가 당뇨병 데이터 업로드
#
# print(diabetes_X.data.shape)       #당뇨병 데이터 모양 확인,크기
# print(diabetes_y.data.shape)
#
# diabetes_X_new=diabetes_X[:, np.newaxis,2]   #[:,2] 1차원의 데이터 구조를 np.newaxis(새로운 차원 만들때 사용)를 통해 2차원으로 유지, 머신러닝 훈련용 데이터는 2차원이어야함
#
# from sklearn.model_selection import train_test_split        #훈련, 테스트 데이터 분할
# X_train,X_test,y_train,y_test=train_test_split(diabetes_X_new,diabetes_y, test_size=0.1, random_state=0)
#
#
# regr = linear_model.LinearRegression()
# regr.fit(X_train, y_train)
#
# y_pred = regr.predict(X_test)
# plt.plot(y_test,y_pred,'.')
# plt.xlim([-0.075, 0.1])
# plt.scatter(X_test,y_test,color='black')
# plt.plot(X_test, y_pred, color='blue',linewidth=3)
# plt.show()

##############################################################################################################

# #선형 회귀 실습 모델
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# X=np.array([2,4,6,8])   #x값에 대한 y값 지정
# y=np.array([81,93,91,97])
#
# plt.scatter(X,y)
# plt.show()
#
# w=10      #w(가중치),b(바이어스)를 0으로 초기화
# b=15
#
# lr=0.03   #hyperparameter, learning rate값 조정, 학습률 조정
#
# epochs=2001     #몇번 반복될지 설정, 반복횟수
#
# n=len(X)        #x데이터의 갯수=>손실(loss)를 구하기 위해 (y^-y)^2을 모두 sum한 값을 갯수로 나눈다
#
# for i in range(epochs):
#     y_pred=w*X+b        #경사하강법을 이용해 w,b 구함, simple regression
#
#     dw=(2/n)*sum(X*(y_pred-y))      #w에대한 편미분을 통해 기울기가 0에 가까이가도록 조정
#     db=(2/n)*sum(y_pred-y)          #b에대한 편미분을 통해 기울기가 0에 가까이가도록 조정
#
#     w=w-lr*dw       #반복을 통해 최적의 w,b값을 찾는다
#     b=b-lr*db
#
#     if i%100==0:    #2001번 반복될때마다 현재의 w값, b값을 출력
#         print("epoch=%.f, 기울기=%.04f,절편=%0.04f"%(i,w,b))
#
# y_pred=w*X+b    #구한 최적의 w,b값을 통해 y절편에 대입해 그래프를 그림
#
# plt.scatter(X,y)       #X값에 대한 y값 점선도로 그래프 출력
# plt.plot(X,y_pred,'r')  #최종적으로 구한 w,b에 대한 최종그래프 직선 출력
# plt.show()      #그래프 출력
#

#################################################################################################################
# #p153, simple linear regression
# import matplotlib.pyplot as plt
# from sklearn import linear_model
#
# reg=linear_model.LinearRegression()     #선형회귀 모델 생성
#
# X=[[174],[152],[138],[128],[186]]     #머신러닝에서 입력은 반드시 2차원 배열 이어야한다.
# y=[71,55,46,38,88]
#
# reg.fit(X,y)        #모델학습
#
# print(reg.predict([[165]]))     #예측할 데이터 값은 반드시 2차원 형태 이어야한다!
#
# plt.scatter(X,y,color='black')      #X,y 값을 점선도로 그림
#
# # y_pred=reg.predict(X)       #X값(학습데이터)를 입력으로 하여 예측값 계산
# #
# # plt.plot(X,y_pred,color='blue',linewidth=3)
# plt.show()


###################################################################################################################

#multiple linear Regression, 다중 선형회귀, 특징 최소 2개이상

import numpy as np
import matplotlib.pyplot as plt

x1=np.array([2,4,6,8])      #특징 2개 출력 1개
x2=np.array([0,4,2,3])
y=np.array([81,93,91,97])

fig=plt.figure()        #그래프 출력 입력 2개 출력1개이므로 3차원 공간상 출력, 데이터의 분포 출력
ax=fig.add_subplot(111,projection='3d')
ax.scatter3D(x1,x2,y);
plt.show()

w1=0    #가중치,바이어스 0으로 초기화
w2=0
b=0

lr=0.025     #hyperparameter(학습율)

epochs=2001     #반복횟수

n=len(x1)       #x1,x2 수가 같으므로 x1만 계산

#경사하강법
for i in range(epochs):
    y_pred=w1*x1 + w2*x2 + b

    dw1=(2/n)*sum(x1*(y_pred-y))        #x1편미분
    dw2 = (2 / n) * sum(x2*(y_pred - y))    #x2편미분
    db = (2 / n) * sum(y_pred - y)  #b편미분

    w1=w1-lr*dw1       #lr(학습율)곱해서 기존값 업뎃
    w2=w2-lr*dw2       #lr(학습율)곱해서 기존값 업뎃
    b=b-lr*db          #lr(학습율)곱해서 기존값 업뎃

    if i % 100==0:
        print("epoch=%.f, 기울기1=%.04f, 기울기2=%.04f, 절편=%0.04f"%(i,w1,w2,b))

print("실제 점수: ",y)
print("예측 점수: ",y_pred)






