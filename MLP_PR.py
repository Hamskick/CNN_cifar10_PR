import numpy as np # numpy라이브러리 호출 np로 사용

def actf(x): # 시그모이드 함수 정의
    return 1/(1+np.exp(-x)) # 시그모이드 함수 값 반환

def actf_deriv(x): # 시그모이드 함수의 미분
    return x*(1-x)

# 입력유닛의 개수, 은닉유닛의 개수, 출력유닛의 개수
inputs, hiddens, outputs = 2, 2, 1
learning_rate=0.2 # 학습률 0.2로 지정

X = np.array([[0,0], [0,1], [1,0], [1,1]]) #4개의 feature 
T = np.array([[1],[0],[0],[1]])

W1 = np.array([[0.10,0.20],[0.30,0.40]])
W2 = np.array([[0.50],[0.60]])
B1 = np.array([0.1,0.2])
B2 = np.array([0.3])

# 순방향 전파 계산
def predict(x):
    layer0 = x
    Z1 = np.dot(layer0, W1)+B1
    layer1 = actf(Z1)
    Z2 = np.dot(layer1, W2)+B2
    layer2 = actf(Z2)
    return layer0, layer1, layer2

# 역전파 전파 계산
def fit():
    global W1, W2, B1, B2
    for i in range(90000):
        for x, y in zip(X,T):
            x = np.reshape(x, (1, -1))
            y = np.reshape(y, (1, -1))

            layer0 , layer1, layer2 = predict(x)
            layer2_error = layer2-y
            layer2_delta = layer2_error*actf_deriv(layer2)
            layer1_error = np.dot(layer2_delta, W2.T)
            layer1_delta = layer1_error*actf_deriv(layer1)

            W2 += -learning_rate*np.dot(layer1.T, layer2_delta) #
            W1 += -learning_rate*np.dot(layer0.T, layer1_delta)
            B2 += -learning_rate*np.sum(layer2_delta, axis=0)
            B1 += -learning_rate*np.sum(layer1_delta, axis=0)

def test():
    for x,y in zip(X, T):
        x = np.reshape(x, (1,-1))
        layer0, layer1, layer2 = predict(x)
        print(x, y, layer2)

fit()
test()