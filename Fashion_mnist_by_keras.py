import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']     #데이터 셋에 클래스 이름이 들어있지 않기 때문에 별도로 이름 만들어 지정

#입력 값을 정규화 함으로써 최적의 매개변수를 보다 빨리 습득, 값의 범위:0~1 사이
train_images = train_images / 255.0
test_images = test_images / 255.0

# 데이터 전처리를 위한 ImageDataGenerator 객체를 생성합니다.
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,          # 평균을 0으로 만듭니다.
    featurewise_std_normalization=True # 표준 편차를 1로 만듭니다.
)

# 이미지 데이터를 전처리합니다.
data_generator.fit(train_images.reshape(60000, 28, 28, 1))
x_train_normalized = data_generator.standardize(train_images)
x_test_normalized = data_generator.standardize(test_images)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train_normalized, train_labels, test_size=0.2, random_state=42)


from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras import initializers
He_normal = tf.keras.initializers.he_normal(seed=None)



#모델 형성
model = tf.keras.Sequential()   #선형 스택 모델 사용, 신경망을 레고 조립하듯이 만들 수 있음
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))   #이미지이기때문에 1차원 배열로 평탄화 진행, 28*28=784개의 노드, input_shape:입력층

model.add(tf.keras.layers.Dense(128, activation='relu'))       #은닉층, 128개의 노드를 지니며 활성함수 'relu' ,,   kernel_initializer=He_normal)
model.add(Dropout(0.2))
model.add(tf.keras.layers.Dense(64, activation='relu'))       #은닉층, 64개의 노드를 지니며 활성함수 'relu',,   kernel_initializer=He_normal)
model.add(Dropout(0.2))

model.add(tf.keras.layers.Dense(10, activation='softmax'))         #출력층, 10개의 노드(0~9 label), 활성함수 softmax, 반환된 10개의 확률 총합=1, 10개의 노드중 하나에 속할 확률 출력

model.summary()

#모델 컴파일
model.compile(optimizer='adam',   #최적화 보폭과 방향을 모두 고려한 adam 사용
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),   #손실함수는 sapeseCategoricalCrossentropy 사용, 레이블 0~9 까지 정수로 표현되어 있기때문에, 레이블을 원핫 인코딩으로 변환 하지 않아도 손실함수 계산
              metrics=['accuracy'])


# 모델 학습하기
history=model.fit(x_train, y_train, batch_size=len(x_train),epochs=50, validation_data=(x_val,y_val))

# 모델 평가하기
loss, accuracy = model.evaluate(x_test_normalized, test_labels, verbose=2)      #학습한 모델 평가, verbose=0(출력 x),1(자세히),2(함축적 정보 출력)
print(f'Test loss: {loss:.3f}')
print(f'Test accuracy: {accuracy:.3f}')


# loss, val_loss 그래프 출력
history_dict=history.history
loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']
acc=history_dict['accuracy']
epochs=range(1,len(acc)+1)

plt.figure(1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss Plot')
plt.legend(['train error','val error'],loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()