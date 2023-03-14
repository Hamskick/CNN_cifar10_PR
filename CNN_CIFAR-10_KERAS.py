import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd


# CIFAR-10 데이터 불러오기
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(train_images.shape, train_labels.shape)
print(test_images.shape,test_labels.shape)

# Convert to floats
train_images = train_images.astype('float32') / 255     #0~255의 범위값(R,G,B)을 가지는 픽셀값들을 255로 나눔으로써 0~1사이 값으로 졍규화진행
test_images = test_images.astype('float32') / 255

# 데이터 전처리를 위한 ImageDataGenerator 객체를 생성합니다.
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,          # 평균을 0으로 만듭니다.
    featurewise_std_normalization=True # 표준 편차를 1로 만듭니다.
)


# 이미지 데이터를 전처리합니다.
data_generator.fit(train_images)
x_train_normalized = data_generator.standardize(train_images)
x_test_normalized = data_generator.standardize(test_images)


# 레이블을 One-hot encoding
train_labels = to_categorical(train_labels)     #One-hot-encoding 과정으로 각 샘플의 해당되는 레이블의 값만 1로 지정 하고 나머지는 모두 0으로 처리
test_labels = to_categorical(test_labels)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']        #레이블의 이름을 지정

from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras import initializers
from keras.layers import BatchNormalization
#from keras.initializers import RandomNormal


# 모델 구성하기
model = tf.keras.Sequential()   #선형 스택 모델 사용, 신경망을 레고 조립하듯이 만들 수 있음
He_normal = tf.keras.initializers.he_normal(seed=None)

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="valid", activation='relu', input_shape=(32, 32, 3))),
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="valid")),
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="valid")),
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="valid")),
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="valid")),


#initializer = tf.keras.initializers.GlorotUniform(seed=None)   # 가중치 초기화 방법으로 Glorot uniform 사용
model.add(tf.keras.layers.Flatten())#3차원을 1차원의 배열 형태로 평탄화(32*32*3=3072)

model.add(tf.keras.layers.Dense(1700, activation='relu',   kernel_initializer=He_normal))       #은닉층, 1700개의 노드를 지니며 활성함수 'relu'
model.add(Dropout(0.2))

model.add(tf.keras.layers.Dense(1200, activation='relu',   kernel_initializer=He_normal))      #은닉층, 1200개의 노드를 지니며 활성함수 'relu'
model.add(Dropout(0.2))

model.add(tf.keras.layers.Dense(750, activation='relu',   kernel_initializer=He_normal))       #은닉층, 750개의 노드를 지니며 활성함수 'relu'
model.add(Dropout(0.2))

model.add(tf.keras.layers.Dense(350, activation='relu',  kernel_initializer=He_normal))        #은닉층, 350개의 노드를 지니며 활성함수 'relu'
model.add(Dropout(0.2))

model.add(tf.keras.layers.Dense(150, activation='relu', kernel_initializer=He_normal))        #은닉층, 150개의 노드를 지니며 활성함수 'relu' 따라서 은닉층 5개의 구조이다
model.add(Dropout(0.2))

model.add(tf.keras.layers.Dense(10, activation='softmax'))         #출력층의 노드는 10개의 클래스를 지니고 있기때문에 10개 설정, 활성함수는 소프트맥스

model.summary()

# 모델 컴파일하기(Adam)
from keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])


from sklearn.model_selection import KFold
# 데이터셋 분리를 위한 k-fold 객체 생성
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# k-fold 교차 검증 수행
for fold_num, (train_indices, val_indices) in enumerate(kfold.split(x_train_normalized)):
    # 현재 fold 번호 출력
    print(f"Fold {fold_num+1}/{kfold.n_splits}")
    print("train_indices: ",train_indices)

# 현재 fold에서 사용할 학습용/검증용 데이터 추출
x_train_fold, y_train_fold = x_train_normalized[train_indices], train_labels[train_indices]
x_val_fold, y_val_fold = x_train_normalized[val_indices], train_labels[val_indices]



# 모델 학습하기
history=model.fit(x_train_fold, y_train_fold, batch_size=len(x_train_fold), epochs=50, validation_data=(x_val_fold, y_val_fold))

# 검증용 데이터에 대한 정확도 출력
val_loss, val_acc = model.evaluate(x_val_fold, y_val_fold)
print(f"Fold {fold_num + 1} validation accuracy: {val_acc}")

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

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss Plot')
plt.legend(['train error','val error'],loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


