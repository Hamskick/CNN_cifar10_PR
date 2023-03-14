import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd

# CIFAR-10 데이터 불러오기
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# from PIL import Image
# # 이미지 크기 조정
# new_size = (64, 64)
# x_train_resized = []
# x_test_resized = []
#
# for i in range(len(train_images)):
#     img = Image.fromarray(train_images[i])
#     img_resized = img.resize(new_size)
#     x_train_resized.append(np.array(img_resized))
#
# for i in range(len(train_images)):
#     img = Image.fromarray(test_images[i])
#     img_resized = img.resize(new_size)
#     x_test_resized.append(np.array(img_resized))
#
# x_train_resized = np.array(x_train_resized)
# x_test_resized = np.array(x_test_resized)

train_images
train_labels

test_images
test_labels

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

# plt.figure(figsize=(8,6))
# plt.imshow(x_train_normalized[0])

# 레이블을 One-hot encoding
train_labels = to_categorical(train_labels)     #One-hot-encoding 과정으로 각 샘플의 해당되는 레이블의 값만 1로 지정 하고 나머지는 모두 0으로 처리
test_labels = to_categorical(test_labels)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']        #레이블의 이름을 지정


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train_normalized, train_labels, test_size=0.2, random_state=42)


from keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(
    width_shift_range=0.1,  # 이미지를 가로로 랜덤 이동 (이동 비율)
    height_shift_range=0.1, # 이미지를 세로로 랜덤 이동 (이동 비율)
    horizontal_flip=True,   # 이미지를 가로로 랜덤 뒤집기
    rotation_range=10,      # 이미지를 랜덤 회전 (회전 각도 범위)
    zoom_range=0.1          # 이미지를 랜덤 확대/축소 (확대/축소 비율 범위)
)

gen.fit(x_train_normalized)



# plt.figure(figsize=(10, 10))       #10x10 사이즈의 그림 만들기
# for i in range(25):     #범위 25지정
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])      #x축의 눈금 표시 안함
#     plt.yticks([])      #y축 눈금 표시 안함
#     plt.grid(False)     #그림에서격자제거
#     plt.imshow(test_images[i])      #훈련용 사진 0부터 24까지 총 25개 이미지 출력
#     plt.xlabel(class_names[train_labels[i][0]])      #x축에 각 이미지의 맞는 class_names  출력
# plt.show()


from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras import initializers
from keras.layers import BatchNormalization

#from keras.initializers import RandomNormal

# 모델 구성하기
He_normal = tf.keras.initializers.he_normal(seed=None)#
#initializer = tf.keras.initializers.GlorotUniform(seed=None)   # 가중치 초기화 방법으로 Glorot uniform 사용
model = tf.keras.Sequential()   #선형 스택 모델 사용, 신경망을 레고 조립하듯이 만들 수 있음
model.add(tf.keras.layers.Flatten(input_shape=(32, 32, 3)))#3차원을 1차원의 배열 형태로 평탄화(32*32*3=3072)

model.add(tf.keras.layers.Dense(1700, activation='sigmoid'))      #은닉층, 1700개의 노드를 지니며 활성함수 'relu'
model.add(Dropout(0.2))

model.add(tf.keras.layers.Dense(1200, activation='sigmoid'))      #은닉층, 1200개의 노드를 지니며 활성함수 'relu'
model.add(Dropout(0.2))

model.add(tf.keras.layers.Dense(750, activation='sigmoid'))       #은닉층, 750개의 노드를 지니며 활성함수 'relu'
model.add(Dropout(0.2))

model.add(tf.keras.layers.Dense(350, activation='sigmoid'))        #은닉층, 350개의 노드를 지니며 활성함수 'relu'
model.add(Dropout(0.2))

model.add(tf.keras.layers.Dense(150, activation='sigmoid'))        #은닉층, 150개의 노드를 지니며 활성함수 'relu' 따라서 은닉층 5개의 구조이다
model.add(Dropout(0.2))

model.add(tf.keras.layers.Dense(10, activation='sigmoid'))         #출력층의 노드는 10개의 클래스를 지니고 있기때문에 10개 설정, 활성함수는 소프트맥스

model.summary()

# 모델 컴파일하기(sgd)
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# # 모델 컴파일하기(Adagrad)
# from keras.optimizers import Adagrad
# model.compile(optimizer=Adagrad(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# # 모델 컴파일하기(RMSprop)
# from keras.optimizers import RMSprop
# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])

from keras.losses import BinaryCrossentropy
# # 모델 컴파일하기(Adam)
# from keras.optimizers import Adam
# model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])



# # 조기 종료 콜백
# early_stopping = EarlyStopping(monitor='val_loss', patience=3)  #monitor: 기준되는 값, patience: monitor 값 되는 값의 개선이 없는경우


# 모델 학습하기
history=model.fit(gen.flow(x_train, y_train, batch_size=len(x_train)),  epochs=150, validation_data=(x_val,y_val))
#steps_per_epoch=len(x_train)/64,
# 모델 평가하기
loss, accuracy = model.evaluate(x_test_normalized, test_labels, verbose=2)      #학습한 모델 평가, verbose=0(출력 x),1(자세히),2(함축적 정보 출력)
print(f'Test loss: {loss:.3f}')
print(f'Test accuracy: {accuracy:.3f}')


# pd.DataFrame(history.history).plot()
# plt.show()

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




# #예측하기
# predictions = model.predict(x_test)


# # 그림으로 출력 하기 위해 정의
# def plot_image(i, predictions_array, true_label, img):
#     predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#
#     plt.imshow(img, cmap=plt.cm.binary)
#
#     predicted_label = np.argmax(predictions_array)
#     if predicted_label == true_label.all():
#         color = 'blue'
#     else:
#         color = 'red'
#
#     plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                          100 * np.max(predictions_array),
#                                          class_names[true_label[0]]),
#                color=color)
#
#
# def plot_value_array(i, predictions_array, true_label):
#     predictions_array, true_label = predictions_array[i], true_label[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     thisplot = plt.bar(range(10), predictions_array, color="#777777")
#     plt.ylim([0, 1])
#     predicted_label = np.argmax(predictions_array)
#
#     thisplot[predicted_label].set_color('red')
#     thisplot[true_label[0]].set_color('blue')
#
# i = 9992
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, y_test, x_test)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,  y_test)
# plt.show()


