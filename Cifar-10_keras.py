import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# CIFAR-10 데이터 불러오기
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images
train_labels

test_images
test_labels
# 이미지 데이터 reshape
x_train = np.reshape(train_images, (len(train_images), 32*32*3))
x_test = np.reshape(test_images, (len(test_images), 32*32*3))

# Convert to floats
train_images = train_images.astype('float32') / 255     #0~255의 범위값(R,G,B)을 가지는 픽셀값들을 255로 나눔으로써 0~1사이 값으로 졍규화진행
test_images = test_images.astype('float32') / 255

# Calculate the mean and standard deviation of the training set
mean = np.mean(train_images,axis=(0,1,2))
std = np.std(train_images,axis=(0,1,2))

# Normalize the data
train_images = (train_images - mean) / std
test_images = (test_images - mean) / std



# 레이블을 One-hot encoding
train_labels = to_categorical(train_labels)     #One-hot-encoding 과정으로 각 샘플의 해당되는 레이블의 값만 1로 지정 하고 나머지는 모두 0으로 처리
test_labels = to_categorical(test_labels)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']        #레이블의 이름을 지정

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# # Create an ImageDataGenerator object for data augmentation
# datagen = ImageDataGenerator(
#     width_shift_range=0.1, # randomly shift images horizontally
#     height_shift_range=0.1, # randomly shift images vertically
#     horizontal_flip=True, # randomly flip images horizontally
#     rotation_range=10, # randomly rotate images
#     zoom_range=0.1 # randomly zoom in or out on images
# )

# # Fit the ImageDataGenerator on the training data
# datagen.fit(train_images)
#
# # Create a new generator with augmented data
# augmented_train_generator = datagen.flow(train_images, train_labels, batch_size=32)

# plt.figure(figsize=(10, 10))       #10x10 사이즈의 그림 만들기
# for i in range(25):     #범위 25지정
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])      #x축의 눈금 표시 안함
#     plt.yticks([])      #y축 눈금 표시 안함
#     plt.grid(False)     #그림에서격자제거
#     plt.imshow(test_images[i])      #훈련용 사진 0부터 24까지 총 25개 이미지 출력
#     plt.xlabel(class_names[train_labels[i][0]])      #x축에 각 이미지의 맞는 class_names  출력
# plt.show()





from tensorflow.keras.models import Sequential

# 모델 구성하기
model = tf.keras.Sequential([    #선형 스택 모델 사용, 신경망을 레고 조립하듯이 만들 수 있음
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),   #3차원을 1차원의 배열 형태로 평탄화(32*32*3=3072)
    tf.keras.layers.Dense(512, activation='relu'),       #은닉층, 512개의 노드를 지니며 활성함수 'sigmoid'
    tf.keras.layers.Dense(256, activation='relu'),       #은닉층, 256개의 노드를 지니며 활성함수 'sigmoid'
    tf.keras.layers.Dense(64, activation='relu'),        #은닉층, 64개의 노드를 지니며 활성함수 ''sigmoid' 따라서 은닉층 3개의 구조이다
    tf.keras.layers.Dense(10, activation='softmax')         #출력층의 노드는 10개의 클래스를 지니고 있기때문에 10개 설정, 활성함수는 소프트맥스
])

# 모델 컴파일하기(sgd)
sgd = tf.keras.optimizers.SGD(learning_rate=0.2,momentum=0.9)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# # 모델 컴파일하기(Adagrad)
# from keras.optimizers import Adagrad
# model.compile(optimizer=Adagrad(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
#
# # 모델 컴파일하기(RMSprop)
# from keras.optimizers import RMSprop
# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])
#
# from keras.losses import BinaryCrossentropy
# # 모델 컴파일하기(Adam)
# from keras.optimizers import Adam
# model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])



# # 조기 종료 콜백
# early_stopping = EarlyStopping(monitor='val_loss', patience=3)  #monitor: 기준되는 값, patience: monitor 값 되는 값의 개선이 없는경우


# 모델 학습하기
history=model.fit(train_images, train_labels, epochs=10, batch_size=len(train_images), validation_split=0.2)

# 모델 평가하기
loss, accuracy = model.evaluate(test_images, test_labels, verbose=2)      #학습한 모델 평가, verbose=0(출력 x),1(자세히),2(함축적 정보 출력)
print(f'Test loss: {loss:.3f}')
print(f'Test accuracy: {accuracy:.3f}')


# loss, val_loss 그래프 출력
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()




# #예측하기
# predictions = model.predict(x_test)
#
#
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