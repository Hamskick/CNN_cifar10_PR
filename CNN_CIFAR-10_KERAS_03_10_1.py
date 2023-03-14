import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras import initializers
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.layers import BatchNormalization



# CIFAR-10 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test,y_test)=cifar10.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape,y_test.shape)

# Convert to floats
#스케일링 진행
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0

# 레이블을 One-hot encoding
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']        #레이블의 이름을 지정


# #데이터 증대
# x_train_normalized=x_train_normalized[0:12,]; y_train=y_train[0:12,]   #앞 12개만 증대적용
#
# # 앞 12개 영상을 그려줌
# plt.figure(figsize=(16,2))
# plt.suptitle("First 12 images in the train set")
#
# for i in range(12):
#     plt.subplot(1,12,i+1)
#     plt.imshow(x_train_normalized[i])
#     plt.xticks([]); plt.yticks([])
#     plt.title(class_names[int(y_train[i])])
#
# # 영상 증대기 생성
# batch_siz=6   # 한 번에 생성하는 양
# generator=ImageDataGenerator(rotation_range=30.0, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
# gen=generator.flow(x_train_normalized, y_train, batch_size=batch_siz)
#
#
# # 첫 번째 증대하고 그리기
# img, label=gen.next()
# plt.figure(figsize=(16,3))
# plt.suptitle("Generatior trial 1")
# for i in range (batch_siz):
#     plt.subplot(1,batch_siz,i+1)
#     plt.imshow (img[i])
#     plt.xticks([]); plt.yticks([])
#     plt.title(class_names[int(label[i])])
#
# # 두 번째 증대하고 그리기
# img, label=gen.next ()
# plt.figure(figsize=(16,3))
# plt.suptitle("Generatior trial 2")
# for i in range (batch_siz):
#     plt.subplot(1,batch_siz,i+1)
#     plt.imshow(img[i])
#     plt.xticks([]); plt.yticks([])
#     plt.title(class_names[int(label[i])])


# from sklearn.model_selection import train_test_split
# (x_train_images, x_val, y_train_labels, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# 신경망 모델 설계
cnn=Sequential()  #선형 스택 모델 사용, 신경망을 레고 조립하듯이 만들 수 있음

#특징찾기
cnn.add(Conv2D(32,(3,3), activation='relu', padding='same', input_shape=(32,32,3)))
# cnn.add(BatchNormalization())  # 배치 정규화 추가
cnn.add(MaxPooling2D(pool_size=(2,2), padding='same'))

cnn.add(Conv2D(64,(3,3),activation='relu', padding='same'))
# cnn.add(BatchNormalization())  # 배치 정규화 추가
cnn.add(MaxPooling2D (pool_size=(2,2), padding='same'))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(64,(3,3),activation='relu', padding='same'))
# cnn.add(BatchNormalization())  # 배치 정규화 추가
cnn.add(MaxPooling2D (pool_size=(2,2), padding='same'))
cnn.add(Dropout(0.25))


cnn.add(Conv2D(32,(3,3),activation='relu', padding='same'))
# cnn.add(BatchNormalization())  # 배치 정규화 추가
cnn.add(MaxPooling2D (pool_size=(2,2), padding='same'))
cnn.add(Dropout(0.25))

#영상인식
cnn.add(Flatten())  #평탄화 진행
cnn.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# cnn.add(BatchNormalization())  # 배치 정규화 추가#은닉층,512개의 노드를 지니며 활성함수 'relu'
cnn.add(Dropout(0.2))


cnn.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))        #은닉층, 128개의 노드를 지니며 활성함수 'relu' 따라서 은닉층 5개의 구조이다
# cnn.add(BatchNormalization())  # 배치 정규화 추가
cnn.add(Dropout(0.2))

cnn.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))        #은닉층, 128개의 노드를 지니며 활성함수 'relu' 따라서 은닉층 5개의 구조이다
# cnn.add(BatchNormalization())  # 배치 정규화 추가
cnn.add(Dropout(0.2))


cnn.add(Dense(10, activation='softmax',kernel_regularizer=regularizers.l2(0.01)))         #출력층의 노드는 10개의 클래스를 지니고 있기때문에 10개 설정, 활성함수는 소프트맥스

cnn.summary()

# 모델 컴파일하기(Adam)
from keras.optimizers import Adam
cnn.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])


# #영상 증대기 활용
# generator=ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, rotation_range=10.0)
# # gen=generator.flow(x_train_normalized, y_train, batch_size=batch_siz)



from sklearn.model_selection import KFold
# 데이터셋 분리를 위한 k-fold 객체 생성
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# k-fold 교차 검증 수행
for fold_num, (train_indices, val_indices) in enumerate(kfold.split(x_train)):
    # 현재 fold 번호 출력
    print(f"Fold {fold_num+1}/{kfold.n_splits}")
    print("train_indices: ",train_indices)

# 현재 fold에서 사용할 학습용/검증용 데이터 추출
x_train_fold, y_train_fold = x_train[train_indices], y_train[train_indices]
x_val_fold, y_val_fold = x_train[val_indices], y_train[val_indices]



# 모델 학습하기
history=cnn.fit(x_train_fold, y_train_fold, batch_size=128, epochs=200, validation_data=(x_val_fold, y_val_fold))

# # 검증용 데이터에 대한 정확도 출력
# val_loss, val_acc = cnn.evaluate(x_test_normalized, y_test)


for layer in cnn.layers:        #컨볼루션층의 커널을 시각화
    if 'conv' in layer.name:
        kernel,biases=layer.get_weights()
        print(layer.name,kernel.shape)   # 커널의 텐서 모양을 출력


kernel,biases=cnn.layers[5].get_weights()  # 층 0의 커널 정보를 저장
minv,maxv=kernel.min(),kernel.max()
kernel=(kernel-minv) /(maxv-minv)
n_kernel=64     #커널갯수.

plt.figure(figsize=(24,3))
plt.suptitle("Kernels of conv2d")

for i in range(n_kernel):   #i번째 채널
    f=kernel[:,:,:,i]
    for j in range(3):  # j번째 채널
        plt.subplot (3,n_kernel,j*n_kernel+i+1)
        plt.imshow(f[:,:,j],cmap='gray')
        plt.xticks([]); plt.yticks([])
        plt.title(str(i)+'_'+str(j))
plt.show()


for layer in cnn. layers:       #컨볼루션층의특징 맵을 시각화
    if 'conv' in layer.name:
        print(layer.name, layer.output.shape)   # 특징 맵의 텐서 모양을 출력


from keras.models import Model

partial_model=Model(inputs=cnn.inputs, outputs=cnn.layers[5].output) # 층 0만 떼어냄
partial_model.summary()
feature_map=partial_model.predict(x_test)       #부분 모델로 테스트 집합을 예측
fm=feature_map[1]       #1번영상의 특징 맴을 시각화

plt.imshow(x_test[1])       #1번영상 출력


plt.figure(figsize=(20,6))
plt.suptitle("Feature maps of conv2d")
for i in range(64):
    plt.subplot (4,16,i+1)
    plt.imshow(fm[:,:,i], cmap='gray')
    plt.xticks([]); plt.yticks([])
    plt.title("map"+str(i))
plt.show()






# 모델 평가하기
loss, accuracy = cnn.evaluate(x_test, y_test, verbose=2)      #학습한 모델 평가, verbose=0(출력 x),1(자세히),2(함축적 정보 출력)
print(f'Test loss: {loss:.3f}')
print(f'Test accuracy: {accuracy:.3f}')


# 손실 함수 그래프
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.grid()
plt.show()

# 정확률 그래프
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.grid()
plt.show()


# Making the Predictions
pred = cnn.predict(x_test)
print(pred)

# Converting the predictions into label index
pred_classes = np.argmax(pred, axis=1)
print(pred_classes)


# Plotting the Actual vs. Predicted results

fig, axes = plt.subplots(5, 5, figsize=(15,15))
axes = axes.ravel()

for i in np.arange(0, 25):
    axes[i].imshow(x_test[i])
    axes[i].set_title("True: %s \nPredict: %s" % (class_names[np.argmax(y_test[i])], class_names[pred_classes[i]]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)
plt.show()


