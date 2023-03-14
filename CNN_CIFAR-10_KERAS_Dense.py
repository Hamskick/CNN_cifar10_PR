import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras import initializers



# CIFAR-10 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test,y_test)=cifar10.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape,y_test.shape)

# Convert to floats
#스케일링 진행
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0

# 데이터 전처리를 위한 ImageDataGenerator 객체를 생성합니다.
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,          # 평균을 0으로 만듭니다.
    featurewise_std_normalization=True # 표준 편차를 1로 만듭니다.
)


# 이미지 데이터를 전처리합니다.
data_generator.fit(x_train)
x_train_normalized = data_generator.standardize(x_train)
x_test_normalized = data_generator.standardize(x_test)


# 레이블을 One-hot encoding
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']        #레이블의 이름을 지정

# from keras.models import Sequential
# from keras.layers import Dropout, Dense
# from keras import initializers
# from keras.layers import BatchNormalization
# #from keras.initializers import RandomNormal


# 신경망 모델 설계
cnn=Sequential()  #선형 스택 모델 사용, 신경망을 레고 조립하듯이 만들 수 있음
He_normal = tf.keras.initializers.he_normal(seed=None)

# cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="valid", activation='relu', input_shape=(32, 32, 3))),
# cnn.add(tf.keras.layers.MaxPooling2D((2, 2), padding="valid")),
# cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="valid")),
# cnn.add(tf.keras.layers.MaxPooling2D((2, 2), padding="valid")),
# cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="valid")),
#, padding="valid"
#특징찾기
cnn.add(Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3)))
cnn.add(Conv2D(32,(3,3),activation='relu'))
cnn.add(MaxPooling2D (pool_size=(2,2)))
cnn.add(Dropout (0.25))

cnn.add(Conv2D(64,(3,3),activation='relu'))

cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(MaxPooling2D (pool_size=(2,2)))
cnn.add(Dropout (0.25))

#영상인식
cnn.add(Flatten())  #3차원을 1차원의 배열 형태로 평탄화(32*32*3=3072)
cnn.add(Dense(512, activation='relu', kernel_initializer=He_normal))       #은닉층, 1700개의 노드를 지니며 활성함수 'relu'
cnn.add(Dropout(0.2))

cnn.add(Dense(64, activation='relu', kernel_initializer=He_normal))      #은닉층, 1200개의 노드를 지니며 활성함수 'relu'
cnn.add(Dropout(0.2))

cnn.add(Dense(10, activation='softmax'))         #출력층의 노드는 10개의 클래스를 지니고 있기때문에 10개 설정, 활성함수는 소프트맥스

cnn.summary()

# 모델 컴파일하기(Adam)
from keras.optimizers import Adam
cnn.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])


from sklearn.model_selection import KFold
# 데이터셋 분리를 위한 k-fold 객체 생성
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# k-fold 교차 검증 수행
for fold_num, (train_indices, val_indices) in enumerate(kfold.split(x_train_normalized)):
    # 현재 fold 번호 출력
    print(f"Fold {fold_num+1}/{kfold.n_splits}")
    print("train_indices: ",train_indices)

# 현재 fold에서 사용할 학습용/검증용 데이터 추출
x_train_fold, y_train_fold = x_train_normalized[train_indices], y_train[train_indices]
x_val_fold, y_val_fold = x_train_normalized[val_indices], y_train[val_indices]



# 모델 학습하기
history=cnn.fit(x_train_fold, y_train_fold, batch_size=1024, epochs=30, validation_data=(x_val_fold, y_val_fold))

# 검증용 데이터에 대한 정확도 출력
val_loss, val_acc = cnn.evaluate(x_val_fold, y_val_fold)
print(f"Fold {fold_num + 1} validation accuracy: {val_acc}")

# 모델 평가하기
loss, accuracy = cnn.evaluate(x_test_normalized, y_test, verbose=2)      #학습한 모델 평가, verbose=0(출력 x),1(자세히),2(함축적 정보 출력)
print(f'Test loss: {loss:.3f}')
print(f'Test accuracy: {accuracy:.3f}')



# loss, val_loss 그래프 출력
# history_dict=history.history
# loss_values=history_dict['loss']
# val_loss_values=history_dict['val_loss']
# acc=history_dict['accuracy']
# epochs=range(1,len(acc)+1)

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


