# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images.shape    #훈련용 이미지 형태 확인, 60000장(28x28)픽셀 형태

train_labels.shape    #훈련용 라벨:60000장

test_images.shape

test_labels   #0~9까지 옷종류 클래스 라벨

train_images

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']     #데이터 셋에 클래스 이름이 들어있지 않기 때문에 별도로 이름 만들어 지정

#데이터 확인
plt.figure()
plt.imshow(train_images[1])   #train_images의 첫번째 샘플 이미지
plt.colorbar()    #오른쪽 픽셀 값의 범위 0~255 출력
plt.grid(False)   #False를 함으로써 이미지의 격자 제거
plt.show()    #이미지 출력

#입력 값을 정규화 함으로써 최적의 매개변수를 보다 빨리 습득, 값의 범위:0~1 사이
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])  #모든 xticks 제거, x값의 범위 나타내는것 제거
    plt.yticks([])
    plt.grid(False)   #격자 제거
    plt.imshow(train_images[i])   #훈련용_이미지 25장 출력
    plt.xlabel(class_names[train_labels[i]])    #앞에 지정했던 클래스 이름들 25개 차례대로 x라벨에 지정
plt.show()

model = tf.keras.Sequential([  #선형 스택 모델 사용, 신경망을 레고 조립하듯이 만들 수 있음
    tf.keras.layers.Flatten(input_shape=(28, 28)),    #이미지이기때문에 1차원 배열로 평탄화 진행, 28*28=784개의 노드, input_shape:입력층
    tf.keras.layers.Dense(128, activation='relu'),    #은닉층, 128개의 노드, 활성 함수= relu
    tf.keras.layers.Dense(10, activation='softmax')   #출력층, 10개의 노드(0~9 label), 활성함수 softmax, 반환된 10개의 확률 총합=1, 10개의 노드중 하나에 속할 확률 출력
])

model.compile(optimizer='sgd',   #최적화 보폭과 방향을 모두 고려한 adam 사용
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),   #손실함수는 sapeseCategoricalCrossentropy 사용, 레이블 0~9 까지 정수로 표현되어 있기때문에, 레이블을 원핫 인코딩으로 변환 하지 않아도 손실함수 계산
              metrics=['accuracy'])

model.compile(optimizer='sgd',   #최적화 보폭과 방향을 모두 고려한 adam 사용
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),   #손실함수는 sapeseCategoricalCrossentropy 사용, 레이블 0~9 까지 정수로 표현되어 있기때문에, 레이블을 원핫 인코딩으로 변환 하지 않아도 손실함수 계산
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=100, batch_size=1000)  #학습

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)    #학습한 모델에 test 실행 및 평가(예측),   verbose=0(출력 x),1(자세히),2(함축적 정보 출력)
print('\nTest accuracy:', test_acc)   #테스트 정확도 출력=> 테스트 정확도가 학습 정확도 보다 낮은것을 보며 과적합(overfitting)된 것을 알수 있음. 해결책(drop 또는 더 많은 데이터 학습, 손실 증가시 조기 종료)



#예측하기
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

for i in range(15):   #반복문을 통한 10개 예측
  np.argmax(predictions[i])   #argmax: 여러개의 원소중 가장 큰 원소의 인덱스 반환
  print(np.argmax(predictions[i]))


for i in range(15):  # 15개의 test_label 출력하여 예측한 레이블 값과 비교
    test_labels[i]
    print("test_label",test_labels[i])



#예측을 모두 그래프로 표현 정의
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()