from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
import pandas as pd, numpy as np

csv = pd.read_csv("bmi.csv")
csv["weight"] /= 100
csv["height"] /= 200

bmi_class = {
  "thin": [1, 0, 0],
  "normal": [0, 1, 0],
  "fat": [0, 0, 1]
}

y = np.empty((20000, 3))

for i, v in enumerate(csv["label"]):
  y[i] = bmi_class[v]

x = csv[["weight", "height"]].as_matrix()

x_train, y_train = x[1: 15001], y[1:15001]
x_test, y_test = x[15001:20001], y[15001:20001]

model = Sequential()
model.add(Dense(512, input_shape=(2,)))
model.add(Activation("relu"))
model.add(Dropout(0.1))
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.1))
model.add(Dense(3))
model.add(Activation("softmax"))
model.compile("rmsprop", "categorical_crossentropy", metrics=["accuracy"])

model.fit(
  x_train,
  y_train,
  batch_size=100,
  nb_epoch=20,
  validation_split=0.1,
  callbacks=[EarlyStopping(monitor="val_loss", patience=2)],
  verbose=1
)

# model.predict()

score = model.evaluate(x_test, y_test)
print("score loss:", score[0])
print("score accuracy:", score[1])
print(x_train, y_train)

# import random
# # BMI를 계산해서 레이블을 리턴하는 함수
# def calc_bmi(h, w):
#     bmi = w / (h/100) ** 2
#     if bmi < 18.5: return "thin"
#     if bmi < 25: return "normal"
#     return "fat"
# # 출력 파일 준비하기
# fp = open("bmi.csv","w",encoding="utf-8")
# fp.write("height,weight,label\r\n")
# # 무작위로 데이터 생성하기
# cnt = {"thin":0, "normal":0, "fat":0}
# for i in range(20000):
#     h = random.randint(120,200)
#     w = random.randint(35, 80)
#     label = calc_bmi(h, w)
#     cnt[label] += 1
#     fp.write("{0},{1},{2}\r\n".format(h, w, label))
# fp.close()
# print("ok,", cnt)