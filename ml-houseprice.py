import pymysql
import pandas as pd
from pandas import DataFrame
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns


db = pymysql.connect('docker.for.mac.localhost', 'root', '', 'landbook_development', charset='utf8')

cursor = db.cursor()

cursor.execute('select price, pnu, year_approval, floor, area_exclusive from villa_sales where deal_type="주택" limit 100000;')

data = cursor.fetchall()
db.close()

dataFrame = {
  'price': [],
  'pnu': [],
  'year_approval': [],
  'floor': [],
  'area_exclusive': []
}
for row in data:
  dataFrame['price'].append(row[0])
  dataFrame['pnu'].append(int(row[1][:11]))
  dataFrame['year_approval'].append(int(row[2].year if row[2] != None else 2010))
  dataFrame['floor'].append(int(''.join(x for x in row[3] if x.isdigit()) if ''.join(x for x in row[3] if x.isdigit()) else 4))
  dataFrame['area_exclusive'].append(row[4])

cols = ['pnu', 'floor', 'year_approval', 'area_exclusive', 'price']

df = DataFrame(dataFrame, columns=cols)
df.to_csv('villasale.txt', index=False)

x_data = []
y_data = []

for v in df.values:
  x_data.append(v[3:4])

for v in df.values:
  y_data.append(v[-1:])

# sns.pairplot(DataFrame(x_data, columns=['year_approval', 'area_exclusive', 'price']), size=5)
# plt.savefig("bb.png")

X = tf.placeholder(tf.float32, shape=[None, 1])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.get_variable("W", shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(101):
  cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
    feed_dict={X: x_data, Y: y_data})
  
  if step % 100 == 0:
    print(step, "Cost: ", cost_val)

print("빌라가격은: ", sess.run(hypothesis, feed_dict={X: [[18.65]]})[0][0])
print(df.values[0][3])
