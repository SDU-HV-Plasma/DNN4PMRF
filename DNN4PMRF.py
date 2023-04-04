import time
start = time.time()
import tensorflow.compat.v1 as tf
import tensorflow as tf1
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
tf.disable_v2_behavior()
import numpy as np
import csv

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#Training set

df = pd.read_csv('Training set.CSV',encoding='gbk')

#Normalization

xa_scaler=MinMaxScaler(feature_range=(0,1))
xb_scaler=MinMaxScaler(feature_range=(0,1))

ya_scaler=MinMaxScaler(feature_range=(0,1))
yb_scaler=MinMaxScaler(feature_range=(0,1))

#First cycle
a=df[df.Cycle>=0]
dfa=a[a.Cycle<1.0]
xa=dfa[['Cycle','Duty cycle']]
xaa=xa_scaler.fit_transform(xa)
xa_data=np.array(xaa,dtype='float32')
ya=dfa[['Current density']]
yaa=ya_scaler.fit_transform(ya)
ya_data=np.array(yaa,dtype='float32')

#Second cycle
b=df[df.Cycle>=1.0]
dfb=b[b.Cycle<=2.0]
xb=dfb[['Cycle','Duty cycle']]
xbb=xb_scaler.fit_transform(xb)
xb_data=np.array(xbb,dtype='float32')
yb=dfb[['Current density']]
ybb=yb_scaler.fit_transform(yb)
yb_data=np.array(ybb,dtype='float32')

#Testing set

dft = pd.read_csv('Testing set 50Per.CSV',encoding='gbk')

#First cycle
at=dft[dft.Cycle>=0]
dfat=at[at.Cycle<1.0]
xat=dfat[['Cycle','Duty cycle']]
xaat=xa_scaler.transform(xat)
xat_data=np.array(xaat,dtype='float32')
yat=dfat[['Current density']]
yaat=ya_scaler.transform(yat)
yat_data=np.array(yaat,dtype='float32')

#Second cycle
bt=dft[dft.Cycle>=1.0]
dfbt=bt[bt.Cycle<=2.0]
xbt=dfbt[['Cycle','Duty cycle']]
xbbt=xb_scaler.transform(xbt)
xbt_data=np.array(xbbt,dtype='float32')
ybt=dfbt[['Current density']]
ybbt=yb_scaler.transform(ybt)
ybt_data=np.array(ybbt,dtype='float32')

# Input

xs1 = tf.placeholder(tf.float32, [None,2])
ys1 = tf.placeholder(tf.float32, [None,1])
xs2 = tf.placeholder(tf.float32, [None,2])
ys2 = tf.placeholder(tf.float32, [None,1])

# Hidden layer

l1 = add_layer(xs1, 2, 30, activation_function=tf1.nn.relu)
l2 = add_layer(l1, 30, 30, activation_function=tf1.tanh)
l3 = add_layer(l2, 30, 30, activation_function=tf1.tanh)
l4 = add_layer(l3, 30, 30, activation_function=tf1.sigmoid)

l5 = add_layer(xs2, 2, 30, activation_function=tf1.nn.relu)
l6 = add_layer(l5, 30, 30, activation_function=tf1.tanh)
l7 = add_layer(l6, 30, 30, activation_function=tf1.tanh)
l8 = add_layer(l7, 30, 30, activation_function=tf1.sigmoid)

# Output

prediction1 = add_layer(l4, 30, 1, activation_function=None)
prediction2 = add_layer(l8, 30, 1, activation_function=None)

# Training loss

loss1 = tf.reduce_mean(tf.square(ys1 - prediction1))
loss2 = tf.reduce_mean(tf.square(ys2 - prediction2))

train_step1 = tf.train.AdamOptimizer(0.0001).minimize(loss1)
train_step2 = tf.train.AdamOptimizer(0.0001).minimize(loss2)

# Computation loss

lossc1 = 100*tf.reduce_mean(tf.abs((ys1 - prediction1)/ys1))
lossc2 = 100*tf.reduce_mean(tf.abs((ys2 - prediction2)/ys2))

# Training

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    is_train = True
    is_mod = True
    saver = tf.train.Saver(max_to_keep=1)

    if is_train:
      if is_mod:
        model_file1 = tf1.train.latest_checkpoint('save1/')
        saver.restore(sess, model_file1)
        for i in range(1000001):
         sess.run(train_step1, feed_dict={xs1: xa_data, ys1: ya_data})
         if i % 100 == 0:
           print(sess.run(loss1, feed_dict={xs1: xa_data, ys1: ya_data}), (i/1000001)*100, '%')
        saver.save(sess, 'save1/model1', global_step=i + 1)
      else:
        model_file2 = tf1.train.latest_checkpoint('save2/')
        saver.restore(sess, model_file2)
        for i in range(1000001):
          sess.run(train_step2, feed_dict={xs2: xb_data, ys2: yb_data})
          if i % 100 == 0:
            print(sess.run(loss2, feed_dict={xs2: xb_data, ys2: yb_data}), (i/1000001)*100, '%')
        saver.save(sess, 'save2/model2', global_step=i + 1)

# Computation
    else:
        with open("Predicted current density 50Per.csv","w",newline='') as f:
         b_csv = csv.writer(f)
         model_file1 = tf1.train.latest_checkpoint('save1/')
         saver.restore(sess, model_file1)
         print('The relative error of Current density1:')
         print(sess.run(lossc1, feed_dict={xs1: xat_data, ys1: yat_data}))
         yap = ya_scaler.inverse_transform(sess.run(prediction1, feed_dict={xs1: xat_data}))
         b_csv.writerows(yap)

         model_file2 = tf1.train.latest_checkpoint('save2/')
         saver.restore(sess, model_file2)
         print('The relative error of Current density2:')
         print(sess.run(lossc2, feed_dict={xs2: xbt_data, ys2: ybt_data}))
         ybp = yb_scaler.inverse_transform(sess.run(prediction2, feed_dict={xs2: xbt_data}))
         b_csv.writerows(ybp)

time_used = (time.time() - start)/3600
print('Total running time(hour):')
print(time_used)