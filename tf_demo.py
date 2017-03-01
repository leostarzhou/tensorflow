#!/usr/bin/python
#coding=utf-8
import tensorflow as tf
import numpy as np
import sys

# ʹ�� NumPy ���ɼ�����(phony data), �ܹ� 100 ����.
x_data = np.float32(np.random.rand(2, 100)) # �������
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# ����һ������ģ��
#
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# ��С������
loss = tf.reduce_mean(tf.square(y - y_data))
loss1 = tf.reduce_mean(tf.abs(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
train1 = optimizer.minimize(loss1)

 # ��ʼ������,�ɺ�����initialize_all_variables���Ѿ����������滻Ϊ�º���
init = tf.global_variables_initializer()

 # ����ͼ (graph)
sess = tf.Session()
sess.run(init)

 # ���ƽ��
for step in xrange(0, 201):
    sess.run(train)
    print sess.run(loss), sess.run(loss1)
    #if step % 20 == 0:
        #print step, sess.run(W), sess.run(b)

# �õ������Ͻ�� W: [[0.100  0.200]], b: [0.300]