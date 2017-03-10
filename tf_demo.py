#!/usr/bin/python
#coding=utf-8
import tensorflow as tf
import numpy as np
import sys

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型
#
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
loss1 = tf.reduce_mean(tf.abs(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
train1 = optimizer.minimize(loss1)

 # 初始化变量,旧函数（initialize_all_variables）已经被废弃，替换为新函数
init = tf.global_variables_initializer()

 # 启动图 (graph)
sess = tf.Session()
sess.run(init)

# tensorboard
#merged_summary_op = tf.merge_all_summaries()
#summary_writer = tf.train.SummaryWriter('~/log', sess.graph)
#total_step = 0

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('/home/leostarzhou/tensorflow/log', sess.graph)
total_step = 0

 # 拟合平面
for step in xrange(0, 201):
    summary = sess.run(train)
    #print sess.run(loss), sess.run(loss1)
    writer.add_summary(summary, step)
    #total_step += 1
    #summary_str = sess.run(merged_summary_op)
    #summary_writer.add_summary(summary_str, total_step)
    #if step % 20 == 0:
        #print step, sess.run(W), sess.run(b)

writer.close()
# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]
