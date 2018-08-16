import tensorflow as tf
sess=tf.InteractiveSession()
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x=tf.placeholder(tf.float32,shape=[None,784])
y_=tf.placeholder(tf.float32,shape=[None,10])

x_image=tf.reshape(x,[-1,28,28,1]) #[모든데이터 받으려고 -1, 28by28 , 흑백이라 1channel]

#첫번째 합성곱 레이어 filter 32개
#W_conv1=weight_variable([5,5,1,32])
#b_conv1=bias_variable([32]) 
W_conv1=tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
b_conv1=tf.Variable(tf.constant(0.1,shape=[32]))

conv2d01=tf.nn.conv2d(x_image,W_conv1,strides=[1,1,1,1],padding='SAME')
h_conv1=tf.nn.relu(conv2d01+b_conv1)
h_pool1=tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#=============================================


#두번째 합성곱 레이어
#W_conv1=weight_variable([5,5,32,64])
#b_conv1=bias_variable([64]) 
W_conv2=tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
b_conv2=tf.Variable(tf.constant(0.1,shape=[64]))

#h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
#h_pool2=max_pool_2X2(h_conv2)
conv2d02=tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')
h_conv2=tf.nn.relu(conv2d02+b_conv2)
h_pool2=tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#===================================================

##완전연결 계층
#W_fc1=weight_variable([7*7*64,1024])
#b_fc1-bias_variable([1024])
W_fc1=tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
b_fc1=tf.Variable(tf.constant(0.1,shape=[1024]))

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#=========================================================

#드롭아웃
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

##최종 소프트맥스
W_fc2=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b_fc2=tf.nn.softmax(tf.constant(0.1,shape=[10]))
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)


#최종
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch=mnist.train.next_batch(50)
    if i % 100 ==0:
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print('step %d, training accurary %g'%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
print('test accuracy %g'%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))


# # -*- coding: utf-8 -*-
# """
# Created on Wed Dec 13 15:14:08 2017

# @author: student



# """


# C_END     = "\033[0m"
# C_BOLD    = "\033[1m"
# C_INVERSE = "\033[7m"
 
# C_BLACK  = "\033[30m"
# C_RED    = "\033[31m"
# C_GREEN  = "\033[32m"
# C_YELLOW = "\033[33m"
# C_BLUE   = "\033[34m"
# C_PURPLE = "\033[35m"
# C_CYAN   = "\033[36m"
# C_WHITE  = "\033[37m"
 
# C_BGBLACK  = "\033[40m"
# C_BGRED    = "\033[41m"
# C_BGGREEN  = "\033[42m"
# C_BGYELLOW = "\033[43m"
# C_BGBLUE   = "\033[44m"
# C_BGPURPLE = "\033[45m"
# C_BGCYAN   = "\033[46m"
# C_BGWHITE  = "\033[47m"




# def printComment(str):
#   print(C_BOLD + C_GREEN)
#   print(str)
#   print(C_END)
#   # print(C_BOLD +  C_GREEN + str + C_END)
# def printError(str):
#   print(C_BOLD + C_RED)
#   print(str)
#   print(C_END)
#   # print(C_BOLD +  C_RED + str + C_END)






# import tensorflow as tf
# import math

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# ########################
# ## input
# #######################

# # x = tf.placeholder(tf.float32, [None, 784])
# # y_ = tf.placeholder(tf.float32, shape = [10])

# # x_image = tf.reshape(x,[-1, 28,28,1])

# # W_conv1 = tf.Variable(tf.truncated_normal([5,5,32,64],
# #                       stddev = 0.1))
# # b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))

# # conv2d01 = tf.nn.conv2d(x_image, W_conv1, strides = [1,1,1,1], padding = 'SAME')

# # h_conv1 = tf.nn.relu(conv2d01 + b_conv1)
# # h_pool1 = tf.nn.max_pool(h_conv1,ksize = [1,2,2,1],strides= [1,2,2,1],padding = 'SAME')


# # W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev = 0.1))
# # b_conv2 = tf.variable(tf.constant(0.1, shape=[64]))

# # h_conv2 = tf.nn.relu(conv2d01 + b_conv1)
# # h_pool2 = tf.nn.max_pool(h_conv1,
# #                          ksize = [1,2,2,1],
# #                          strides= [1,2,2,1],
# #                          padding = 'SAME')


# x=tf.placeholder(tf.float32,shape=[None,784])
# y_=tf.placeholder(tf.float32,shape=[None,10])

# x_image=tf.reshape(x,[-1,28,28,1]) #[모든데이터 받으려고 -1, 28by28 , 흑백이라 1channel]

# #첫번째 합성곱 레이어 filter 32개
# #W_conv1=weight_variable([5,5,1,32])
# #b_conv1=bias_variable([32]) 
# W_conv1=tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
# b_conv1=tf.Variable(tf.constant(0.1,shape=[32]))

# conv2d01=tf.nn.conv2d(x_image,W_conv1,strides=[1,1,1,1],padding='SAME')
# h_conv1=tf.nn.relu(conv2d01+b_conv1)
# h_pool1=tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# #=============================================


# #두번째 합성곱 레이어
# #W_conv1=weight_variable([5,5,32,64])
# #b_conv1=bias_variable([64]) 
# W_conv2=tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
# b_conv2=tf.Variable(tf.constant(0.1,shape=[64]))

# #h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
# #h_pool2=max_pool_2X2(h_conv2)
# conv2d02=tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')
# h_conv2=tf.nn.relu(conv2d02+b_conv2)
# h_pool2=tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')



# h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
# W_fc1 = tf.Variable(
#   tf.truncated_normal([7*7*64, 1024],
#     stddev = 0.1)
# )
# b_fc1 = tf.Variable(tf.constant(0.1, shape = [1024]))
# h_fc1 = tf.nn.relu(
#   tf.matmul(h_pool2_flat, W_fc1) + b_fc1
# )

# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# W_fc2 = tf.Variable(tf.truncated_normal([1024,10],
#   stddev = 0.1))
# b_fc2 = tf.Variable(tf.constant(0.1, shape= [10]))

# y_conv = tf.nn.softmax(
#   tf.matmul(h_fc1_drop, W_fc2)
# )

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv)),reduction_indices=[1])

# train_step = tf.train.AdamOptimizer(1e-04).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv),tf.argmax(y_,1))

# accuracy = tf.reduce_mean(tf.cast(correct_prediction,
#   tf.float32))

# sess.run(tf.global_variables_initializer)
# for i in range(20000):
#   batch = mnist.train.next_batch(50)
#   if i % 100:
#     train_accuray = accuracy.eval(feed_dict = {
#       x : batch[0], y_:batch[1], keep_prob: 1.0
#       })
#   train_step.run(feed_dict={x:batch[0], y_:batch[1]},
#     keep_prob = 0.5)


# printComment("test 정확도 : %g" % accuracy.eval(feed_dict={
#   x : mnist.test.images , y_: mnist.test.labelsm,
#   keep_prob : 1.0
#   }))


# sys.exit()
















# ########################
# ## hidden 1
# #######################

# W1 = tf.Variable(tf.truncated_normal(
#         [784, 100], 
#         stddev=1.0/math.sqrt(float(784))))
# # W1 = tf.Variable(tf.zeros([784, 100]))
# b1 = tf.Variable(tf.zeros([100]))
# y1 = tf.nn.tanh(tf.matmul(x, W1) + b1)

# ####################
# ## hidden 2
# ####################

# W2 = tf.Variable(tf.truncated_normal(
#         [100, 20], 
#         stddev=1.0/math.sqrt(float(100))))
# # W2 = tf.Variable(tf.zeros([100, 20]))
# b2 = tf.Variable(tf.zeros([20]))
# y2 = tf.nn.tanh(tf.matmul(y1, W2) + b2)

# ##################################
# ## output layer
# ##################################
# W = tf.Variable(tf.zeros([20, 10]))

# printError("W : ")
# printComment(W)
# b = tf.Variable(tf.zeros([10]))

# printError("b : ")
# printComment(b)
# y = tf.nn.softmax(tf.matmul(y2, W) + b)

# printError("y : ")
# printComment(y)
# y_ = tf.placeholder(tf.float32, [None, 10])

# printError("y_ : ")
# printComment(y_)



# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# printError("cross_entropy : ")
# printComment(cross_entropy)


# # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
# train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
# init = tf.global_variables_initializer()

# sess = tf.Session()
# sess.run(init)

# for i in range(20000):
#   batch_xs, batch_ys = mnist.train.next_batch(100)
#   sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print(sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels}))
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# print(sess.run(tf.argmax(y,1), feed_dict={x: mnist.test.images, y_: mnist.test.labels}))




# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x=tf.placeholder(tf.float32,shape=[None,784])
# y_=tf.placeholder(tf.float32,shape=[None,10])

# x_image=tf.reshape(x,[-1,28,28,1]) #[모든데이터 받으려고 -1, 28by28 , 흑백이라 1channel]

# #첫번째 합성곱 레이어 filter 32개
# #W_conv1=weight_variable([5,5,1,32])
# #b_conv1=bias_variable([32]) 
# W_conv1=tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
# b_conv1=tf.Variable(tf.constant(0.1,shape=[32]))

# conv2d01=tf.nn.conv2d(x_image,W_conv1,strides=[1,1,1,1],padding='SAME')
# h_conv1=tf.nn.relu(conv2d01+b_conv1)
# h_pool1=tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# #=============================================


# #두번째 합성곱 레이어
# #W_conv1=weight_variable([5,5,32,64])
# #b_conv1=bias_variable([64]) 
# W_conv2=tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
# b_conv2=tf.Variable(tf.constant(0.1,shape=[64]))

# #h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
# #h_pool2=max_pool_2X2(h_conv2)
# conv2d02=tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')
# h_conv2=tf.nn.relu(conv2d02+b_conv2)
# h_pool2=tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# #===================================================

# ##완전연결 계층
# #W_fc1=weight_variable([7*7*64,1024])
# #b_fc1-bias_variable([1024])
# W_fc1=tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
# b_fc1=tf.Variable(tf.constant(0.1,shape=[1024]))

# h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
# h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
# #=========================================================

# #드롭아웃
# keep_prob=tf.placeholder(tf.float32)
# h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

# ##최종 소프트맥스
# W_fc2=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
# # h_pool2_flath_pool2_flatb_fc2=tf.nn.softmax(tf.)


# cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
# train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
# accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# sess.run(tf.global_variables_initializer())

# for i in range(20000):
#     batch=mnist.train.next_batch(50)
#     if i % 100 ==0:
#         train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
#         print('step %d, training accurary %g'%(i,train_accuracy))
#     train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
# print('test accuracy %g'%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
























