import tensorflow as tf
import time
"""
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")

f = x*x*y + y + 2

"""

"""
# 1st method
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
"""

"""
# 2st method
with tf.Session() as sess:
    x.initializer.run()  # equals to tf.get_default_session().run(x.initializer)
    y.initializer.run()  # equals to tf.get_default_session().run(y.initializer)
    result = f.eval()  # equals to tf.get_default_session().run(f)
"""

"""
# by using this init, we can initialize each variables at once
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()
"""

"""
# Any node you create is automatically added to the default graph
x1 = tf.Variable(1)
print(x1.graph is tf.get_default_graph())

# if multiple independent graphs are needed, you can create new Graph
graph = tf.Graph()
with graph.as_default():  # temporal default graph "graph"
    x2 = tf.Variable(2)

print(x2.graph is graph)
print(x2.graph is tf.get_default_graph())
print(graph)
print(tf.get_default_graph())
"""

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

# when you evaluate a node, TensorFlow automatically determines set of nodes
# that it depends on and it evaluates these nodes first
# nodes are evaluated recursively
# TensorFlow does not reuse previous evaluation(each w and x are evaluated twice)
with tf.Session() as sess:
    start_time = time.time()
    print(y.eval())
    print(z.eval())
    print("Evaluating Separately: %s" %(time.time() - start_time))  # 0.0069811344146728516


# To evaluate y and z efficiently(not evaluating w and x twice),
# you must ask TensorFlow to evaluate both in just one graph run
with tf.Session() as sess:
    start_time = time.time()
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)
    print("Evaluating At Once: %s" % (time.time() - start_time))  # 0.003989458084106445

import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape  # m*n matrix

housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
Xt = tf.transpose(X)

theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(Xt, X)), Xt), y)

with tf.Session() as sess:
    theta_value = theta.eval()
    # print(theta_value)