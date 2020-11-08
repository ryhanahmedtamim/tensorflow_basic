import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Define Node

A = tf.placeholder(tf.int16, name="A")
B = tf.placeholder(tf.int16, name="B")

# Define graph
C = tf.subtract(x=A, y=B, name="subtract")

with tf.Session() as session:
    result = session.run(C, feed_dict={A: [1, 2, 7], B: [-4, 5, 1]})
    print(result)

