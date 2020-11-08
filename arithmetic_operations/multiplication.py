import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# define node

a = tf.placeholder(tf.int16, name="a")
b = tf.placeholder(tf.int16, name="b")

# define graph
c = tf.multiply(a, b, name="multiplication")

with tf.Session() as session:
    result = session.run(c, feed_dict={a: [4, 5], b: [9, 10]})
    print(result)


