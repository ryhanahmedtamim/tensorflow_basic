import os
import tensorflow as tf

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define Node
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name='Y')

# Define arithmetic_operations node

Z = tf.add(x=X, y=Y, name="Addition")

with tf.Session() as session:
    result = session.run(Z, feed_dict={X: [1, 2, 3], Y: [4, 5, 6]})
    print(result)

