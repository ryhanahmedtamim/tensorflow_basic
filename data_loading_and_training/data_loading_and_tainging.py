import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

training_data = pd.read_csv("sales_data_training.csv", dtype=float)
testing_data = pd.read_csv("sales_data_test.csv", dtype=float)

X_training_data = training_data.drop(labels="total_earnings", axis=1)
Y_training_data = training_data[["total_earnings"]]


X_testing_data = testing_data.drop(labels="total_earnings", axis=1)
Y_testing_data = testing_data[["total_earnings"]]

X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

X_scaled_training_data = X_scaler.fit_transform(X_training_data)
Y_scaled_training_data = Y_scaler.fit_transform(Y_training_data)

X_scaled_testing_data = X_scaler.fit_transform(X_testing_data)
Y_scaled_testing_data = Y_scaler.fit_transform(Y_testing_data)

# Define model parameters
learning_rate = 0.001
training_epochs = 100
display_step = 5

# Define how many inputs and outputs are in our neural network
number_of_inputs = 9
number_of_outputs = 1

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

# layer 1

with tf.variable_scope("input"):
    X = tf.placeholder(dtype=tf.float32, shape=(None, number_of_inputs), name="input")

with tf.variable_scope("layer_1"):
    weights = tf.get_variable(name="weights_of_layer1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="bias_of_layer1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)


with tf.variable_scope("layer_2"):
    weights = tf.get_variable(name="weight_of_layer2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="bias_of_layer2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)


# Layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# Output Layer
with tf.variable_scope('output'):
    weights = tf.get_variable(name="weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.matmul(layer_3_output, weights) + biases
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))


# Section Three: Define the optimizer function that will be run to optimize the neural network

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# Initialize a session so that we can run TensorFlow operations
with tf.Session() as session:

    # Run the global variable initializer to initialize all variables and layers of the neural network
    session.run(tf.global_variables_initializer())

    # Run the optimizer over and over to train the network.
    # One epoch is one full run through the training data set.
    for epoch in range(training_epochs):

        # Feed in the training data and do one step of neural network training
        session.run(optimizer, feed_dict={X: X_scaled_training_data, Y: Y_scaled_training_data})

        # Print the current training status to the screen
        print("Training pass: {}".format(epoch))

    # Training is now complete!
    print("Training is complete!")



