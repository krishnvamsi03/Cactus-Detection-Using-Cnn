import numpy as np 
import pandas as pd 

import os
print(os.listdir())
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
import matplotlib.pyplot as plt
import cv2

train_data = pd.read_csv('train.csv')

train_data.head()

path = 'train'
train_images = []
for img in os.listdir(path):
    train_images.append(cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE))

train_images = np.array(train_images)

plt.figure(figsize = (2, 2))
plt.imshow(train_images[1], cmap = plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.show()

labels = train_data['has_cactus'].values
names = train_data['id'].values

plt.figure(figsize = (8, 8))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    plt.xlabel(labels[i])

def cnn_model_fn(features, labels, mode):
    
    input_layer = tf.reshape(features, [-1, 32, 32, 1])
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size = [5, 5],
        padding = 'same',
        activation = tf.nn.relu
    )
    
    pool1 = tf.layers.max_pooling2d(
        inputs = conv1,
        pool_size = [2, 2],
        strides = 2
    )
    
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64,
        kernel_size = [5, 5],
        padding = 'same',
        activation = tf.nn.relu
    )
    
    pool2 = tf.layers.max_pooling2d(
        inputs = conv2,
        pool_size = [2, 2],
        strides = 2
    )
    
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
    
    layer1 = tf.layers.dense(inputs = pool2_flat, units = 1024, activation = tf.nn.relu)
    layer2 = tf.layers.dense(inputs = layer1, units = 1000, activation = tf.nn.relu)
    dropout = tf.layers.dropout(inputs = layer2, rate = 0.4, training = mode == tf.estimator.ModeKeys.TRAIN)
    
    logits = tf.layers.dense(inputs = dropout, units = 1)
    prediction = {
        'class' : tf.round(tf.nn.sigmoid(logits)),
        'probabilities': tf.nn.sigmoid(logits)
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = prediction)
    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.cast(labels, dtype = tf.float32), 
                                                               logits = logits, name = 'loss'))
    if mode == tf.estimator.ModeKeys.TRAIN:
        
        optimizer = tf.train.AdamOptimizer(0.03)
        train_ops = optimizer.minimize(
            loss = cost,
            global_step = tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode = mode, loss = cost, train_op = train_ops)
    
    eval_metric_op = {
        'accuracy': tf.metrics.accuracy(labels = labels, predictions = prediction['class'])
    }
    return tf.estimator.EstimatorSpec(mode = mode, loss = cost, eval_metric_ops = eval_metric_op)

classifier = tf.estimator.Estimator(model_fn = cnn_model_fn)

tensorhook = {'loss_funtion': 'loss'}
logginhook = tf.train.LoggingTensorHook(
    tensors = tensorhook, every_n_iter = 100
)

len(train_images)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_images, labels, test_size = 0.33, random_state = 24)

X_train.shape

X_train = X_train / np.float32(255)
X_test = X_test / np.float32(255)

X_train.shape

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = X_train,
    y = np.matrix(y_train).T,
    shuffle = True,
    num_epochs = None,
    batch_size = 100
)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = X_test,
    y = np.matrix(y_test).T,
    num_epochs = 1,
    shuffle = False
)

classifier.train(
    input_fn = train_input_fn, 
    steps = 1500, 
    hooks = [logginhook]
)

eval_results = classifier.evaluate(
    input_fn = test_input_fn,
    steps = 1
)

print('Accuracy is {} and loss is {} '.format(eval_results['accuracy'], eval_results['loss']))
