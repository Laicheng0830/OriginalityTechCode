import tensorflow as tf
import torch

import numpy as np
import time

import tensorlayer as tl
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

class Dense(object):
    def __init__(self, w_init):
        self.w = w_init

    def __call__(self, x):
        return tf.matmul(x, self.w)

class ModelBuild(object):
    def __init__(self):
        self.weights = []
        self.w1 = tf.Variable(self.w_init(shape=(784, 200)), name='dense1/w')
        self.w2 = tf.Variable(self.w_init(shape=(200, 50)), name='dense2/w')
        self.w3 = tf.Variable(self.w_init(shape=(50, 10)), name='dense3/w')
        self.dense1 = Dense(self.w1)
        self.dense2 = Dense(self.w2)
        self.dense3 = Dense(self.w3)

    def get_model(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    @property
    def get_train_weights(self):
        return self.weights

    def w_init(self, shape):
        return tf.random.truncated_normal(shape=shape)

    def __setattr__(self, key, value):
        if isinstance(value, tf.Variable):
            self.weights.append(value)
        object.__setattr__(self, key, value)


model = ModelBuild()
inputs = tf.constant(value=1.0, shape=(50, 784))
print(model.get_model(inputs).shape)
# print(model.get_train_weights)
for w in model.get_train_weights:
    print(w.name, w.shape)

n_epoch = 50
batch_size = 500
print_freq = 5

optimizer = tf.optimizers.Adam(learning_rate=0.0001)

for epoch in range(n_epoch):  ## iterate the dataset n_epoch times
    start_time = time.time()
    ## iterate over the entire training set once (shuffle the data via training)
    for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):

        with tf.GradientTape() as tape:
            ## compute outputs
            _logits = model.get_model(X_batch)
            ## compute loss and update model
            _loss = tl.cost.softmax_cross_entropy_with_logits(_logits, y_batch, name='train_loss')
        grad = tape.gradient(_loss, model.get_train_weights)
        optimizer.apply_gradients(zip(grad, model.get_train_weights))

    ## use training and evaluation sets to evaluate the model every print_freq epoch
    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=False):
            _logits = model.get_model(X_batch)
            train_loss += tl.cost.softmax_cross_entropy_with_logits(_logits, y_batch, name='eval_loss')
            train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))

        val_loss, val_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
            _logits = model.get_model(X_batch)  # is_train=False, disable dropout
            val_loss += tl.cost.softmax_cross_entropy_with_logits(_logits, y_batch, name='eval_loss')
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   val loss: {}".format(val_loss / n_iter))
        print("   val acc:  {}".format(val_acc / n_iter))