import tensorflow as tf
import numpy as np
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

from model import Model

# train baseline on vanilla sgd and for later time using mswc/op-mswc
def train_task(sess, model, num_iter, disp_freq, data, x, y_, lams, t):
    trainsets, trainY, testsets, testY = data
    model.restore(sess) # reassign optimal weights from previous training session
    if(lams[0] == 0):
        model.set_vanilla_loss()
    else:
        model.update_mswc_loss(lams[0], lams[1], lams[2], lams[3], t/7)
        
    # train on current task
    for iter in range(num_iter):
        train_range = np.array(range(len(trainY)))
        random.shuffle(train_range)
        batch_id = train_range[:64]
        model.train_step.run(feed_dict={x: trainsets[batch_id], y_: trainY[batch_id]})
        if iter % disp_freq == 0:
            MSE = sess.run(model.loss, feed_dict={x: testsets, y_: testY})
            print("[{} iter] total regression loss: {:.4f}".format(iter, MSE))

    return sess.run(model.y, feed_dict={x:testsets})

