import tensorflow as tf
import numpy as np

from model import Model
from mswc_regression import train_task
from read_adni import load_ADNI

flags = tf.app.flags
flags.DEFINE_integer("max_iter", 1200, "Epoch to train")
flags.DEFINE_integer("test_iter", 200, "Epoch to display")
flags.DEFINE_float("lr", 0.001, "Learing rate(init) for train")
flags.DEFINE_float("train_ratio", 0.9, "ratio for split training and testing")
flags.DEFINE_string("label", "MMSE", "predict MMSE or ADAS")
flags.DEFINE_string("data", "sparse", "use data mTBM or sparse")
flags.DEFINE_integer("batch_size", 256, "The size of batch for 1 iteration")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory path to save the checkpoints")
flags.DEFINE_float("lambda1", 0, "lambda for ewc")
flags.DEFINE_float("lambda2", 0, "lambda for sparse term of mswc")
flags.DEFINE_float("lambda3", 0, "lambda for group norm of mswc")
flags.DEFINE_float("lambda4", 0, "lambda for time order preserving term")
flags.DEFINE_string("mode", "lifelong", "learn AD in batch or sequence")
flags.DEFINE_integer("num_split", 2, "The number of splitting test data")
FLAGS = flags.FLAGS

#given each time point name
timePoint=["base", "M6", "M12", "M18", "M24", "M36", "M48"]

#for batch mode, we use vanilla loss, set all hyperparameter for MSWC to zero.
if FLAGS.mode == "batch":
    FLAGS.lambda1=0
    FLAGS.lambda2=0
    FLAGS.lambda3=0
    FLAGS.lambda4=0

result = np.zeros((5, FLAGS.num_split))
wR = np.zeros((1, FLAGS.num_split))
for j in range(FLAGS.num_split):
    wR[0, j] = 0
    trainXs, trainYs, testXs, testYs = load_ADNI(FLAGS.label, FLAGS.data, FLAGS.train_ratio, FLAGS.mode)
    x = tf.placeholder(tf.float32, shape=[None, trainXs[0].shape[1]])
    y_ = tf.placeholder(tf.float32, shape=[None, 7])
    model = Model(x, y_, FLAGS.lr) # simple 2-layer network
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(4):
        data = trainXs[i], trainYs[i], testXs[i], testYs[i]
        if i == 0:
            pred = train_task(sess, model, FLAGS.max_iter, FLAGS.test_iter, data, x, y_, [0], i+1)
            rMSE = tf.sqrt(tf.reduce_sum(tf.pow(pred[:, i+2] - testYs[i][:, i+2], 2))/(testXs[i].shape[0]))
            result[i][j] = sess.run(rMSE)
            print("rMSE is {:.4f} for predicting time point {}".format(result[i][j], timePoint[i+2]))
            model.compute_fisher(trainXs[i], sess, num_samples=200, plot_diffs=False)
            model.star()
        else: 
            pred = train_task(sess, model, FLAGS.max_iter, FLAGS.test_iter, data, x, y_, [FLAGS.lambda1, FLAGS.lambda2, FLAGS.lambda3, FLAGS.lambda4], i+1)
            rMSE = tf.sqrt(tf.reduce_sum(tf.pow(pred[:, i+2] - testYs[i][:, i+2], 2))/ (testXs[i].shape[0]))
            result[i][j] = sess.run(rMSE)
            print("rMSE is {:.4f} for predicting time point {}".format(result[i][j], timePoint[i+2]))
            model.compute_fisher(trainXs[i], sess, num_samples=100, plot_diffs=False)
            model.star()

    rMSE = tf.sqrt(tf.reduce_sum(tf.pow(pred[:, i+3] - testYs[i][:, i+3], 2))/ (testXs[i].shape[0])) 
    result[i+1][j] = sess.run(rMSE)
    print("rMSE is {:.4f} for predicting time point {}".format(result[i+1][j], timePoint[i+3]))

    print("Finish trained {}-th time random split test set of DMopWC.".format(j+1))

for i in range(5):
    print("Average rMSE is {:.2f}+/-{:.2f} for predicting time point {}".format(np.average(result[i]), np.std(result[i]), timePoint[i+2]))


