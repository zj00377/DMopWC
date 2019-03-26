import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display

# variable initialization functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Model:
    def __init__(self, x, y_, lr):

        in_dim = int(x.get_shape()[1]) 
        out_dim = int(y_.get_shape()[1]) 

        self.x = x # input placeholder

        # simple 2-layer network
        W1 = weight_variable([in_dim,50])
        b1 = bias_variable([50])

        W2 = weight_variable([50,out_dim])
        b2 = bias_variable([out_dim])

        h1 = tf.nn.relu(tf.matmul(x,W1) + b1) # hidden layer
        #maxpool_h1 = tf.layers.max_pooling1d(h1, pool_size=1102)
        self.y = tf.matmul(h1,W2) + b2 # output layer
        #self.y = tf.matmul(maxpool_h1, W2) + b2
        self.lr = lr
        self.var_list = [W1, b1, W2, b2]
        
        # vanilla single-task loss
       #self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
        self.loss = tf.sqrt(tf.losses.mean_squared_error(self.y, y_))
        self.set_vanilla_loss()

        # performance metrics
        #correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        #self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def compute_fisher(self, imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10):
        # computer Fisher information for each parameter
        # initialize Fisher information for most recent task
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

        # sampling a random class from softmax
        probs = tf.nn.softmax(self.y)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(self.F_accum)
            mean_diffs = np.zeros(0)

        for i in range(num_samples):
            # select random input image
            im_ind = np.random.randint(imgset.shape[0])
            # compute first-order derivatives
            ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
            # square the derivatives and add to total
            for v in range(len(self.F_accum)):
                self.F_accum[v] += np.square(ders[v])
            if(plot_diffs):
                if i % disp_freq == 0 and i > 0:
                    # recording mean diffs of F
                    F_diff = 0
                    for v in range(len(self.F_accum)):
                        F_diff += np.sum(np.absolute(self.F_accum[v]/(i+1) - F_prev[v]))
                    mean_diff = np.mean(F_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)
                    for v in range(len(self.F_accum)):
                        F_prev[v] = self.F_accum[v]/(i+1)
                    plt.plot(range(disp_freq+1, i+2, disp_freq), mean_diffs)
                    plt.xlabel("Number of samples")
                    plt.ylabel("Mean absolute Fisher difference")
                    display.display(plt.gcf())
                    display.clear_output(wait=True)

        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples

    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())

    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))

    def l21_norm(self, W):
        # Computes the L21 norm of a symbolic matrix W
        return tf.reduce_sum(tf.norm(W, axis=1))

    def group_regularizer(self, v):
        # Computes a group regularization loss from a list of weight matrices corresponding
        const_coeff = lambda W: tf.sqrt(tf.cast(W.get_shape().as_list()[1], tf.float32))
     
        return tf.reduce_sum([tf.multiply(const_coeff(W), self.l21_norm(W)) for W in v if np.array((W.get_shape().as_list())).size == 2])

    def set_vanilla_loss(self):
        self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

    def update_ewc_loss(self, lam):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints

        if not hasattr(self, "ewc_loss"):
            self.ewc_loss = self.loss

        for v in range(len(self.var_list)):
            self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
        self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.ewc_loss)

    def update_mswc_loss(self, lam1, lam2, lam3, lam4, w):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints

        if not hasattr(self, "ewc_loss"):
            self.ewc_loss = self.loss

        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
        #weights = tf.trainable_variables() # all vars of your graph
        self.l1_loss = lam2 * tf.contrib.layers.apply_regularization(l1_regularizer, self.var_list)
        
        self.gl_loss = lam3 * self.group_regularizer(self.var_list)

        for v in range(len(self.var_list)):
            self.ewc_loss += (lam1/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))

        self.op_mswc_loss = self.ewc_loss + self.l1_loss + self.gl_loss
        print("Finish update the multi-stage weight consolidation loss function!")
        for v in range(len(self.var_list)):
            self.op_mswc_loss += lam4*w*tf.reduce_sum(tf.square(self.var_list[v] - self.star_vars[v]))
        
        print("Finish update the time order-preserving loss function!")
        self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.op_mswc_loss)





