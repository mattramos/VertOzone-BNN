import tensorflow as tf
from sklearn.utils import shuffle
import pickle as pkl
import datetime
import numpy as np
import os
from utils import initialize_uninitialized, batch_retrieve


class BNN(object):

    def __init__(self, sess, checkpoint_dir, log_dir, 
                 x_dim, y_dim, num_models, n_data,
                 hidden_size, learning_rate, lambda_anchor,
                 init_std_1w, init_std_1b, init_std_2w, 
                 init_std_2b, init_std_biasw, init_std_noisew,
                 load_from_epoch=False):

        self.sess = sess
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = tf.summary.FileWriter(self.log_dir)
        self.model_name = 'BNN'

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_data = n_data
        self.lambda_anchor = lambda_anchor

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_mdls = num_models

        self.inputs = tf.placeholder(tf.float32, [None, x_dim], name='inputs')
        self.modelpred = self.inputs[:, :self.n_mdls]
        self.spacetime = self.inputs[:, self.n_mdls:]
        self.y_target = tf.placeholder(tf.float32, [None, y_dim], name='target')

        self.layer1 = tf.layers.Dense(hidden_size,
                                      activation=tf.nn.tanh,
                                      name='layer1',
                                      kernel_initializer=tf.random_normal_initializer(mean=0.,
                                                                                      stddev=init_std_1w),
                                      bias_initializer=tf.random_normal_initializer(mean=0.,
                                                                                    stddev=init_std_1b))
        self.layer1_out = self.layer1(self.spacetime)
        self.layer2 = tf.layers.Dense(num_models,
                                      activation=None,
                                      name='layer2',
                                      kernel_initializer=tf.random_normal_initializer(mean=0.,
                                                                                      stddev=init_std_2w),
                                      bias_initializer=tf.random_normal_initializer(mean=0.,
                                                                                    stddev=init_std_2b))
        self.layer2_out = self.layer2(self.layer1_out)
        self.model_coeffs = tf.nn.softmax(self.layer2_out)
        self.modelbias_layer = tf.layers.Dense(y_dim,
                                               activation=None,
                                               name='layer-bias',
                                               use_bias=False,
                                               kernel_initializer=tf.random_normal_initializer(mean=0.,
                                                                                               stddev=init_std_biasw))
        self.modelbias = self.modelbias_layer(self.layer1_out)

        self.output = tf.reduce_sum(self.model_coeffs * self.modelpred, axis=1) + tf.reshape(self.modelbias, [-1])

        self.noise_layer = tf.layers.Dense(self.y_dim,
                                           activation=tf.nn.sigmoid, 
                                           name='layer-noise',
                                           use_bias=False, 
                                           kernel_initializer=tf.random_normal_initializer(mean=0.,
                                                                                           stddev=init_std_noisew))
        self.noise_pred = 0.06 * self.noise_layer(self.layer1_out)

        self.opt_method = tf.train.AdamOptimizer(self.learning_rate)

        self.noise_sq = tf.square(self.noise_pred)[:,0] + 1e-6
        self.err_sq = tf.reshape(tf.square(self.y_target[:,0] - self.output), [-1])
        num_data_inv = tf.cast(tf.divide(1, tf.shape(self.inputs)[0]), dtype=tf.float32)

        self.mse_ = num_data_inv * tf.reduce_sum(self.err_sq) 
        self.loss_ = num_data_inv * (tf.reduce_sum(tf.divide(self.err_sq, self.noise_sq)) + tf.reduce_sum(tf.log(self.noise_sq)))
        self.optimizer = self.opt_method.minimize(self.loss_)

        # Summary stats
        self.mse_sum = tf.summary.scalar("mse", self.mse_)
        self.loss_sum = tf.summary.scalar("loss", self.loss_)

        self.build_model()

        self.saver = tf.train.Saver(max_to_keep=100)


        if self.load_model(load_from_epoch):
            print('Loading from pre-existing model')
        else:
            print('Initialising new model')
        
        self.writer.add_graph(self.sess.graph)


        return


    def build_model(self):
        """ The model is regularised around its initial parameters"""

        # Initialise parameters
        initialize_uninitialized(self.sess)

        # Get initial vars
        ops = [self.layer1.kernel,
               self.layer1.bias,
               self.layer2.kernel,
               self.layer2.bias,
               self.modelbias_layer.kernel,
               self.noise_layer.kernel]
        w1, b1, w2, b2, wbias, wnoise = self.sess.run(ops)

        # Anchor the model
        self.w1_init, self.b1_init, self.w2_init, self.b2_init, self.wbias_init, self.wnoise_init = w1, b1, w2, b2, wbias, wnoise
        loss_anchor = self.lambda_anchor[0]*tf.reduce_sum(tf.square(self.w1_init - self.layer1.kernel))
        loss_anchor += self.lambda_anchor[1]*tf.reduce_sum(tf.square(self.b1_init - self.layer1.bias))
        loss_anchor += self.lambda_anchor[2]*tf.reduce_sum(tf.square(self.w2_init - self.layer2.kernel))
        loss_anchor += self.lambda_anchor[3]*tf.reduce_sum(tf.square(self.b2_init - self.layer2.bias))
        loss_anchor += self.lambda_anchor[4]*tf.reduce_sum(tf.square(self.wbias_init - self.modelbias_layer.kernel))
        loss_anchor += self.lambda_anchor[5]*tf.reduce_sum(tf.square(self.wnoise_init - self.noise_layer.kernel)) # new param

        self.loss_anchor = tf.cast(1.0/self.n_data, dtype=tf.float32) * loss_anchor
        self.loss_anc_sum = tf.summary.scalar("loss_anchor", self.loss_anchor)
        
        # combine with original loss
        self.loss_ = self.loss_ + tf.cast(1.0/self.n_data, dtype=tf.float32) * self.loss_anchor
        self.optimizer = self.opt_method.minimize(self.loss_)
        
        return

    def predict(self, X):
        return batch_retrieve(self, self.output, X)


    def save_model(self, epoch):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, self.model_name),
                        global_step=epoch)
        return

    
    def load_model(self, load_from_epoch):
        print("Reading checkpoints...")
        if load_from_epoch:
            ckpt_name = self.model_name + '-{}'.format(load_from_epoch)
            print('Loading from checkpoint: {}'.format(ckpt_name))
            print(os.path.join(self.checkpoint_dir, ckpt_name))
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            return True
        else:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print('Loading from checkpoint: {}'.format(ckpt))
                
                return True
            else:
                return False


    def train(self, n_epochs, ep_0, X_train, y_train, shuff=50, batch_size=1000, print_freq=100, save_freq=250):
        
        self.summary = tf.summary.merge([self.mse_sum, self.loss_sum, self.loss_anc_sum])
        ep_ = ep_0
        while ep_ < n_epochs + ep_0:
            ep_ += 1
            # Train in batches
            j_max = int(X_train.shape[0]/batch_size)
            for j in range(int(X_train.shape[0]/batch_size)):
                feed_b = {}
                feed_b[self.inputs] = X_train[j*batch_size:(j+1)*batch_size, :]
                feed_b[self.y_target] = y_train[j*batch_size:(j+1)*batch_size, :]
                blank = self.sess.run(self.optimizer, feed_dict=feed_b)
            if (ep_ % print_freq) == 0: 
                feed_b = {}
                feed_b[self.inputs] = X_train
                feed_b[self.y_target] = y_train
                summary_str = self.sess.run(self.summary, feed_dict=feed_b)
                self.writer.add_summary(summary_str, ep_)
                
#                 loss_mse = self.sess.run(self.mse_, feed_dict=feed_b)
#                 loss_anch = self.sess.run(self.loss_, feed_dict=feed_b)
#                 loss_anch_term, summary_str = self.sess.run([self.loss_anchor, self.summary], feed_dict=feed_b)
#                 print('epoch:' + str(ep_) + ' at ' + str(datetime.datetime.now()))
#                 print(', rmse_', np.round(np.sqrt(loss_mse),5), ', loss_anch', np.round(loss_anch,5), ', anch_term', np.round(loss_anch_term,5))
                
            if (ep_ % save_freq) == 0:
                self.save_model(ep_)
            # Shuffle 
            if (ep_ % shuff == 0):
                X_train, y_train = shuffle(X_train, y_train, random_state=ep_)

        return

         