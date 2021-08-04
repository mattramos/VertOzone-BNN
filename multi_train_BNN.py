import sys
import os
from sklearn.utils import shuffle

args = sys.argv
# print(args)

seed = int(args[1])
GPU_num = args[2]
num_NN0 = int(args[3])
num_NNs = int(args[4])
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_num
print('Running on GPU: {}'.format(GPU_num))
print('Setting seed as: {}'.format(seed))

import tensorflow as tf
from BNN import BNN
from preprocess_data import read_data
import numpy as np
import warnings
warnings.filterwarnings('ignore')
tf.logging.set_verbosity(tf.logging.ERROR)

# Set the seed
np.random.seed(seed)
tf.set_random_seed(seed)

# Load data
X_train, y_train, X_test, y_test, X_interp, y_interp, X_extrap, y_extrap, X_at, y_at = read_data('vmro3_refC1SD_70x36_13mdls_masked_extrap_and_interp.pkl')
X_train, y_train = shuffle(X_train, y_train)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

checkpoint_dir = './checkpoints/model{}/'
log_dir = './logs/model{}/'
x_dim = X_train.shape[1]
y_dim = y_train.shape[1]
num_models = 13
n_data = X_train.shape[0]
hidden_size = 500
learning_rate = 0.0001

# Priors 
bias_mean = 0.00
bias_std = 0.02
noise_mean = 0.03
noise_std = 0.02
layer_scale = 1.1

alpha_dim = x_dim - num_models
init_std_1w =  np.sqrt(150.0/(alpha_dim))
init_std_1b = init_std_1w
init_std_2w =  (layer_scale)/np.sqrt(hidden_size)
init_std_2b = init_std_2w
init_std_biasw = (1.05 * layer_scale * bias_std)/np.sqrt(hidden_size)
init_std_noisew = (2.5)/np.sqrt(hidden_size)
lambda_anchor = 1.0/(np.array([init_std_1w,init_std_1b,init_std_2w,init_std_2b,init_std_biasw,init_std_noisew])**2)

print('Building NNs {} to {}'.format(num_NN0, num_NN0 + num_NNs - 1))

# Build the NNs within the ensemble
NNs = []
for i in range(num_NN0, num_NN0 + num_NNs):
    NNs.append(BNN(sess, checkpoint_dir.format(i),
                   log_dir.format(i),
                   x_dim, y_dim, num_models, n_data,
                   hidden_size, learning_rate, lambda_anchor, 
                   init_std_1w, init_std_1b, init_std_2w,
                   init_std_2b, init_std_biasw, init_std_noisew))

batch_size = 7500
n_epochs = 100000
ep_0 = 0

# Use a command line like this to monitor training on tensorboard
# and go to localhost:6006 in a browser
# tensorboard --logdir ./logs --host=127.0.0.1

# Train
for i, NN in enumerate(NNs):
    print('NN:',num_NN0 + i)
    NN.train(n_epochs, ep_0, X_train, y_train, shuff=100, batch_size=batch_size, print_freq=25, save_freq=5000)
    
