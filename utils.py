import tensorflow as tf
import numpy as np

def initialize_uninitialized(sess):
    
    global_vars = tf.global_variables()
    is_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
        
    return

# Functions for retrievals from NNs

def batch_retrieve(NN, var, X):
    # Use this to batch process large datasets
    # Useful on lower memory machines
    if X.shape[0] <= 20000:
        out_var = NN.sess.run(var, feed_dict={NN.inputs: X})
    else:
        batches_out = []
        batches_in = np.array_split(X, np.ceil(X.shape[0] // 20000) , axis=0)
        for b in batches_in:
            batch_out = NN.sess.run(var, feed_dict={NN.inputs: b})
            if batch_out.ndim == 1:
                batches_out.append(batch_out.reshape(-1,1))
            else:
                batches_out.append(batch_out)
        out_var = np.vstack(batches_out)
    return out_var

def get_pre_softmax_stds(NNs, X):
    stds = np.zeros(shape=[X.shape[0], len(NNs)])
    for i, NN in enumerate(NNs):
        stds[:, i] = np.std(NN.sess.run(NN.layer2_out, feed_dict = {NN.inputs: X}), axis=1)
    return stds

def get_weights(NNs, X):
    weights = np.zeros(shape=[X.shape[0], NNs[0].n_mdls, len(NNs)])
    for i, NN in enumerate(NNs):
        weights[:, :, i] = batch_retrieve(NN, NN.model_coeffs, X)
    return weights

def get_bias(NNs, X):
    bias = np.zeros(shape=[X.shape[0], len(NNs)])
    for i, NN in enumerate(NNs):
        bias[:, i] = batch_retrieve(NN, NN.modelbias, X).ravel()
    return bias

def get_aleatoric_noise(NNs, X):
    noise = np.zeros(shape=[X.shape[0], len(NNs)])
    for i, NN in enumerate(NNs):
        noise[:, i] = np.sqrt(batch_retrieve(NN, NN.noise_sq, X).ravel())
    return noise


def predict_ensemble(NNs, X):
    y_pred=[]
    y_pred_noise_sq=[]
    for i, NN in enumerate(NNs):
        y_pred.append(NN.predict(X))
        y_pred_noise_sq.append(batch_retrieve(NN, NN.noise_sq, X))
    y_preds_train = np.array(y_pred)
    y_preds_noisesq_train = np.array(y_pred_noise_sq)
    y_preds_mu_train = np.mean(y_preds_train, axis=0)
    y_preds_std_train_epi = np.std(y_preds_train, axis=0)
    y_preds_std_train = np.sqrt(np.mean((y_preds_noisesq_train + np.square(y_preds_train)), axis = 0) - np.square(y_preds_mu_train))
    
    return y_preds_train, y_preds_mu_train, y_preds_std_train, y_preds_std_train_epi, y_preds_noisesq_train

# Functions for other

def calc_RMSE(y_pred, y):
    return np.sqrt(np.mean(np.square(y_pred.ravel() - y.ravel())))

def calc_NLL(y_pred, y, y_std):
    return np.mean(0.5 * (((y_pred.ravel() - y.ravel()) ** 2) / ((y_std.ravel()**2))
                    + np.log(y_std.ravel() ** 2) + np.log(2 * np.pi)))

def report_on_percentiles(y_pred, y, y_std):

    n = len(y.ravel())

    n1 = np.sum(np.abs(y_pred.ravel() - y.ravel()) <= y_std.ravel() * 1)
    n2 = np.sum(np.abs(y_pred.ravel() - y.ravel()) <= y_std.ravel() * 2)
    n3 = np.sum(np.abs(y_pred.ravel() - y.ravel()) <= y_std.ravel() * 3)
    print('Using {} data points'.format(n))
    print('{} within 1 std'.format(100 * n1 / n))
    print('{} within 2 std'.format(100 * n2 / n))
    print('{} within 3 std'.format(100 * n3 / n))

    return