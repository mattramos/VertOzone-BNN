import warnings
import tensorflow as tf
import subprocess
import numpy as np

num_gpus = 4
num_ens_members = 48
num_par_pro = 2
num_ens_per_pro = num_ens_members // (num_gpus * num_par_pro)

running_pro = 1
while running_pro <= num_ens_members:
    for gpu in range(0, num_gpus):
        for pro in range(num_par_pro):
            seed = running_pro + 1
            num_NN0 = running_pro
            num_NNs = num_ens_per_pro
            running_pro += num_NNs
            print('NNs {} to {} running with seed {} on GPU {}'.format(num_NN0, num_NN0 + num_NNs - 1, seed, gpu))
            pro = subprocess.Popen(['python',
                                    'multi_train_BNN.py',
                                    str(seed),
                                    str(gpu),
                                    str(num_NN0),
                                    str(num_NNs)]) #,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# stdout, stderr = pro.communicate()
