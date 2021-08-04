# VertOzone-BNN
A Bayesian neural network (BNN) to fuse chemistry-climate models with observations to produce a continuous record of vertically resolved ozone.

This is an extension of the work from [Sengupta et. al., (2020)](URL 'https://proceedings.neurips.cc/paper/2020/file/0d5501edb21a59a43435efa67f200828-Paper.pdf') which looked at total ozone column. The scripts that follow are the most up-to-date scripts for Bayesian neural network gophysical model ensembling, and work with tensorflow v1.x

Check out this binder example of how ensembling geophysical models with Bayesian neural networks works! [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mattramos/Toy-bayesian-neural-network-ensemble/master?filepath=toy_dataset_example.ipynb)

## Data descriptions
- [pretrained-checkpoints](URL 'https://github.com/mattramos/VertOzone-BNN/tree/master/pretrained-checkpoints') contains the 48 pretrained BNN ensemble members
- [outputRAW](URL 'https://github.com/mattramos/VertOzone-BNN/tree/master/outputRAW') contains the log scaled prediction and uncertainty fields and the defining coordinates.
- zmo3_BNNOz.nc is the predicted output of zonally averaged vertically resolved ozone field including uncertainty.

## File description
- utils.py and preprocess_data.py contain utility functions that are data specific. All the data mapping and coordiante scaling is handled by these scripts as well as function that retrieve variables such as model weights and model bias.
- train_dispatch.py is used to distribute training of the BNN to multiple GPUs. In this example it is set up for 4 GPUs. This script calls multi_train_BNN.py which trains the individual networks and contains parameters such as number of epochs etc.
- checking_priors.ipynb checks that the priors have been appropriately encoded within the BNN
- load_and_save_raw_output.ipynb loads model checkpoints, either self trained or pretrained, outputs predictions and uncertainties, and performs basic validation testing. 
- Plotting_for_publication.ipynb has a small script for plotting some of the output.
- constructing_zmo3_output.ipynb is the script used to construct the netCDF file that contains the zonal mean ozone prediction and its uncertainty. 
