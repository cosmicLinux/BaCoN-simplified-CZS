#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
My training file

@author: Ben Bose - 4.11.2022
@editor: Linus Thummel - 31.01.2023
"""

# Training options:
import subprocess

# Type of model
bayesian='True' # Bayesian NN or traditional NN (weights single valued or distributions?)
model_name='custom' # Custom or dummy - dummy has presets as given in models.py

#Test mode?
test_mode='False' # Network trained on 1 batch of minimal size or not?
seed='1312' # Initial seed for test mode batch

# Saving or restoring training
restore='False' # Restore training from a checkpoint?
save_ckpt='True' # Save checkpoints?

# -------------------- change data <<<<<<<<<<<<<<<<<<<
# Directories
mypath=None # Parent directory
# ------ Data folder - which folder to use for training
# new ee2 - wrong n_s
# hmcode
#DIR='data/data_ben/hmcode_updated/hmcode2020/train_data_hmcode-new_5'
# ee2
#DIR='data/data_ben/ee2_obstd_0p016/nu_wb/training_sets_5'
# normalisation file
norm_data_name = '/planck_hmcode2020.txt'
# old directories - 2022
DIR='data/train_data_full' # old bacon github data
#DIR='data/data_ben/training_data_full_noDS' # new data HMcode2020 (massive n + baryons) - without split, 18475 files
curves_folder = 'data/curve_files_sys/curve_files_train1k_2500batchsize_sysFactor0o04_start0o03_dirChange0'

# fine-tune = only 'LCDM' and 'non-LCDM' 
fine_tune='False' # Model trained to distinguish between 'LCDM' and 'non-LCDM' classes or between all clases. i.e. binary or multi-classifier
# k_max either 1 h/Mpc (NB) or 2.5h/Mpc (OB)
k_max='2.5' # which k-modes to include?
sample_pace='4'

# Edit this with base model directory
mdir='models/'
# Set name of output directory (for fine tuning, the same as for the training !)
#fname='NB_split_6_rand_kmax1_planck-HMcode2020_epoch100' # 6
train_fname = 'NB_Halofit_train18k'
test_fname = '5'
planck_fname = 'hmcode2020'
fname_extra = '2500Curves-70epochs-Halofit-withsys-k25'

# >>>>>>>>>>>>>>>>>>> change data ------------------

# noise model
n_noisy_samples='10' # How many noise examples to train on?
# Currently these are drawn from Gaussian with example mean and standard deviation given by cosmic variance + shot noise
# Cosmic variance and shot noise are Euclid-like
add_noise='True'# Add noise?
add_shot='False'# Add shot noise term?
add_sys='True'# Add systematic error term?
add_cosvar='True'# Add cosmic var term?

sys_factor='0.04'
# sys error curves 

GPU='False' # Use GPUs?
val_size='0.15' # Validation set % of data set

add_FT_dense='False' #if True, adds an additional dense layer before the final 2D one
n_epochs='70' # How many epochs to train for?
patience='100' # terminate training after 'patience' epochs if no decrease in loss function
### batch size, is a multiple of #models * noise realisations, e.g. 4 models, 10 noise --> must be multiple of 40
batch_size='2500' # 5 models, 10 noise for 1k-data
lr='0.01' # learning rate
decay='0.95' #decay rate: If None : Adam(lr), 



# -------------------- set name -----------------
#fname='NB_newEE2OmegabTrain5_5_kmax1_planck-hmcode2020_epoch100_train100_noShot_samplePace4_sysScaled-4perc_sysNorm_smurves008_noPrint_noiseSamples4' 
fname = 'curves_' + train_fname + '_' + test_fname + '_samplePace' + sample_pace + '_kmax' + k_max + '_planck-' + planck_fname + '_epoch' + n_epochs 

if add_noise=='True':
   fname = fname + '_noiseSamples' + n_noisy_samples
   if add_cosvar=='True':
      fname = fname + '_wCV'
   else:
      fname = fname + '_noCV'
   if add_shot=='True':
      fname = fname + '_wShot'
   else:
      fname = fname + '_noShot'
   if add_sys=='True':
      fname = fname + '_wSys' 
   else:
      fname = fname + '_noSys'
else:
   fname = fname + '_noNoise'

if GPU=='True':
   fname = fname + '_GPU'
   
fname = fname + '_' + fname_extra
fname = fname.replace('.', 'o')


# -------------------- BNN parameters -----------------

# Example image details
im_depth='500' # Number in z direction (e.g. 500 wave modes for P(k) examples)
im_width='1' # Number in y direction (1 is a 2D image : P(k,mu))
im_channels='4'  # Number in x direction (e.g. 4 redshifts for P(k,z))
swap_axes='True' # Do we swap depth and width axes? True if we have 1D image in 4 channels
sort_labels='True' # Sorts labels in ascending/alphabetical order

z1='0' # which z-bins to include? Assumes each channel is a new z bin.
z2='1' # which z-bins to include? Assumes each channel is a new z bin.
z3='2' # which z-bins to include? Assumes each channel is a new z bin.
z4='3' # which z-bins to include? Assumes each channel is a new z bin.

# Number of layers and kernel sizes
k1='10'
k2='5'
k3='2'
 # The dimensionality of the output space (i.e. the number of filters in the convolution)
f1='8'
f2='16'
f3='32'
 # Stride of each layer's kernel
s1='2'
s2='2'
s3='1'
# Pooling layer sizes
p1='2'
p2='2'
p3='0'
# Strides in Pooling layer
sp1='2'
sp2='1'
sp3='0'

n_dense='1' # Number of dense layers

# labels of different cosmologies
#c0_label = 'lcdm'
#c1_label = 'fR dgp wcdm'

log_path = mdir+fname+'_log'
if fine_tune:
  log_path_original= log_path
  log_path += '_FT'
  log_path_original += '.txt'
else:
   log_path_original=''

log_path += '.txt'

# -------------- adapt c_0 and c_1 =  'fR', 'dgp', 'ds', 'wcdm', 'rand' <<<<<<<<<<<<<<<
proc = subprocess.Popen(["python3", "train.py", "--test_mode" , test_mode, "--seed", seed, \
                  "--bayesian", bayesian, "--model_name", model_name, \
                  "--fine_tune", fine_tune, "--log_path", log_path_original,\
                  "--restore", restore, \
                  "--models_dir", mdir, \
                  "--fname", fname, \
                  "--DIR", DIR, \
                  '--norm_data_name', norm_data_name, \
                  '--curves_folder', curves_folder,\
                  "--c_0", 'lcdm', \
                  "--c_1", 'fR', 'dgp', 'wcdm', 'rand', \
                  "--save_ckpt", save_ckpt, \
                  "--im_depth", im_depth, "--im_width", im_width, "--im_channels", im_channels, \
                  "--swap_axes", swap_axes, \
                  "--sort_labels", sort_labels, \
                  "--add_noise", add_noise, "--add_shot", add_shot, "--add_sys", add_sys,"--add_cosvar", add_cosvar, \
                  "--sample_pace", sample_pace,\
                  "--n_noisy_samples", n_noisy_samples, \
                  "--val_size", val_size, \
                  "--z_bins", z1,z2,z3,z4, \
                  "--filters", f1,f2,f3, "--kernel_sizes", k1,k2,k3, "--strides", s1,s2,s3, "--pool_sizes", p1,p2,p3, "--strides_pooling", sp1,sp2,sp3, \
                  "--k_max", k_max,\
                  "--n_dense", n_dense,\
                  "--add_FT_dense", add_FT_dense, \
                  "--n_epochs", n_epochs, "--patience", patience, "--batch_size", batch_size, "--lr", lr, \
                  "--decay", decay, \
                  "--GPU", GPU],\
                  stdout=subprocess.PIPE, \
                  stderr=subprocess.PIPE)

with open(log_path, "w") as log_file:
  while proc.poll() is None:
     line = proc.stderr.readline()
     if line:
        my_bytes=line.strip()
        print ("err: " + my_bytes.decode('utf-8'))
        log_file.write(line.decode('utf-8'))
     line = proc.stdout.readline()
     if line:
        my_bytes=line.strip()
        print ("out: " + my_bytes.decode('utf-8'))
        log_file.write(line.decode('utf-8'))
