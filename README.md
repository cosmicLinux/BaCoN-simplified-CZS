# BaCoN for the Carl-Zeiss-Stiftung Summer School 2023

This is a simplfied version of the BaCoN classifier for modification at the CSZ summer school 2023. The original BaCoN code can be found [here](https://github.com/Mik3M4n/BaCoN). 

## Run on Goolge Colab
We recommend to clone this github repo to a personal google drive and then run it in google colab. This can be done when loading the notebook in Colab. (Select the GPU runtime. go to the arrow at the upper right corner and then select 'Change runtime type' -> GPU). 

## BaCoN (BAyesian COsmological Network)

BaCoN allows to train and test Bayesian Convolutional Neural Networks in order to **classify dark matter power spectra as being representative of different cosmologies**, as well as to compute the classification confidence. 
The code now supports the following theories:  **LCDM, wCDM, f(R), DGP, and a randomly generated class** (see the reference for details).

**We also provide a jupyter notebook that allows to load the pre-trained model, classify any matter power spectrum and compute the classification confidence with the method described in the paper (see [4 - Classification](https://github.com/Mik3M4n/BaCoN#4---Classification)). This only requires the raw values of the power spectrum. Feedback on the results of classification is particularly welcome!**

The first release of BaCoN was accompagnied by the paper [Seeking New Physics in Cosmology with Bayesian Neural Networks I: Dark Energy and Modified Gravity](https://arxiv.org/abs/2012.03992). 

Bibtex:

```
@misc{mancarella2020seeking,
      title={Seeking New Physics in Cosmology with Bayesian Neural Networks I: Dark Energy and Modified Gravity}, 
      author={Michele Mancarella and Joe Kennedy and Benjamin Bose and Lucas Lombriser},
      year={2020},
      eprint={2012.03992},
      archivePrefix={arXiv},
      primaryClass={astro-ph.CO}
}
```

data generated with React by Ben Bose 
synthetic data of matter power spectra 
using massive neutrinos and baryonic effects


## Overview and code organisation


The package provides the following modules:

* ```data generator.py```: data generator that generates batches of data. Data are dark matter power spectra normalised to the Planck LCDM cosmology, in the redshift bins (0.1,0.478,0.783,1.5) and k in the range 0.01-2.5 h Mpc^-1.
* ```models.py``` : contains models' architecture
* ```train.py```: module to train and fine-tune models. Checkpoints are automatically saved each time the validation loss decreases. Both bayesian and "traditional" NNs are available.
* ```test.py```: evaluates the accuracy and the confusion matrix.

A jupyter notebook to classify power spectra with pre-trained weights and computing the confidence in classification is available in ```notebooks/```. 

The first base model is a five-label classifier with LCDM, wCDM, f(R), DGP, and "random" as classes, while the second is a two-label classifier with classes LCDM and non-LCDM.

Details on training, data preparation, variations ot the base model, and extensions are available in the dedicated sections. The Bayesian implementation uses [Tensorflow probability](https://www.tensorflow.org/probability) with [Convolutional](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/Convolution1DFlipout) and [DenseFlipout](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseFlipout) methods.



## Data

### Data folders

THe data folders contain training sets with 1000 or 100 samples per class and testing with 1000 or 10 (for quick tests) samples per class. The organisation of the  ```data/``` folder looks as follows:
```bash
data/training_set/
		├── dgp/
			├──1.txt
			├──2.txt
			├──...
		├── fR/
			├──1.txt
			├──2.txt
			├──...
		├── lcdm/
			├──1.txt
			├──2.txt
			├──...
		├── rand/
			├──1.txt
			├──2.txt
			├──...	
		├── wcdm/
			├──1.txt
			├──2.txt
			├──...	
		└── planck.txt		
```
The data folder structures of the provided samples are already adapted to the code. For completness, here are the requirements on the data sets:


The file names in each folder must have the same indexing, with the same label numbers in all cosmology subfolders. Note that despite the names, data in each folder should be uncorrelated. Nonetheless, the data generator shuffles the indexes so that different indexes in different folders are used when producing a batch, in order to further prevent any spurious correlation that may arise when generating data.
The data generator will automatically check the data folder and assign labels corresponding to the subfolders' names. At test time, make sure that the test data folder has the same names as the training set in order not to run into errors.
The file ```planck.txt``` contains the reference power spectrum used to normalize the input to the network. It can be renamed in the ```train_parameter.py``` files.

### Data format

Each file should contain in the first column the values of k and in each of the following columns the values of P(k) at different redshift bins ordered from the highest to the lowest redshift. The data are generated in redshift bins (0.1,0.478,0.783,1.5) [If generating new data with different redshifst bins, note that the noise in Eq. 3.1 makes use of the galaxy number density and comoving volume at the corresponding z, so this should be changed in ```data_generator.py``` if using different redshift bins].

The provided data contain 500 redshift bins between 0.01 and 10 h Mpc^-1. In order to down-sample to have a number of k bins comparable to Euclid, we sample one every four points. This is done with the option ```sample_pace=4``` (default) in ```DataGenerator ``` (see next section) or when using ```train.py``` (see *Usage* ). Furthermore, we restrict to k<2.5 h Mpc^_1 . This can be done in two ways: either by specifying ```k_max=2.5``` (default) in ```DataGenerator ```/```train.py```, in which case the code will restrict to k<k\_max, or specifying ```i_max```, i.e. the index suck that all k<k[i_max] are <k\_max. (Note that this cut is applied after the sampling).  

The default values are ```sample_pace=4```, ```k_max=2.5```.



## Usage

### Training networks

To train a model and save the result in ```models/my_model/```:

```
python train.py --fname='my_model'
```

Note that if running on Google Colab, the saving of the log file contained in the code sometime fails. In this case the output has to be manually redirected to file. This can be done with ```tee```:

```
python train.py --fname='my_model' | tee my_model_log.txt; mv my_model_log.txt my_model/my_model
```

The code allows to vary the number of convolutional and dense layers, filter size, stride, padding, kernel size, batch normalization and pooling layers, number of z bins and max k.

Full list of options:

```
train.py [-h] [--bayesian BAYESIAN] [--test_mode TEST_MODE]
                [--n_test_idx N_TEST_IDX] [--seed SEED]
                [--fine_tune FINE_TUNE] [--one_vs_all ONE_VS_ALL]
                [--c_0 C_0 [C_0 ...]] [--c_1 C_1 [C_1 ...]]
                [--dataset_balanced DATASET_BALANCED]
                [--include_last INCLUDE_LAST] [--log_path LOG_PATH]
                [--restore RESTORE] [--fname FNAME] [--model_name MODEL_NAME]
                [--my_path MY_PATH] [--DIR DIR] [--TEST_DIR TEST_DIR]
                [--models_dir MODELS_DIR] [--save_ckpt SAVE_CKPT]
                [--out_path_overwrite OUT_PATH_OVERWRITE]
                [--im_depth IM_DEPTH] [--im_width IM_WIDTH]
                [--im_channels IM_CHANNELS] [--swap_axes SWAP_AXES]
                [--sort_labels SORT_LABELS] [--normalization NORMALIZATION]
                [--sample_pace SAMPLE_PACE] [--k_max K_MAX] [--i_max I_MAX]
                [--add_noise ADD_NOISE] [--n_noisy_samples N_NOISY_SAMPLES]
                [--add_shot ADD_SHOT] [--add_sys ADD_SYS]
                [--sigma_sys SIGMA_SYS] [--z_bins Z_BINS [Z_BINS ...]]
                [--n_dense N_DENSE] [--filters FILTERS [FILTERS ...]]
                [--kernel_sizes KERNEL_SIZES [KERNEL_SIZES ...]]
                [--strides STRIDES [STRIDES ...]]
                [--pool_sizes POOL_SIZES [POOL_SIZES ...]]
                [--strides_pooling STRIDES_POOLING [STRIDES_POOLING ...]]
                [--add_FT_dense ADD_FT_DENSE] [--trainable TRAINABLE]
                [--unfreeze UNFREEZE] [--lr LR] [--drop DROP]
                [--n_epochs N_EPOCHS] [--val_size VAL_SIZE]
                [--test_size TEST_SIZE] [--batch_size BATCH_SIZE]
                [--patience PATIENCE] [--GPU GPU] [--decay DECAY]
                [--BatchNorm BATCHNORM]
```

### Output
The results will be saved in a new folder inside the folder specified in the input parameter ```mdir```. The folder name is passed to the code through the parameter ```fname```. 
At the end of training, the ouput folder will contain (see ```models/``` for an example):

* a plot of the learning curves (training and validation accuracies and losses) in ```hist.png```.
*  a subfolder tf_ckpts containing:
	* 	 the checkpoints of the models 
	* 	  four ```.txt``` files with the history of test and validation accuracies and losses, namely:
		* hist_accuracy.txt : training accuracy
		* hist_val\_accuracies.txt: validation accuracy
		* hist_loss.txt: training loss
		* hist_val\_loss.txt: validation loss
	* two files ```idxs_train.txt``` and ```idxs_val.txt``` containing the indexes of the files used for the training and validaiton sets, to make the training set reproducible (note however that noise realizations will add randomness to the underlying spectra.).


### Testing
The ```test.py``` module loads the weights of a previoulsy trained model, computes the accuracy on the test data and outputs a confusion matrix. The result will be saved in the same folder as the training.

For a bayesian net, classification is performed by drawing MC samples from the weights and averaging the result. The number of samples (default 100) can be specified via ```n_monte_carlo_samples ```. One example is classified if the max probability among the labels exceeds a threshold values of 0.5 . This threshold can be changed with the argument ```th_prob ```. In the confiusion matrix, a "Non Classified" class is added to account for this examples. THe accuracy is also computed leaing the un classified examples out, for comparison.

One can also vary the number of noise realizations and noise options at test time.
Finally, if using only one noise realization ```n_noisy_samples =1```, there is the possibility of saving the indexes of the spectra used in each batch, in order to be able to check the corresponding parameters' values. This is done by setting ```save_indexes='True'```.

To test the five-label network:

```
python test.py --log_path='models/five_label/five_label_log.txt'
```


Full options:

```
test.py [-h] --log_path LOG_PATH [--TEST_DIR TEST_DIR]
               [--models_dir MODELS_DIR]
               [--n_monte_carlo_samples N_MONTE_CARLO_SAMPLES]
               [--th_prob TH_PROB] [--batch_size BATCH_SIZE]
               [--add_noise ADD_NOISE] [--n_noisy_samples N_NOISY_SAMPLES]
               [--add_shot ADD_SHOT] [--add_sys ADD_SYS]
               [--sigma_sys SIGMA_SYS] [--save_indexes SAVE_INDEXES]
```

Happy training! 
