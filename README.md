# DeepBend
This repository contains code, models and dataset of the paper titled [DeepBend: An Interpretable Model of DNA Bendability](https://doi.org/10.1016/j.isci.2023.105945).

## Table of Content
- [DeepBend](#deepbend)
  - [Dataset](#dataset)
  - [Model](#model)
  - [Setup](#setup)
  - [Training and evaluation](#training-and-evaluation)
  - [Exploring a trained model](#exploring-a-trained-model)
  - [References](#references)
## Dataset
We have used 4 datasets from [Basu et al. 2021](#references) for training and testing our models. These datasets are 1. *random*, 2. *nucleosomal*, 3. *tiling* and 4. *chrV* datasets. For the *tiling* and *chrV* dataset, we designed special train and test datasets to avoid leakage due to overlapping sequences. These are: 5. *tiling_train*, 6. *tiling_test*, 7. *chrV_train* and 8. *chrV_test*. All these datasets are in the directory `data/`

## Model
Our model has been implemented in `tensorflow v2` and is designated as `model35`. The hyperparameters for our model can be changed. Some examples of hyperparameters are kept in the folder `hyperparameters/`. The weights of a pre-trained model with the hyperparamters from `hyperparameters/parameter.txt` are in the file `model_weights/model35_parameters_parameter_274`.

## Setup
In order to setup our repository, at first clone it into your local machine, create a virtual environment and install all dependencies into it. This can be done using the following commands in the terminal.
```bash
git clone https://github.com/SameeLab-BCM/DeepBend.git
cd DeepBend
virtualenv .deepbend-env
source .deepbend-env/bin/activate
pip install -r requirements.txt
```
The required data for running our codes are available in the following public Google Drive folder [https://drive.google.com/drive/folders/1Rb143RvlVEu8rciV7e0BTWUMkk72t4VM?usp=sharing](https://drive.google.com/drive/folders/1Rb143RvlVEu8rciV7e0BTWUMkk72t4VM?usp=sharing). Download the entire folder and unzip it in the project root directory.

## Training and evaluation
For training the model, the following command can be used. Here we are training the DeepBend model with hyperparamters from `hyperparameters/hyperparameter262.txt`, on the *chrV_train* dataset and using the tiling_test dataset for validation. The weights of the trained model will be saved in the directory `model_weights/` with the name depending on the model name, hyperparamters file name, training dataset and the seed. 
```bash
python src/train.py --model model35 --train-dataset data/chrV_train.txt \
    --validation-dataset data/tiling_test.txt \
    --hyperparameters hyperparameters/hyperparameter262.txt
```
In order to test the trained model we can use the following command. We are testing on the *chrV_test* dataset here.
```bash
python src/test.py --model model35 --test-dataset data/chrV_test.txt \
    --hyperparameters hyperparameters/hyperparameter262.txt \
    --model-weights model_weights/<model_weights_file_name>
```
For k-fold cross validation, we can use the following command. Here we are doing 10-fold cross validation on the *nucleosomal* dataset.
```bash
python src/cross_validate.py --model model35 --dataset data/nucleosomal.txt \ 
  --k 10 --hyperparameters hyperparameters/hyperparameter262.txt
```
Examples of hyperparameter files can be found [here](https://github.com/SameeLab-BCM/DeepBend/tree/master/hyperparameters). The hyperparameters are arranged in the form of key-value pairs in each line. Here `filters_n` means the number of filters in the n-th layer. `kernel_size_n` is the size of the kernel in the n-th layer. `regularizer_2` indicates the regularizer (`l1`, `l2`, `l1l2`, `lvariance` etc.) used in the last layer. `alpha` indicates the alpha parameter for multinomial convolution layer, whereas `A`, `C`, `G` & `T` are the background parameters.

## Exploring a trained model

After training models, we can interpret our models to get patterns, motifs and other information from our model.
In the `notebook notebooks/model_motifs_patterns.ipynb` we have shown how motifs and first-order patterns are derived from a trained DeepBend model.
In the `notebook notebooks/higher_order_relations.ipynb` we have shown the relation between the first-order patterns of a pairs of motifs.
In the `notebook notebooks/motif_GAAGAGC.ipynb` we have found out information about the GAAGAGC 7-mer. 
In the `notebook notebooks/motif_presence_in_nucleosomal_region`.ipynb we have found the presence of motifs around the nucleosomal regions.

## References
Basu, Aakash, Dmitriy G. Bobrovnikov, Zan Qureshi, Tunc Kayikcioglu, Thuy T. M. Ngo, Anand Ranjan, Sebastian Eustermann, et al. 2021. “Measuring DNA Mechanics on the Genome Scale.” Nature 589 (7842): 462–67. [https://doi.org/10.1038/s41586-020-03052-3](https://doi.org/10.1038/s41586-020-03052-3)
