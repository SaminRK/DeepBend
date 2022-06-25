import time
import sys
import random
import numpy as np
import tensorflow as tf
import os
import pickle
import argparse

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from statistics import mean

from train import store_model_performance, store_model_weights, store_training_history, train
from utils.data_preprocess import get_dataset
from models.model_dispatcher import get_model
from utils.utils import get_hyperparameters, get_cross_validation_id, get_model_id


def main():
    # excute the code
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--encoding", default="one-hot", choices=["one-hot", "dinucleotide"])
    parser.add_argument("--dataset")
    parser.add_argument("--k", default=10, type=int)
    parser.add_argument("--hyperparameters")

    args = parser.parse_args()

    if not (args.model and args.encoding and args.dataset and args.k and args.hyperparameters):
        parser.error("Provide all arguments correctly")

    model_name = args.model
    data_filename = args.dataset
    hyperparameter_filename = args.hyperparameters
    encoding = args.encoding
    num_folds = args.k

    hyperparams = get_hyperparameters(hyperparameter_filename)

    dataset = get_dataset(data_filename, encoding)
    
    Y = dataset["readout"]
    X1 = dataset["forward"]
    X2 = dataset["reverse"]

    np.set_printoptions(threshold=sys.maxsize)
    val_r2s = []
    val_spearmans = []
    val_pearsons = []
    train_r2s = []
    train_spearmans = []
    train_pearsons = []
    validation_dataset_size = Y.shape[0] // num_folds
    cv_id = get_cross_validation_id(model_name, hyperparameter_filename, data_filename)
    cross_validation_results_filename = f"{cv_id}.csv"

    first_line = "k,train,,,test,,\n"
    second_line = ",r2,pearsonr,spearmanr,r2,pearsonr,spearmanr\n"
    with open(cross_validation_results_filename, "w") as f:
        print(first_line + second_line, file=f)

    for k in range(num_folds):
        # Reproducibility
        seed = random.randint(1, 1000)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        b1 = int(k*validation_dataset_size)
        b2 = int((k+1)*validation_dataset_size)

        print(f"Validation sequences: {b1}-{b2}")

        val_x1 = X1[b1:b2]
        val_x2 = X2[b1:b2]
        val_y = Y[b1:b2]

        if k > 0 and k+1 < num_folds:
            train_x1 = np.concatenate((X1[:b1], X1[b2:]))
            train_x2 = np.concatenate((X2[:b1], X2[b2:]))
            train_y = np.concatenate((Y[:b1], Y[b2:]))
        elif k == 0:
            train_x1 = X1[b2:]
            train_x2 = X2[b2:]
            train_y = Y[b2:]
        else:
            train_x1 = X1[:b1]
            train_x2 = X2[:b1]
            train_y = Y[:b1]

        nn = get_model(model_name)(hyperparameters=hyperparams)
        model = nn.create_model()
        
        history, validation_history = train(model, hyperparams, 
                                            {"forward": train_x1, "reverse": train_x2, "readout": train_y},
                                            {"forward": val_x1, "reverse": val_x2, "readout": val_y})
        
        print(f"Seed number is {seed}")

        model_id = get_model_id(model_name, hyperparameter_filename, f"{data_filename}:{b1}:{b2}", seed)
        store_model_weights(model, model_id, "model_weights")
        # store_training_history(model_id, history, "training_histories")
        # store_model_performance(model.metrics_names, hyperparams, model_id, history, 
        #                     val_history, "model_performances.csv")

        val_history = model.evaluate({'forward': val_x1, 'reverse': val_x2}, val_y)

        print('metric values of model.evaluate: ' + str(val_history))
        print('metrics names are ' + str(model.metrics_names))

        val_predictions = model.predict({'forward': val_x1, 'reverse': val_x2}).flatten()
        val_r2 = r2_score(val_y, val_predictions)
        val_pearson = pearsonr(val_y, val_predictions)
        val_spearman = spearmanr(val_y, val_predictions)

        val_r2s.append(val_r2)
        val_pearsons.append(val_pearson[0])
        val_spearmans.append(val_spearman[0])

        train_predictions = model.predict({'forward': train_x1, 'reverse': train_x2}).flatten()
        train_r2 = r2_score(train_y, train_predictions)
        train_pearson = pearsonr(train_y, train_predictions)
        train_spearman = spearmanr(train_y, train_predictions)

        train_r2s.append(train_r2)
        train_pearsons.append(train_pearson[0])
        train_spearmans.append(train_spearman[0])

        with open(cross_validation_results_filename, "a") as f:
            line = f"{k},{train_r2},{train_pearson[0]},{train_spearman[0]},{val_r2},{val_pearson[0]},{val_spearman[0]}"
            print(line, file=f)

        print('train')
        print('r2_score: ' + str(train_r2))
        print('pearsonr: ' + str(train_pearson))
        print('spearmanr: ' + str(train_spearman))
        print('validation')
        print('r2_score: ' + str(val_r2))
        print('pearsonr: ' + str(val_pearson))
        print('spearmanr: ' + str(val_spearman))

    with open(cross_validation_results_filename, "a") as f:
        line = f"mean,{mean(train_r2s)},{mean(train_pearsons)},{mean(train_spearmans)},{mean(val_r2s)},{mean(val_pearsons)},{mean(val_spearmans)}"
        print(line, file=f)


    # reports time consumed during execution (secs)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    sys.exit(main())
