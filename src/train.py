import time
import sys
import random
import numpy as np
import tensorflow as tf
import os
import pickle
import argparse

from tensorflow.keras.callbacks import EarlyStopping

from utils.data_preprocess import get_dataset
from models.model_dispatcher import get_model
from utils.utils import get_model_id, get_hyperparameters


def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--encoding", default="one-hot", choices=["one-hot", "dinucleotide"])
    parser.add_argument("--train-dataset")
    parser.add_argument("--validation-dataset")
    parser.add_argument("--hyperparameters")

    args = parser.parse_args()

    if not (args.model and args.train_dataset and args.validation_dataset and args.hyperparameters and args.encoding):
        parser.error("Provide all arguments correctly")

    model_name = args.model
    hyperparameter_filename = args.hyperparameters
    train_filename = args.train_dataset
    validation_filename = args.validation_dataset
    encoding = args.encoding
    hyperparams = get_hyperparameters(hyperparameter_filename)

    # For reproducibility
    seed = random.randint(1, 1000)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    nn = get_model(model_name)(hyperparameters=hyperparams)
    model = nn.create_model()

    train_dataset = get_dataset(train_filename, encoding)
    validation_dataset = get_dataset(validation_filename, encoding)
    
    history, val_history = train(model, hyperparams, train_dataset, validation_dataset)
                            
    np.set_printoptions(threshold=sys.maxsize)

    print(f"Seed number is {seed}")

    if not os.path.isdir("model_weights"):
        os.mkdir("model_weights")
    
    model_id = get_model_id(model_name, hyperparameter_filename, train_filename, seed)
    
    store_model_weights(model, model_id, "model_weights")
    store_training_history(model_id, history, "training_histories")
    store_model_performance(model.metrics_names, hyperparams, model_id, history, 
                            val_history, "model_performances.csv")

    print("--- %s seconds ---" % (time.time() - start_time))


def train(model, hyperparams, train_dataset, validation_dataset):
    y_train = train_dataset["readout"]
    x1_train = train_dataset["forward"]
    x2_train = train_dataset["reverse"]

    y_val = validation_dataset["readout"]
    x1_val = validation_dataset["forward"]
    x2_val = validation_dataset["reverse"]

    early_stopping_callback = EarlyStopping(patience=15, verbose=1, restore_best_weights=True)

    history = model.fit({"forward": x1_train, "reverse": x2_train}, y_train,
                        epochs=hyperparams["epochs"], batch_size=hyperparams["batch_size"],
                        validation_data=({"forward": x1_val, "reverse": x2_val}, y_val),
                        callbacks=[early_stopping_callback])

    val_history = model.evaluate({"forward": x1_val, "reverse": x2_val}, y_val)

    return history, val_history


def store_model_weights(model, model_id, directory):
    if not os.path.isdir(directory):
            os.mkdir(directory)
    model.save_weights(f"model_weights/{model_id}", save_format="h5")
    

def store_training_history(model_id, history, directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    with open(f"{directory}/{model_id}", "wb") as f:
        pickle.dump(history.history, f)


def store_model_performance(metrics_names, 
                            hyperparams, 
                            model_id, 
                            history, 
                            validation_history,
                            csv_filename):
    if not os.path.isfile(csv_filename):
        metric_str = ",".join(metrics_names)
        parameter_str = ",".join(hyperparams.keys())
        first_line = ",validation,,,train,,,parameters\n"
        second_line = "model_id," + metric_str + "," + metric_str + "," + parameter_str + "\n"
        with open(csv_filename, "w") as f:
            print(first_line+second_line, file=f)

    with open(csv_filename, "a") as f:
        line = f"{model_id}," + ",".join(["{:.5f}".format(x) for x in validation_history]) + ","
        line += ",".join(["{:.5f}".format(history.history[metric_name][-1])
                          for metric_name in metrics_names]) + ","
        line += ",".join([str(v) for v in hyperparams.values()])
        print(line, file=f)


if __name__ == "__main__":
    sys.exit(main())
