import sys
import time
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

from plotnine import ggplot, aes, xlim, ylim, stat_bin_2d

from models.model_dispatcher import get_model
from utils.utils import get_hyperparameters
from utils.data_preprocess import get_dataset

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--model-weights")
    parser.add_argument("--encoding", default="one-hot", choices=["one-hot", "dinucleotide"])
    parser.add_argument("--test-dataset")
    parser.add_argument("--hyperparameters")

    args = parser.parse_args()
    if not (args.model and args.model_weights and args.encoding and args.hyperparameters):
        parser.error("Provide all arguments correctly")

    model_name = args.model
    model_weights_filename = args.model_weights
    test_filename = args.test_dataset
    hyperparameter_filename = args.hyperparameters
    encoding = args.encoding

    hyperparams = get_hyperparameters(hyperparameter_filename)

    nn = get_model(model_name)(hyperparameters=hyperparams)
    model = nn.create_model()
    model.load_weights(model_weights_filename)

    np.set_printoptions(threshold=sys.maxsize, precision=5, suppress=True)

    test_dataset = get_dataset(test_filename, encoding)

    y = test_dataset["readout"]
    x1 = test_dataset["forward"]
    x2 = test_dataset["reverse"]

    history = model.evaluate({"forward": x1, "reverse": x2}, y)

    print("metric values of model.evaluate: " + str(history))
    print("metrics names are " + str(model.metrics_names))

    predictions = model.predict({"forward": x1, "reverse": x2}).flatten()
    print("r2_score: " + str(r2_score(y, predictions)))
    print("pearsonr: " + str(pearsonr(y, predictions)))
    print("spearmanr: " + str(spearmanr(y, predictions)))

    scatterplot_values(predictions, y)

    # reports time consumed during execution (secs)
    print("--- %s seconds ---" % (time.time() - start_time))


def scatterplot_values(predicted_values, true_values):
    df = pd.DataFrame({"predicted_value": predicted_values, "true_value": true_values})
    p = (
        ggplot(data=df, mapping=aes(x="true_value", y="predicted_value"))
        + stat_bin_2d(bins=150)
        + xlim(-2.75, 2.75)
        + ylim(-2.75, 2.75)
    )
    print(p)


if __name__ == "__main__":
    sys.exit(main())
