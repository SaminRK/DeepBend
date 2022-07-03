import keras

from models.train import train


class TestTrain:
    def test_load_model(self):
        model, history = train()
        print(history.history)
        assert isinstance(model, keras.Model)
        assert set(history.history.keys()) == {
            "loss",
            "coeff_determination",
            "spearman_fn",
            "val_loss",
            "val_spearman_fn",
            "val_coeff_determination",
        }
        assert history.history["coeff_determination"][-1] > 0.25
        assert history.history["spearman_fn"][-1] > 0.5
        assert history.history["val_coeff_determination"][-1] > -0.2
        assert history.history["val_spearman_fn"][-1] > 0.1
