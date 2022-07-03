import numpy as np

from feature_model.model import Model, ModelRunner, ModelCat
from feature_model.data_organizer import (
    TrainTestSequenceLibraries,
    SequenceLibrary,
    DataOrganizeOptions,
    DataOrganizer,
)
from feature_model.feat_selector import FeatureSelectorFactory
from util.constants import TL, RL


class TestModel:
    def test_classify(self):
        libraries = TrainTestSequenceLibraries(
            train=[SequenceLibrary(name=TL, quantity=200)],
            test=[SequenceLibrary(name=RL, quantity=50)],
        )
        options = DataOrganizeOptions(
            k_list=[2],
            range_split=np.array([0.2, 0.6, 0.2]),
            binary_class=False,
            balance=False,
            c0_scale=20,
        )

        featsel = FeatureSelectorFactory("corr").make_feature_selector()
        organizer = DataOrganizer(libraries, None, featsel, options)

        Model.run_seq_classifier(*organizer.get_seq_train_test(classify=True))

    def test_regression(self):
        libraries = TrainTestSequenceLibraries(
            train=[SequenceLibrary(name=TL, quantity=200)],
            test=[SequenceLibrary(name=RL, quantity=50)],
        )
        options = DataOrganizeOptions(
            k_list=[2],
            c0_scale=1,
        )
        featsel = FeatureSelectorFactory("all").make_feature_selector()
        organizer = DataOrganizer(libraries, None, featsel, options)

        Model.run_seq_regression(*organizer.get_seq_train_test(classify=False))


class TestModelRunner:
    def test_classification(self):
        libraries = TrainTestSequenceLibraries(
            train=[SequenceLibrary(name=TL, quantity=200)],
            test=[SequenceLibrary(name=RL, quantity=50)],
        )
        options = DataOrganizeOptions(
            k_list=[2],
            range_split=np.array([0.2, 0.6, 0.2]),
            binary_class=False,
            balance=False,
            c0_scale=20,
        )

        featsel = FeatureSelectorFactory("corr").make_feature_selector()

        ModelRunner.run_model(libraries, options, featsel, ModelCat.CLASSIFIER)

    def test_regression(self):
        libraries = TrainTestSequenceLibraries(
            train=[SequenceLibrary(name=TL, quantity=200)],
            test=[SequenceLibrary(name=RL, quantity=50)],
        )

        options = DataOrganizeOptions(k_list=[2])

        featsel = FeatureSelectorFactory("corr").make_feature_selector()

        ModelRunner.run_model(libraries, options, featsel, ModelCat.REGRESSOR)
