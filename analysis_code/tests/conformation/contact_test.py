import numpy as np

from chromosome.chromosome import Chromosome
from conformation.contact import Contact


class TestContact:
    # TODO: Use fixture for Chromosome('VL')
    def test_correlation_own(self):
        contact = Contact(Chromosome("VL"))
        pearsons, spearmans = contact.correlate_with_c0("own")
        assert 0 < pearsons < 1
        assert 0 < spearmans < 1

    def test_correlation_adjacent(self):
        contact = Contact(Chromosome("VL"))
        pearsons, spearmans = contact.correlate_with_c0("adjacent")
        assert 0 < pearsons < 1
        assert 0 < spearmans < 1
        print("Pearsons", pearsons)
        print("Spearmans", spearmans)

    def test_plot_correlation(self):
        contact = Contact(Chromosome("VL"))
        figpath = contact.plot_correlation_with_c0("adjacent")
        assert figpath.is_file()

    def test_show(self):
        contact = Contact(Chromosome("VL"))
        fig_path = contact.show()
        assert fig_path.is_file()

    def test_generate_matrix(self):
        contact = Contact(Chromosome("VL"))
        mat = contact._generate_mat()
        assert np.count_nonzero(mat) > 0.08 * mat.size
        assert np.count_nonzero(mat.diagonal()) > 0.9 * mat.diagonal().size

    def test_load(self):
        contact = Contact(Chromosome("VL"))
        df = contact._load_contact()
        assert df.columns.tolist() == ["row", "col", "intensity"]
        print(len(df))
        assert len(df) > 0
