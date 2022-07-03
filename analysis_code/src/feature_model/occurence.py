from __future__ import annotations

from util.util import (
    get_possible_seq,
    gen_random_sequences,
    sorted_split,
    append_reverse_compliment,
    PathObtain,
)
from util.reader import DNASequenceReader
from util.constants import RL

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import regex as re
import swifter


class Occurence:
    """
    Contains functions to calculate unit sequence occurences in DNA sequences
    """

    def find_occurence(self, seq_list, unit_size):
        """
        Find number of occurences of all possible nucleotide sequences of particular size in a list of sequences.
        param:
            seq_list: List of DNA sequences
            unit_size: unit bp sequence size
        returns:
            a dictionary mapping unit nucleotide sequence to number of occurences
        """
        ## TODO: Use dataframe, assign, lambda function, str.count on each row, then sum.
        possib_seq = get_possible_seq(unit_size)
        seq_occur_map = dict()

        for seq in possib_seq:
            seq_occur_map[seq] = 0

        for whole_seq in seq_list:
            for i in range(len(whole_seq) - unit_size + 1):
                seq_occur_map[whole_seq[i : i + unit_size]] += 1

        return seq_occur_map

    def find_occurence_individual(self, df: pd.DataFrame, k_list: list) -> pd.DataFrame:
        """
        Find occurences of all possible nucleotide sequences for individual DNA sequences.

        Args:
            df: column `Sequence` contains DNA sequences
            k_list: list of unit sizes to consider

        Returns:
            A dataframe with columns added for all considered unit nucleotide sequences.
        """
        df = df.copy()
        possib_seq = []
        for k in k_list:
            possib_seq += get_possible_seq(k)

        for seq in possib_seq:
            df[seq] = (
                df["Sequence"]
                .swifter.progress_bar(False)
                .apply(lambda x: len(re.findall(seq, x, overlapped=True)))
            )

        return df

    def find_seq_occur_map(
        self, dfs: list[pd.DataFrame], unit_len: int
    ) -> dict[str, np.ndarray]:
        """
        Counts number of occurences of nucleotide sequences of length unit_len in a
        list of dataframes.

        Args:
            dfs: A list of Dataframes which contains DNA sequences in 'Sequence' column
            unit_len: length of unit nucleotides n_bins: Number of bins to split to

        Returns:
            A dictionary mapping possible nucleotide seqs to a numpy array.
        """
        # For each bin, find occurence of bp sequences
        possib_seq = get_possible_seq(unit_len)

        seq_occur_map = dict()  # mapping of possib_seq to a list of occurences

        for seq in possib_seq:
            seq_occur_map[seq] = np.array([])

        for sdf in dfs:
            one_bin_occur_dict = self.find_occurence(sdf["Sequence"].tolist(), unit_len)
            for unit_seq in seq_occur_map:
                seq_occur_map[unit_seq] = np.append(
                    seq_occur_map[unit_seq], one_bin_occur_dict[unit_seq]
                )

        return seq_occur_map

    def normalize_bin_occurence(self, seq_occur_map: dict, bin_len: int) -> dict:
        """
        Normalizes occurences of nucleotides in bins.
        """
        # Generate a large random list of equal length bp DNA sequences
        # and find avg. occurences for a bin
        GENERATE_TIMES = 100
        num_random_sequences = bin_len * GENERATE_TIMES
        random_seq_list = gen_random_sequences(num_random_sequences)
        random_list_occur_dict = self.find_occurence(
            random_seq_list, len(list(seq_occur_map.keys())[0])
        )

        for unit_seq in random_list_occur_dict:
            random_list_occur_dict[unit_seq] /= GENERATE_TIMES

        # Normalize
        for unit_seq in seq_occur_map:
            seq_occur_map[unit_seq] /= random_list_occur_dict[unit_seq]

        return seq_occur_map

    def plot_dinucleotide_heatmap(self, df: pd.DataFrame, df_name: str):
        """
        Plot heatmap of occurence of dinucleotides.

        Splits the sequence list into bins with C0 value.
        """
        N_BINS = 12

        # Get sorted split bins of equal size
        n_seq = len(df) - len(df) % N_BINS
        df = df.iloc[:n_seq, :]
        sorted_dfs = sorted_split(df, n=len(df), n_bins=N_BINS, ascending=True)

        seq_occur_map = self.find_seq_occur_map(sorted_dfs, 2)

        # Plot barplots

        # Sort by occurence in first bin in descending order
        sorted_occur = sorted(
            seq_occur_map.items(), key=lambda x: x[1][0], reverse=True
        )
        arr = np.array([pair[1] for pair in sorted_occur])
        assert arr.shape == (4**2, N_BINS)

        # seaborn.heatmap(arr, linewidth=0.5)
        plt.imshow(arr, cmap="jet", aspect="auto")
        plt.colorbar()
        plt.yticks(
            ticks=np.arange(len(sorted_occur)),
            labels=[pair[0] for pair in sorted_occur],
        )
        plt.savefig(f"{PathObtain.figure_dir()}/seq_occur/{df_name}_bidir_heatmap.png")
        norm_seq_occur_map = self.normalize_bin_occurence(
            seq_occur_map, len(sorted_dfs[0])
        )

        # Sort by occurence in first bin in descending order
        norm_sorted_occur = sorted(
            norm_seq_occur_map.items(), key=lambda x: x[1][0], reverse=True
        )
        arr = np.array([pair[1] for pair in norm_sorted_occur])
        assert arr.shape == (4**2, N_BINS)

        plt.close()
        plt.clf()
        # seaborn.heatmap(arr, linewidth=0.5)
        plt.imshow(arr, cmap="jet", aspect="auto")
        plt.colorbar()
        plt.yticks(
            ticks=np.arange(len(sorted_occur)),
            labels=[pair[0] for pair in sorted_occur],
        )
        plt.savefig(
            f"{PathObtain.figure_dir()}/seq_occur/{df_name}_bidir_norm_heatmap.png"
        )

    def plot_boxplot(self, df: pd.DataFrame, library_name: str):
        """
        Plot boxplots of occurence of dinucleotides.

        Splits the sequence list into bins with C0 value.
        """
        N_BINS = 12

        # Get sorted split bins of equal size
        n_seq = len(df) - len(df) % N_BINS
        df = df.iloc[:n_seq, :]
        sorted_dfs: list[pd.DataFrame] = sorted_split(
            df, n=len(df), n_bins=N_BINS, ascending=True
        )
        assert len(sorted_dfs) == N_BINS

        # Gathered by bins
        all_dnc_bin_occur: list[pd.DataFrame] = list(
            map(self.find_occurence_individual, sorted_dfs, [[2]] * len(sorted_dfs))
        )
        assert (
            len(all_dnc_bin_occur) == N_BINS
        ), f"{len(all_dnc_bin_occur)} is not equal to {N_BINS}"
        assert len(all_dnc_bin_occur[0]) == len(df) // N_BINS
        all_dnc = get_possible_seq(2)

        # Have a list (of 1D numpy arrays for each bin) for each dinucleotide
        # Gathered by dnc
        dnc_occur_map: dict[str, list[np.ndarray]] = dict(
            map(
                lambda dnc: (
                    dnc,
                    list(map(lambda df: df[dnc].to_numpy(), all_dnc_bin_occur)),
                ),
                all_dnc,
            )
        )
        assert len(list(dnc_occur_map.values())[0]) == N_BINS
        assert list(dnc_occur_map.values())[0][0].shape == (len(df) // N_BINS,)

        # Plot
        # mpl.rcParams['figure.figsize'] = 60, 60
        fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
        for i, dnc in enumerate(all_dnc):
            axs[i // 4, i % 4].set_title(dnc)
            axs[i // 4, i % 4].boxplot(dnc_occur_map[dnc], showfliers=False)

        fig.tight_layout()
        plt.savefig(
            f"{PathObtain.figure_dir()}/seq_occur/{library_name}_boxplot_serial.png"
        )
        plt.show()

        # Sort by occurence in first bin in descending order
        # sorted_occur = sorted(seq_occur_map.items(), key=lambda x: x[1][0], reverse=True)
        # arr = np.array([ pair[1] for pair in sorted_occur ])
        # assert arr.shape == (4**2, N_BINS)

    def plot_boxplot_lib(self, library_name: str):
        reader = DNASequenceReader()
        all_lib = reader.get_processed_data()
        df = append_reverse_compliment(all_lib[library_name])
        self.plot_boxplot(df, library_name)
