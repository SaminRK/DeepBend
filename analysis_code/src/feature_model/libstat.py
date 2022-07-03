from pathlib import Path

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt

from util.constants import (
    TL,
    RL,
    CNL,
    CHRVL,
    LIBL,
    TL_LEN,
    RL_LEN,
    CNL_LEN,
    CHRVL_LEN,
    LIBL_LEN,
)
from util.reader import DNASequenceReader


class LibStat:
    """Statistical data and analysis of libraries"""

    @classmethod
    def len(self, name: str):
        if name == TL:
            return TL_LEN
        if name == RL:
            return RL_LEN
        if name == CNL:
            return CNL_LEN
        if name == CHRVL:
            return CHRVL_LEN
        if name == LIBL:
            return LIBL_LEN

    @classmethod
    def compare_common(self, lib_first: str, lib_sec: str):
        """
        Compare common sequences in two libraries

        Possible combinations: (CNL, TL), (CNL, CHRVL), (RL, LIBL)
        """

        def _get_intersection(df1: pd.DataFrame, df2: pd.DataFrame):
            merged_df = pd.merge(df1, df2, how="inner", on=["Sequence"])
            merged_df.columns = merged_df.columns.str.replace("_", "")

            return merged_df

        reader = DNASequenceReader()
        all_df = reader.get_processed_data()
        common_df = _get_intersection(all_df[lib_first], all_df[lib_sec])

        # Save
        common_seq_dir = f"data/generated_data/common_seq"
        if not Path(common_seq_dir).is_dir():
            Path(common_seq_dir).mkdir(parents=True, exist_ok=True)

        common_df.to_csv(
            f"{common_seq_dir}/{lib_first}_{lib_sec}.tsv", sep="\t", index=False
        )
        print(
            f"mean_C0_{lib_first}",
            common_df["C0x"].mean(),
            f"mean_C0_{lib_sec}",
            common_df["C0y"].mean(),
        )

        corr, _ = pearsonr(common_df["C0x"], common_df["C0y"])
        print("Pearsons correlation: %.3f" % corr)
        print("r2 score: ", r2_score(common_df["C0x"], common_df["C0y"]))

        plt.scatter(common_df["C0x"], common_df["C0y"])
        plt.axis("square")
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.show()
