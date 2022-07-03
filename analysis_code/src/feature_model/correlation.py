from pathlib import Path

from feature_model.helsep import HSAggr
from .data_organizer import (
    DataOrganizeOptions,
    DataOrganizer,
    SequenceLibrary,
    TrainTestSequenceLibraries,
)
from util.constants import CNL, TL, RL
from util.reader import SEQ_COL, SEQ_NUM_COL, C0_COL
from util.util import PathObtain, FileSave


class Correlation:
    @classmethod
    def kmer_corr(self, library: SequenceLibrary) -> list[Path]:
        libraries = TrainTestSequenceLibraries(
            train=[library],
            test=[],
        )

        filepaths = []
        for k in [2, 3, 4]:
            options = DataOrganizeOptions(k_list=[k])
            organizer = DataOrganizer(libraries, None, None, options)
            kmer_df, _ = organizer._get_kmer_count()
            kmer_df = kmer_df.drop(columns=[SEQ_NUM_COL, SEQ_COL])

            path = Path(
                f"{PathObtain.gen_data_dir()}/correlation/{library.name}_{k}_corr.tsv"
            )
            filepaths.append(path)
            FileSave.tsv(
                kmer_df.corr()[C0_COL].sort_values(ascending=False),
                path,
                index=True,
                precision=2,
            )

        return filepaths

    @classmethod
    def helsep_corr(self, lib: SequenceLibrary, hsaggr=HSAggr.MAX) -> Path:
        libraries = TrainTestSequenceLibraries(
            train=[lib],
            test=[],
        )

        organizer = DataOrganizer(
            libraries, None, None, DataOrganizeOptions(hsaggr=hsaggr)
        )
        hel_df = (organizer._get_helical_sep()[0]).drop(columns=[SEQ_NUM_COL, SEQ_COL])
        return FileSave.tsv(
            hel_df.corr()[C0_COL].sort_values(ascending=False),
            f"{PathObtain.gen_data_dir()}/correlation/{lib.name}_hel_{hsaggr.value}_corr.tsv",
            index=True,
            precision=2,
        )
