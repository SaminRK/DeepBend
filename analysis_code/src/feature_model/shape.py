from util.util import get_possible_seq, get_random_string

import numpy as np
import pandas as pd

import subprocess
import os
from pathlib import Path
import inspect

SHAPE_NAMES = ["EP", "HelT", "MGW", "ProT", "Roll"]
SHAPE_FULL_FORM = {
    "EP": "Electrostatic potential",
    "HelT": "Helix twist",
    "MGW": "Minor groove width",
    "ProT": "Propeller twist",
    "Roll": "Roll",
}


def get_shape(file_path):
    """
    Read and process a shape file written by DNAShapeR
    returns:
        a numpy 2d array with NA values replaced by None
    """
    with open(file_path, "r") as file:
        all_seq_shapes = []

        for data in file.readlines():
            # Identify a new sequence with '>' and 'NA'
            if data[0] == ">":
                continue
            elif data[:2] == "NA":
                shape_str_list = data.strip().split(sep=",")
                shape_float_list = [
                    None if val == "NA" else float(val) for val in shape_str_list
                ]
                all_seq_shapes.append(shape_float_list)
            else:
                shape_str_list = data.strip().split(sep=",")
                shape_float_list = [
                    None if val == "NA" else float(val) for val in shape_str_list
                ]
                all_seq_shapes[-1] += shape_float_list

        return np.array(all_seq_shapes).astype(float)


def run_dna_shape_r_wrapper(df, prune_na):
    """
    Finds DNA Shape values of DNA sequences by running a R script

    args:
        df (pandas.DataFrame): contains DNA sequences in 'Sequence' column
        prune_na (bool): whether to prune NA shape values
    returns:
        a dict mapping keys ['EP', 'HelT', 'MGW', 'ProT', "Roll"] to a 2d shape array
    """
    # Write sequences one per line to feed into R script
    sequence_file = get_random_string(10)

    df["Sequence"].to_csv(sequence_file, sep=" ", index=False, header=False)
    subprocess.run(
        [
            "Rscript",
            "./shape.r",
            f"{Path(inspect.getabsfile(inspect.currentframe())).parent}/{sequence_file}",
        ],
        capture_output=True,
    )
    os.remove(sequence_file)

    shape_arr_map = {}
    for ext in SHAPE_NAMES:
        arr = get_shape(f"{sequence_file}.{ext}")
        if prune_na:
            valid_cols = find_valid_cols(arr[0].flatten())
            arr = arr[:, valid_cols]

        shape_arr_map[ext] = arr
        os.remove(f"{sequence_file}.{ext}")

    return shape_arr_map


def find_valid_cols(arr):
    "Return column indices of not-nan values of a Numpy 1D array"
    mask = np.invert(np.isnan(arr))
    return np.where(mask)[0]


def find_all_shape_values():
    """
    Generate all 5-bp and 6-bp sequences and write their shape value in a file
    """
    write_dir = "data/generated_data"
    five_bp_seqs = get_possible_seq(5)
    six_bp_seqs = get_possible_seq(6)

    five_seq_df = pd.DataFrame({"Sequence": five_bp_seqs})
    shape_arr = run_dna_shape_r_wrapper(five_seq_df, False)

    helt_five_arr = shape_arr[SHAPE_NAMES[1]]
    mgw_arr = shape_arr[SHAPE_NAMES[2]]
    prot_arr = shape_arr[SHAPE_NAMES[3]]
    roll_five_arr = shape_arr[SHAPE_NAMES[4]]

    mgw_df = pd.concat(
        [five_seq_df, pd.DataFrame({"MGW": mgw_arr[:, 2].flatten()})], axis=1
    )
    mgw_df.sort_values("MGW", ignore_index=True, inplace=True)
    mgw_df.to_csv(f"{write_dir}/MGW_possib_values.tsv", sep="\t", index=False)

    prot_df = pd.concat(
        [five_seq_df, pd.DataFrame({"ProT": prot_arr[:, 2].flatten()})], axis=1
    )
    prot_df.sort_values("ProT", ignore_index=True, inplace=True)
    prot_df.to_csv(f"{write_dir}/ProT_possib_values.tsv", sep="\t", index=False)

    six_seq_df = pd.DataFrame({"Sequence": six_bp_seqs})
    six_shape_arr = run_dna_shape_r_wrapper(six_seq_df, False)

    helt_six_arr = six_shape_arr[SHAPE_NAMES[1]]
    roll_six_arr = six_shape_arr[SHAPE_NAMES[4]]

    def get_helt_roll_df(
        five_arr,
        six_arr,
        shape_str: str,
    ):
        """
        Generate df for helical twist and roll shape from five and six sequence arrays
        """
        imaginary_bp = "0"
        imbp_suffixed_five_seq_df = pd.DataFrame(
            {
                "Sequence": [
                    imaginary_bp + seq for seq in five_seq_df["Sequence"].tolist()
                ]
            }
        )
        imbp_prefixed_five_seq_df = pd.DataFrame(
            {
                "Sequence": [
                    seq + imaginary_bp for seq in five_seq_df["Sequence"].tolist()
                ]
            }
        )

        suf_df = pd.concat(
            [
                imbp_suffixed_five_seq_df,
                pd.DataFrame({shape_str: five_arr[:, 1].flatten()}),
            ],
            axis=1,
        )
        pref_df = pd.concat(
            [
                imbp_prefixed_five_seq_df,
                pd.DataFrame({shape_str: five_arr[:, 2].flatten()}),
            ],
            axis=1,
        )
        six_df = pd.concat(
            [six_seq_df, pd.DataFrame({shape_str: six_arr[:, 2].flatten()})], axis=1
        )
        return pd.concat([suf_df, pref_df, six_df], axis=0)

    helt_df = get_helt_roll_df(helt_five_arr, helt_six_arr, "HelT")
    helt_df.sort_values("HelT", ignore_index=True, inplace=True)
    helt_df.to_csv(f"{write_dir}/HelT_possib_values.tsv", sep="\t", index=False)

    roll_df = get_helt_roll_df(roll_five_arr, roll_six_arr, "Roll")
    roll_df.sort_values("Roll", ignore_index=True, inplace=True)
    roll_df.to_csv(f"{write_dir}/Roll_possib_values.tsv", sep="\t", index=False)


def split_values(val_list, n_split):
    """
    Splits a list of floating values into some ranges

    returns:
        a list of values in ascending each of which defining lower value
        of a range (inclusive)
    """
    # just do
    # min + i * (max-min) / n_split
    pass
