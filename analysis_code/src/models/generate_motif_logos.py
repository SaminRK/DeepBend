import io
import os
import sys

import numpy as np
import pandas as pd
import logomaker as lm
import matplotlib as mpl
import matplotlib.pyplot as plt
from Bio import motifs

from models.prediction import Prediction
from util.constants import FigSubDir, GDataSubDir
from util.util import FileSave, PlotUtil


def gen_motif_logos():
    save_fig = True
    save_cons = False
    with_num = True
    format_ = "png"

    model = Prediction(35)._model

    np.set_printoptions(threshold=sys.maxsize, precision=5, suppress=True)

    museum_layer_num = 2
    museum_layer = model.layers[museum_layer_num]
    _, ic_scaled_prob = museum_layer.get_motifs()

    npa = np.array(ic_scaled_prob)
    mtfs = []
    for i in range(npa.shape[0]):
        df = pd.DataFrame(npa[i], columns=["A", "C", "G", "T"])
        print(df.head())
        # rc('font', weight='bold')
        # plt.rcParams["font.weight"] = "bold"
        # plt.rcParams["axes.labelweight"] = "bold"
        if save_cons:
            tm = "temp"
            df.to_csv(tm, sep=" ", index=False, header=False)
            with open(tm) as f:
                m = motifs.read(f, "pfm-four-columns")
                mtfs.append(m.degenerate_consensus)

            os.remove(tm)

        if save_fig:
            PlotUtil.font_size(48)
            logo = lm.Logo(
                df,
                font_name="Arial Rounded MT",
                color_scheme={
                    'A': 'r', 
                    'C': 'b',
                    'G': 'y',
                    'T': 'g'
                }
            )
            logo.ax.set_ylim((0, 2.2))
            logo.ax.set_ylabel("bits")
            logo.ax.set_yticks([0, 2])
            logo.ax.set_xticks(range(0, 8))
            if with_num:
                logo.ax.add_patch(
                    mpl.patches.Rectangle((-0.5, 1.84), 2.2, 0.36, fc="lightskyblue")
                )
                logo.ax.text(
                    0.6,
                    1.85,
                    f"#{i}",
                    color="b",
                    fontsize=60,
                    font="Arial Rounded MT",
                    ha="center",
                )

            if format_ == "png":
                FileSave.figure_in_figdir(f"{FigSubDir.LOGOS}/logo_{str(i)}.png")
            else:
                FileSave.figure_in_figdir(f"{FigSubDir.LOGOS}/logo_{str(i)}.svg")

            plt.tight_layout()

    if save_cons:
        FileSave.tsv_gdatadir(
            pd.DataFrame({"Motifs": mtfs}),
            f"{GDataSubDir.MOTIF}/motif_m35_consensus.tsv",
        )
