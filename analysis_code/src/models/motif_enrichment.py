from typing import Any
from keras import Model
from nptyping import NDArray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chromosome.chromosome import C0Spread, ChrmCalc, Chromosome

from models.prediction import Prediction
from models.data_preprocess import Preprocess
from util.constants import CHRVL, GDataSubDir
from util.custom_types import YeastChrNum
from util.reader import DNASequenceReader
from util.util import FileSave, PathObtain


class Enr:
    def __init__(self) -> None:
        self.model_no = 35
        self.model = Prediction(self.model_no)._model

    def enr_chrm_save(self, cnum: YeastChrNum):
        clip = True

        mid_layer_num = 2
        mid_layer_fw_output_model = Model(
            inputs=self.model.input,
            outputs=self.model.layers[mid_layer_num].get_output_at(0),
        )
        mid_layer_rc_output_model = Model(
            inputs=self.model.input,
            outputs=self.model.layers[mid_layer_num].get_output_at(1),
        )

        # df = DNASequenceReader().get_processed_data()[CHRVL]
        df = DNASequenceReader().read_yeast_genome(cnum)
        prep = Preprocess(df)
        data = prep.one_hot_encode()

        X1 = data["forward"]
        X2 = data["reverse"]

        batch_size = 1000
        num_batches = int((X1.shape[0] + batch_size - 1) / batch_size)

        chrom_len = ChrmCalc.total_bp(len(df))
        motif_num = 256
        matching_score = np.zeros((chrom_len, motif_num))
        overlap = np.zeros(chrom_len)
        pos = 0

        for batch_num in range(num_batches):
            x1 = X1[batch_num * batch_size : (batch_num + 1) * batch_size]
            x2 = X2[batch_num * batch_size : (batch_num + 1) * batch_size]

            out: NDArray[(Any, 50, 256), float] = mid_layer_fw_output_model.predict(
                {"forward": x1, "reverse": x2}
            )
            print(f"batch {batch_num+1}/{num_batches}")

            for i in range(out.shape[0]):
                matching_score[pos : pos + 50] += out[i]
                overlap[pos : pos + 50] += 1
                pos += 7

        matching_score = matching_score.T
        matching_score /= overlap

        if clip:
            matching_score = matching_score.clip(0)

        matching_score = matching_score.round(5)
        FileSave.npy(
            matching_score,
            f"{PathObtain.gen_data_dir()}/{GDataSubDir.MOTIF}/match_score_{self.model_no}"
            f"{'_cl' if clip else ''}_{cnum}/score.npy",
        )

        # for mi in range(motif_num):
        #     FileSave.nptxt(
        #         matching_score[mi],
        #         f"{PathObtain.gen_data_dir()}/{GDataSubDir.MOTIF}/match_score"
        #         f"{'_cl' if clip else ''}_V/motif_{mi}",
        #     )


def enr_chrm_save_obsolete(alt=True):
    model_id = "model35_parameters_parameter_274"
    pred = Prediction(35)
    model = pred._model
    museum_layer_num = 2
    museum_layer = model.layers[museum_layer_num]
    _, ic_scaled_prob = museum_layer.get_motifs()
    ic_prob = np.array(ic_scaled_prob)

    chrm = Chromosome("VL", prediction=pred, spread_str=C0Spread.mcvr)
    enc_chrm = Preprocess.oh_enc(chrm.seq)
    print(enc_chrm.shape)

    if alt:
        motif_len = 8
        for mi in range(ic_prob.shape[0]):
            enrichment = np.zeros(chrm.total_bp)
            for pos in range(chrm.total_bp - motif_len):
                region = enc_chrm[pos : pos + motif_len, :]
                enrichment[pos + motif_len // 2] = np.sum(region * ic_prob[mi])

            FileSave.nptxt(
                enrichment,
                f"{PathObtain.gen_data_dir()}/{GDataSubDir.MOTIF}/enr_score_{chrm.number}/motif_{mi}",
            )
    else:
        motif_len = 8
        for mi in range(ic_prob.shape[0]):
            enrichment = np.zeros(chrm.total_bp)
            overlap = np.zeros(chrm.total_bp)
            for pos in range(chrm.total_bp - motif_len):
                region = enc_chrm[pos : pos + motif_len, :]
                enrichment[pos : pos + motif_len] += np.sum(
                    region * ic_prob[mi], axis=1
                )
                overlap[pos : pos + motif_len] += 1
            for i, o in enumerate(overlap):
                if overlap[i] > 1:
                    enrichment[i] /= overlap[i]

            print("writting motif %d enrichment to file" % mi)
            np.savetxt(f"enrichments/{model_id}/motif_{mi}", enrichment, "%.5f")


def show_relative_change_of_motif_enrichment_around_nucleosomes(
    nucleosome_positions: np.ndarray, rng: int, num_motifs: int, gene_len: int
):
    cover = np.zeros(gene_len, dtype=bool)

    for index, nuc_pos in enumerate(nucleosome_positions):
        cover[nuc_pos - rng : nuc_pos + rng] = True

    print(np.count_nonzero(cover))
    for mi in range(num_motifs):
        chrV_enrichment = np.loadtxt("enrichments/model30_trained_9/motif_%d" % mi)
        whole_genome_mean = chrV_enrichment.mean()

        masked_enrichment = np.ma.masked_array(chrV_enrichment, cover)
        nucleosome_mean = masked_enrichment.mean()
        print(
            "motif %d => whole genome %f, nucleosome %f, change %f %%"
            % (
                mi,
                whole_genome_mean,
                nucleosome_mean,
                (nucleosome_mean - whole_genome_mean) / whole_genome_mean * 100,
            )
        )
