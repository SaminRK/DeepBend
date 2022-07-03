# Generate a motif logo from a meme-suite motif file

GEN_DATA_DIR=/media/sakib/Windows/sakib/programming/playground/machine_learning/bendability/data/generated_data
BND_DIR=allchrm_s_mcvr_m_35_bndh_res_200_lim_250_perc_1.0
DMN_DIR=allchrm_s_mcvr_m_35_dmnsh_bndh_res_200_lim_250_perc_1.0
BND_SEQ_DIR=${GEN_DATA_DIR}/boundaries/${BND_DIR}
DMN_SEQ_DIR=${GEN_DATA_DIR}/domains/${DMN_DIR}
MEME_DIR=${DMN_SEQ_DIR}/streme_out_mnw_8_mxw_8_pt_10_th_0.2_cnt

MOTIF_NO=8
ceqlogo -i${MOTIF_NO} ${MEME_DIR}/streme.txt -f PNG -d "" -o ${MEME_DIR}/logos/${MOTIF_NO}.png