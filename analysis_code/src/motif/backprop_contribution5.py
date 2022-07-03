import sys
import time
from pathlib import Path
import inspect

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logomaker as lm
import cv2
from numpy.testing._private.utils import assert_almost_equal

from models.cnnmodel import CNNModel30
from models.data_preprocess import Preprocess
from keras import Model
from util.reader import DNASequenceReader
from util.util import FileSave, PathObtain
from util.constants import RL

part_model = [None] * 20


def layer_output(model, x1, x2, layer_num, input_num=0):
    intermediate_layer_model = part_model[layer_num]
    intermediate_output = intermediate_layer_model({"forward": x1, "reverse": x2})[
        input_num
    ]

    return intermediate_output


# for 'same' padding


def layer_influence_whole(weights, output, pad="valid"):
    if pad == "same":
        padding = weights.shape[1] - 1
    else:
        padding = 0
    whole_influence = np.zeros(output.shape[0])
    print(whole_influence)
    for position in range(output.shape[0]):
        start_pos = max(position, int(padding / 2)) - int(padding / 2)
        end_pos = min(output.shape[0], position - int(padding / 2) + weights.shape[0])
        start_pos_w = min(0, start_pos)
        end_pos_w = start_pos_w + end_pos - start_pos
        influence = np.multiply(
            output[
                start_pos:end_pos,
            ],
            weights[start_pos_w:end_pos_w],
        )
        print(influence)


def layer_influence_provided_output(
    weights, intermediate_output, position=0, pad="valid"
):
    if pad == "same":
        padding = weights.shape[0] - 1
    else:
        padding = 0
    # print(intermediate_output.shape)
    # print(weights.shape)
    # print(padding)
    start_pos = max(position, int(padding / 2)) - int(padding / 2)
    # print("wtf %d %d" % (position, min(position - int(padding/2), 0)))
    end_pos = min(
        intermediate_output.shape[0], position - int(padding / 2) + weights.shape[0]
    )
    # print('(%d %d)' % (start_pos, end_pos))

    start_pos_w = min(0, start_pos)
    end_pos_w = start_pos_w + end_pos - start_pos

    # print('(%d %d)' % (start_pos_w, end_pos_w))
    # can be done for all inputs together
    influence = np.multiply(
        intermediate_output[
            start_pos:end_pos,
        ],
        weights[start_pos_w:end_pos_w],
    )

    return influence


def layer_kernel_influence_provided_output(
    weights, intermediate_output, position=0, pad="valid"
):
    influence = layer_influence_provided_output(
        weights, intermediate_output, position, pad
    )
    kernel_influence = np.sum(influence, axis=0)

    return kernel_influence


def layer_position_influence_provided_output(
    weights, intermediate_output, position=0, pad="valid"
):
    influence = layer_influence_provided_output(
        weights, intermediate_output, position, pad
    )
    kernel_influence = np.sum(influence, axis=1)

    return kernel_influence


def backprop_contribution5(argv=None):
    """
    Find contribution of bases at each position of a 50-bp sequence to C0
    - Outputs of various layers are considered
    - Arithmetic operations of weights of different layers determine contribution
    - Contribution from forward and reverse sequences are added
    - Create motif logo from contribution
    """
    np.set_printoptions(threshold=sys.maxsize)

    dim_num = (-1, 50, 4)
    nn = CNNModel30(
        dim_num=dim_num,
        filters=512,
        kernel_size=50,
        pool_type="Max",
        regularizer="L_2",
        activation_type="linear",
        epochs=20,
        batch_size=1024,
        loss_func="coeff_determination",
        optimizer="Adam",
    )
    model = nn.create_model()
    model.load_weights(
        f"{PathObtain.parent_dir(inspect.currentframe())}/w30_train_9.h5"
    )

    part_model[7] = Model(inputs=model.input, outputs=model.layers[7].output)
    part_model[8] = Model(inputs=model.input, outputs=model.layers[8].output)
    part_model[10] = Model(inputs=model.input, outputs=model.layers[10].output)
    part_model[11] = Model(inputs=model.input, outputs=model.layers[11].output)
    part_model[12] = Model(inputs=model.input, outputs=model.layers[12].output)

    prep = Preprocess(DNASequenceReader().get_processed_data()[RL][2599:2604])
    # if want mono-nucleotide sequences
    data = prep.one_hot_encode()
    # if want dinucleotide sequences
    # dict = prep.dinucleotide_encode()

    fw_fasta = data["forward"]
    rc_fasta = data["reverse"]
    readout = data["target"]

    # change from list to numpy array
    x1 = np.asarray(fw_fasta)
    x2 = np.asarray(rc_fasta)
    y = np.asarray(readout)

    predictions = model.predict({"forward": x1, "reverse": x2}).flatten()

    weights_9 = np.array(tf.transpose(model.layers[9].get_weights()[0], [2, 0, 1]))
    weights_13 = np.array(tf.transpose(model.layers[13].get_weights()[0], [2, 0, 1]))

    batch_size = 500
    num_batches = int((x1.shape[0] + batch_size - 1) / batch_size)
    layer = model.layers[2]
    weights = layer.get_weights()
    w = tf.transpose(weights[0], [2, 0, 1])
    alpha = 75.0
    # beta = 1 / alpha
    bkg = tf.constant([0.295, 0.205, 0.205, 0.295])
    bkg_tf = tf.cast(bkg, tf.float32)
    ll = tf.map_fn(
        lambda x: tf.subtract(
            tf.subtract(
                tf.subtract(
                    tf.math.scalar_mul(alpha, x),
                    tf.expand_dims(
                        tf.math.reduce_max(tf.math.scalar_mul(alpha, x), axis=1), axis=1
                    ),
                ),
                tf.expand_dims(
                    tf.math.log(
                        tf.math.reduce_sum(
                            tf.math.exp(
                                tf.subtract(
                                    tf.math.scalar_mul(alpha, x),
                                    tf.expand_dims(
                                        tf.math.reduce_max(
                                            tf.math.scalar_mul(alpha, x), axis=1
                                        ),
                                        axis=1,
                                    ),
                                )
                            ),
                            axis=1,
                        )
                    ),
                    axis=1,
                ),
            ),
            tf.math.log(
                tf.reshape(
                    tf.tile(bkg_tf, [tf.shape(x)[0]]),
                    [tf.shape(x)[0], tf.shape(bkg_tf)[0]],
                )
            ),
        ),
        w,
    )
    prob = tf.map_fn(
        lambda x: tf.multiply(
            tf.reshape(
                tf.tile(bkg_tf, [tf.shape(x)[0]]), [tf.shape(x)[0], tf.shape(bkg_tf)[0]]
            ),
            tf.exp(x),
        ),
        ll,
    )

    # plp = tf.scalar_mul(1.442695041, tf.multiply(prob, ll))
    # ic = tf.reduce_sum(plp, axis=2)
    # ic_scaled_prob = tf.multiply(prob, tf.expand_dims(ic, axis=2))

    probability_matrices = np.array(prob)

    for batch_num in range(int(3 * num_batches / 4), num_batches):
        X1 = x1[batch_num * batch_size : (batch_num + 1) * batch_size]
        X2 = x2[batch_num * batch_size : (batch_num + 1) * batch_size]

        intermediate_output_7 = part_model[7]({"forward": X1, "reverse": X2})
        intermediate_output_8 = part_model[8]({"forward": X1, "reverse": X2})
        intermediate_output_10 = part_model[10]({"forward": X1, "reverse": X2})
        intermediate_output_11 = part_model[11]({"forward": X1, "reverse": X2})
        intermediate_output_12 = part_model[12]({"forward": X1, "reverse": X2})

        for seq_index in range(len(X1)):
            fw_contribs = np.zeros((50, 512))
            rc_contribs = np.zeros((50, 512))

            final_layer_influence = layer_influence_provided_output(
                weights_13[0], intermediate_output=intermediate_output_12[seq_index]
            )

            conv2_arrange1_output = intermediate_output_10[seq_index]
            conv2_arrange2_output = intermediate_output_11[seq_index]

            for p in range(conv2_arrange1_output.shape[0]):
                for k in range(conv2_arrange1_output.shape[1]):
                    weights = np.zeros((17, 512))
                    weights[0, :] = weights_9[k][0, :]
                    weights[8, :] = weights_9[k][1, :]
                    weights[16, :] = weights_9[k][2, :]

                    if conv2_arrange1_output[p][k] > conv2_arrange2_output[p][k]:
                        intermediate_output = intermediate_output_7[seq_index]
                    else:
                        intermediate_output = intermediate_output_8[seq_index]
                    padding = weights.shape[0] - 1
                    start_pos = max(p, int(padding / 2)) - int(padding / 2)
                    end_pos = min(
                        intermediate_output.shape[0],
                        p - int(padding / 2) + weights.shape[0],
                    )
                    start_pos_w = min(0, start_pos)
                    end_pos_w = start_pos_w + end_pos - start_pos

                    influence = np.multiply(
                        intermediate_output[
                            start_pos:end_pos,
                        ],
                        weights[start_pos_w:end_pos_w],
                    ).clip(min=0)
                    sum_influence = np.sum(influence)
                    influence = final_layer_influence[p][k] * influence / sum_influence
                    padded_influence = np.zeros((50, 512))
                    padded_influence[
                        start_pos:end_pos,
                    ] = influence
                    if conv2_arrange1_output[p][k] > conv2_arrange2_output[p][k]:
                        fw_contribs = np.add(fw_contribs, padded_influence)
                    else:
                        rc_contribs = np.add(rc_contribs, padded_influence)
            fw_result = np.sum(fw_contribs)
            rc_result = np.sum(rc_contribs)
            result = fw_result + rc_result
            # print('y = %.5f, predictions = %.5f, fw = %.5f, rc = %.5f, conv sum = %.5f last sum = %.5f' % (
            #     y[seq_index], predictions[seq_index], fw_result, rc_result, result, np.sum(max_layer_influence)))
            fw_seq_contribs = np.zeros((50, 4))
            rc_seq_contribs = np.zeros((50, 4))
            for p in range(fw_contribs.shape[0]):
                for k in range(fw_contribs.shape[1]):
                    weights = probability_matrices[k % probability_matrices.shape[0]]
                    fw_seq = X1[seq_index]
                    rc_seq = X2[seq_index]
                    padding = weights.shape[0] - 1
                    start_pos = max(p, int(padding / 2)) - int(padding / 2)
                    end_pos = min(
                        fw_seq.shape[0], p - int(padding / 2) + weights.shape[0]
                    )
                    start_pos_w = min(0, start_pos)
                    end_pos_w = start_pos_w + end_pos - start_pos

                    fw_influence = np.multiply(
                        fw_seq[
                            start_pos:end_pos,
                        ],
                        weights[start_pos_w:end_pos_w],
                    )
                    rc_influence = np.multiply(
                        rc_seq[
                            start_pos:end_pos,
                        ],
                        weights[start_pos_w:end_pos_w],
                    )
                    fw_influence = (
                        fw_contribs[p][k] * fw_influence / np.sum(fw_influence)
                    )
                    rc_influence = (
                        rc_contribs[p][k] * rc_influence / np.sum(rc_influence)
                    )

                    padded_influence = np.zeros((50, 4))
                    padded_influence[
                        start_pos:end_pos,
                    ] = fw_influence
                    fw_seq_contribs = np.add(fw_seq_contribs, padded_influence)

                    padded_influence = np.zeros((50, 4))
                    padded_influence[
                        start_pos:end_pos,
                    ] = rc_influence
                    rc_seq_contribs = np.add(rc_seq_contribs, padded_influence)
            seq_result = np.sum(fw_seq_contribs) + np.sum(rc_seq_contribs)
            print(
                "%d => y = %.5f, predictions = %.5f, seq contrib sum = %.5f, conv sum = %.5f last sum = %.5f"
                % (
                    batch_num * batch_size + seq_index,
                    y[batch_num * batch_size + seq_index],
                    predictions[batch_num * batch_size + seq_index],
                    seq_result,
                    result,
                    np.sum(final_layer_influence),
                )
            )

            # fw_seq_contribs = np.sum(fw_seq_contribs, axis=1)
            # rc_seq_contribs = np.sum(rc_seq_contribs, axis=1)
            # plt.plot(fw_seq_contribs)
            # plt.plot(rc_seq_contribs)
            # plt.ylim((-1, 1))
            # plt.show()

            seq_contribs = np.add(fw_seq_contribs, np.flip(rc_seq_contribs))
            df = pd.DataFrame(seq_contribs, columns=["A", "C", "G", "T"])
            print(seq_index)
            if seq_index == 1:
                assert_almost_equal(
                    np.round(df[:2].to_numpy(), 3).tolist(),
                    [[0.016, 0.0, 0.0, 0.0], [0.0, 0.0, 0.013, 0.0]],
                    decimal=3,
                )

            elif seq_index == 2:
                assert_almost_equal(
                    np.round(df[-2:].to_numpy(), 3).tolist(),
                    [[0.0, -0.008, 0.0, 0.0], [0.0, 0.0, 0.0, -0.008]],
                    decimal=3,
                )
            elif seq_index == 3:
                assert_almost_equal(
                    np.round(df[23:25].to_numpy(), 3).tolist(),
                    [[0.0, -0.004, 0.0, 0.0], [-0.002, 0.0, 0.0, 0.0]],
                    decimal=3,
                )

            # logo = lm.Logo(df)
            fig_path = FileSave.figure(
                f"{PathObtain.figure_dir()}/contribution_patterns/model30_train_9_top/seq_"
                + str(batch_num * batch_size + seq_index)
                + ".png"
            )

            image = cv2.imread(str(fig_path))
            cv2.putText(
                image,
                "actual %f prediction %f"
                % (
                    y[batch_num * batch_size + seq_index],
                    predictions[batch_num * batch_size + seq_index],
                ),
                (320, 80),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(fig_path), image)

            # fw_df = pd.DataFrame(fw_seq_contribs, columns=['A', 'C', 'G', 'T'])
            # rc_df = pd.DataFrame(rc_seq_contribs, columns=['A', 'C', 'G', 'T'])
            # # print(df.head())
            # ax = plt.subplot(2, 1, 1)
            # ay = plt.subplot(2, 1, 2)

            # logo1 = lm.Logo(fw_df, ax=ax)
            # logo2 = lm.Logo(rc_df, ax=ay)
            # plt.savefig('contribution_patterns/model30_train_9/seq_' +
            #             str(seq_index) + '.png', dpi=400)
            # plt.close()

    for i in range(5):
        assert Path(
            f"{PathObtain.figure_dir()}/contribution_patterns/model30_train_9_top/seq_{i}.png"
        ).is_file()
