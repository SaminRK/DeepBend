start = 718

filter_0s = [128, 176, 256, 512]
filter_1s = [16, 32, 64]
counter = 0

for filter_0 in filter_0s:
    for filter_1 in filter_1s:
        with open(f'parameters/parameter{start+counter}.txt', 'w') as f:
            print(
                """filters_0 {filter_0}
filters_1 {filter_1}
kernel_size_0 8
regularizer_2 l2
epochs 25
batch_size 1024
loss_func coeff_determination
optimizer adam""".format(filter_0=filter_0, filter_1=filter_1), file=f,)
            counter += 1


# start = 493

# alphas = [75, 100, 250, 500, 1000]
# beta_dens = [10, 20, 50, 75, 100]
# filter_0s = [128, 176, 256]
# filter_1s = [16, 32, 64]
# counter = 0

# for alpha in alphas:
#     for beta_den in beta_dens:
#         for filter_0 in filter_0s:
#             for filter_1 in filter_1s:
#                 with open(f'parameters/parameter{start+counter}.txt', 'w') as f:
#                     print(
#                         """filters_0 {filter_0}
# filters_1 {filter_1}
# kernel_size_0 8
# regularizer_2 l2
# epochs 15
# batch_size 1024
# loss_func coeff_determination
# optimizer adam
# alpha {alpha}.0
# beta {beta}""".format(filter_0=filter_0, filter_1=filter_1, alpha=alpha, beta=1/beta_den),
#                         file=f,
#                     )
#                     counter += 1

