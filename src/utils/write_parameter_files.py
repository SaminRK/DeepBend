start = 718

filter_0s = [128, 176, 256, 512]
filter_1s = [16, 32, 64]
counter = 0

for filter_0 in filter_0s:
    for filter_1 in filter_1s:
        with open(f"parameters/parameter{start+counter}.txt", "w") as f:
            print(
                """filters_0 {filter_0}
filters_1 {filter_1}
kernel_size_0 8
regularizer_2 l2
epochs 25
batch_size 1024
loss_func coeff_determination
optimizer adam""".format(
                    filter_0=filter_0, filter_1=filter_1
                ),
                file=f,
            )
            counter += 1
