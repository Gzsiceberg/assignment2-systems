
import math


if __name__ == "__main__":
    from rich import print
    d_model = 16384
    d_ff = 53248
    num_blocks = 126

    w_num_per_block = d_model * d_ff * 2 # 2 FFN layers
    w_num = w_num_per_block * num_blocks

    w_size = w_num * 4.0 / (1024**3) # GB
    grad_size = w_size
    opt_size = w_size * 2 # GB
    print("Question a:")
    print(f"Weight size: {w_size:.2f} GB ")
    print(f"Grad size: {grad_size:.2f} GB ")
    print(f"Optimizer size: {opt_size:.2f} GB ")
    print(f"Total size: {w_size + grad_size + opt_size:.2f} GB ")

    batch_size = 128 * 1024
    activations_mem = batch_size * (d_model + d_ff) * 2.0 * num_blocks / (1024**3) # GB
    print(f"Activation size with batch size {batch_size}: {activations_mem:.2f} GB ")
    total_mem = w_size + grad_size + opt_size + activations_mem
    print(f"Total size: {total_mem:.2f} GB ")
    h100_count = math.ceil(total_mem / 80)
    print(f"Number of H100(80GB) needed: {h100_count}")

    print("-" * 80)
    print("Question b:")
    batch_size = 128 * 1024
    activations_mem = batch_size * (d_model + d_ff) * 2.0 * num_blocks * 0.5 / (1024**3) # GB
    print(f"Activation size: {activations_mem:.2f} GB ")
    total_mem = w_size + grad_size + opt_size + activations_mem
    print(f"Total size: {total_mem:.2f} GB ")

    n_fsdp = math.ceil((w_size + grad_size + opt_size + activations_mem) / 95)
    total_mem_fsdp = total_mem / n_fsdp
    print(f"Total size with FSDP({n_fsdp}): {total_mem_fsdp:.2f} GB ")

    print("-" * 80)
    print("Question c:")
    W_ici = 2 * 9 * 10e10
    C = 4.6 * 10e14
    Mx = 2
    My = 1
    X = 16
    Y = 4
    N = X * Y
    alpha = C / W_ici
    predict_batch = (alpha * alpha) / (Mx * My * d_ff) * N


    T_tp = 2 * 2 * batch_size * d_model / (X * W_ici * My) * 1000.0 # ms
    ratio = T_tp / (2 * 2 * d_ff * d_model * batch_size / (N * C) * 1000.0)
    print(f"Tensor T_commu/T_c ratio: {ratio:.2f}")

    T_fsdp = 2 * 2 * d_ff * d_model / (Y * W_ici * Mx) * 1000.0 # ms
    batch_size = math.ceil(T_fsdp / (2 * 2 * d_ff * d_model / (N * C) * 1000.0))
    print(f"Predicted min batch size to hide FSDP communication: {batch_size:.2f} {batch_size/N:.2f} per GPU")
    base_batch_size = batch_size

    for batch_size in [int(base_batch_size * 0.9), base_batch_size, int(base_batch_size * 1.1)]:
        T_fsdp = d_ff / (Y * Mx)
        T_tp = batch_size / (X * My)
        T_comm = max(T_fsdp, T_tp)
        T_c = d_ff * batch_size / (N * (C / W_ici))
        if T_comm > T_c:
            print(f"Commu bound. batch size {batch_size} T_comm: {T_comm:.2f} T_c: {T_c:.2f} T_fsdp: {T_fsdp:.2f} T_tp: {T_tp:.2f}")
        else:
            print(f"C bound. batch size {batch_size} T_comm: {T_comm:.2f} T_c: {T_c:.2f} T_fsdp: {T_fsdp:.2f} T_tp: {T_tp:.2f}")

    print("-" * 80)
    print("Question d:")
    X = 4
    Y = N / X

    T_tp = 2 * 2 * batch_size * d_model / (X * W_ici * My) * 1000.0 # ms
    ratio = T_tp / (2 * 2 * d_ff * d_model * batch_size / (N * C) * 1000.0)
    print(f"Tensor T_commu/T_c ratio: {ratio:.2f}")

    T_fsdp = 2 * 2 * d_ff * d_model / (Y * W_ici * Mx) * 1000.0 # ms
    batch_size = int(T_fsdp / (2 * 2 * d_ff * d_model / (N * C) * 1000.0)) + 1
    print(f"Predicted min batch size to hide FSDP communication: {batch_size:.2f} {batch_size/N:.2f} per GPU")
    base_batch_size = batch_size

    for batch_size in [int(base_batch_size * 0.9), base_batch_size, int(base_batch_size * 1.1)]:
        T_fsdp = d_ff / (Y * Mx)
        T_tp = batch_size / (X * My)
        T_comm = max(T_fsdp, T_tp)
        T_c = d_ff * batch_size / (N * (C / W_ici))
        if T_comm > T_c:
            print(f"Commu bound. batch size {batch_size} T_comm: {T_comm:.2f} T_c: {T_c:.2f} T_fsdp: {T_fsdp:.2f} T_tp: {T_tp:.2f}")
        else:
            print(f"C bound. batch size {batch_size} T_comm: {T_comm:.2f} T_c: {T_c:.2f} T_fsdp: {T_fsdp:.2f} T_tp: {T_tp:.2f}")
