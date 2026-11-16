
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time
import statistics

from harness import emit_verify_tag

STUDENT_ID = 230103250
P_PHYS = 8
N = 256
SIZES = [8, 16, 32, 64, 96, 128]
BUFFER_SIZE = 64_000_000


def measure(fn, runs=3):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def memory_worker(src_name, dst_name, start, end):
    src = shared_memory.SharedMemory(name=src_name)
    dst = shared_memory.SharedMemory(name=dst_name)

    try:
        dst.buf[start:end] = bytes(src.buf[start:end])
    finally:
        src.close()
        dst.close()


def run_bandwidth():
    src = shared_memory.SharedMemory(
        create=True, size=BUFFER_SIZE
    )
    dst = shared_memory.SharedMemory(
        create=True, size=BUFFER_SIZE
    )

    try:
        src.buf[:BUFFER_SIZE] = b'\x07' * BUFFER_SIZE

        def execute():
            processes = []
            chunk = BUFFER_SIZE // P_PHYS

            for p in range(P_PHYS):
                start = p * chunk
                end = (
                    BUFFER_SIZE if p == P_PHYS - 1
                    else (p + 1) * chunk
                )

                proc = mp.Process(
                    target=memory_worker,
                    args=(src.name, dst.name, start, end)
                )
                processes.append(proc)
                proc.start()

            for proc in processes:
                proc.join()
                if proc.exitcode != 0:
                    raise RuntimeError("Memory worker failed")

        elapsed = measure(execute)
        bandwidth = (2 * BUFFER_SIZE) / elapsed / 1e9

        return bandwidth

    finally:
        src.close()
        dst.close()
        src.unlink()
        dst.unlink()


def compute_worker(operations):
    a = [1.001 + i * 0.001 for i in range(16)]
    b = [0.999 + i * 0.001 for i in range(16)]
    c = [0.0] * 16

    loops = operations // 32

    for _ in range(loops):
        for j in range(16):
            c[j] = a[j] * b[j] + c[j]

    return sum(c)


def run_compute():
    operations_per_worker = 60_000_000

    with mp.Pool(P_PHYS) as pool:
        start = time.perf_counter()

        pool.map(
            compute_worker,
            [operations_per_worker] * P_PHYS
        )

        elapsed = time.perf_counter() - start

    total_flops = operations_per_worker * P_PHYS
    return total_flops / elapsed / 1e9


def gemm_worker(args):
    a_name, b_name, c_name, start_row, end_row, variant, bs = args

    shm_a = shared_memory.SharedMemory(name=a_name)
    shm_b = shared_memory.SharedMemory(name=b_name)
    shm_c = shared_memory.SharedMemory(name=c_name)

    a = (ctypes.c_double * (N * N)).from_buffer(shm_a.buf)
    b = (ctypes.c_double * (N * N)).from_buffer(shm_b.buf)
    c = (ctypes.c_double * (N * N)).from_buffer(shm_c.buf)

    if variant == 1:
        for i in range(start_row, end_row):
            for j in range(N):
                value = 0.0
                for k in range(N):
                    value += a[i * N + k] * b[k * N + j]
                c[i * N + j] = value

    elif variant == 2:
        for i in range(start_row, end_row):
            for k in range(N):
                aik = a[i * N + k]
                for j in range(N):
                    c[i * N + j] += aik * b[k * N + j]

    else:
        for ii in range(start_row, end_row, bs):
            for jj in range(0, N, bs):
                for kk in range(0, N, bs):
                    for i in range(ii, min(ii + bs, end_row)):
                        for k in range(kk, min(kk + bs, N)):
                            aik = a[i * N + k]
                            for j in range(jj, min(jj + bs, N)):
                                c[i * N + j] += aik * b[k * N + j]

    del a, b, c
    shm_a.close()
    shm_b.close()
    shm_c.close()


def run_gemm(variant, bs=16):
    size = N * N * 8

    shm_a = shared_memory.SharedMemory(create=True, size=size)
    shm_b = shared_memory.SharedMemory(create=True, size=size)
    shm_c = shared_memory.SharedMemory(create=True, size=size)

    try:
        a = (ctypes.c_double * (N * N)).from_buffer(shm_a.buf)
        b = (ctypes.c_double * (N * N)).from_buffer(shm_b.buf)
        c = (ctypes.c_double * (N * N)).from_buffer(shm_c.buf)

        for i in range(N * N):
            a[i] = (i % 17) * 0.01
            b[i] = (i % 13) * 0.01

        del a, b, c

        def execute():
            shm_c.buf[:] = bytes(size)

            processes = []
            rows_per_worker = N // P_PHYS

            for p in range(P_PHYS):
                start_row = p * rows_per_worker
                end_row = (
                    N if p == P_PHYS - 1
                    else (p + 1) * rows_per_worker
                )

                proc = mp.Process(
                    target=gemm_worker,
                    args=((
                        shm_a.name,
                        shm_b.name,
                        shm_c.name,
                        start_row,
                        end_row,
                        variant,
                        bs
                    ),)
                )
                processes.append(proc)
                proc.start()

            for proc in processes:
                proc.join()
                if proc.exitcode != 0:
                    raise RuntimeError("GEMM worker failed")

        elapsed = measure(execute)
        gflops = (2 * N ** 3) / elapsed / 1e9

        result = list(
            (ctypes.c_double * (N * N)).from_buffer_copy(shm_c.buf)
        )

        return gflops, result

    finally:
        shm_a.close()
        shm_b.close()
        shm_c.close()
        shm_a.unlink()
        shm_b.unlink()
        shm_c.unlink()


def main():
    print("PHASE 3A: ROOFLINE")

    bandwidth = run_bandwidth()
    compute = run_compute()
    ridge = compute / bandwidth

    print(f"Bandwidth: {bandwidth:.3f} GB/s")
    print(f"Compute ceiling: {compute:.3f} GFLOP/s")
    print(f"Ridge point: {ridge:.3f} FLOP/byte")

    print("\nPHASE 3B: GEMM")

    naive_gflops, naive_result = run_gemm(1)
    reordered_gflops, reordered_result = run_gemm(2)

    print(f"Naive: {naive_gflops:.4f} GFLOP/s")
    print(f"Reordered: {reordered_gflops:.4f} GFLOP/s")

    best_bs = None
    best_gflops = -1
    best_result = None

    for bs in SIZES:
        gflops, result = run_gemm(3, bs)

        footprint = 3 * bs * bs * 8
        intensity = bs / 12

        print(
            f"BS={bs} | "
            f"GFLOP/s={gflops:.4f} | "
            f"Tile={footprint} bytes | "
            f"Intensity={intensity:.2f}"
        )

        if gflops > best_gflops:
            best_gflops = gflops
            best_bs = bs
            best_result = result

    error = max(
        abs(x - y)
        for x, y in zip(best_result, reordered_result)
    )

    print(f"\nBest tile size: {best_bs}")
    print(f"Maximum error: {error:.12f}")
    print(f"Verification passed: {error < 1e-9}")

    print("\nVERIFICATION")
    print(
        emit_verify_tag(
            "PHASE3",
            STUDENT_ID,
            naive_gflops,
            best_gflops
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
