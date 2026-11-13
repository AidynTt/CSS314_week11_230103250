import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import csv

from harness import run_benchmark


def fs_worker(shm_name, slot_idx, iters):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005

    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc

    del arr
    shm.close()


def test_stride(p, stride, M):
    shm = shared_memory.SharedMemory(
        create=True, size=4096
    )

    try:
        processes = []

        for t in range(p):
            process = mp.Process(
                target=fs_worker,
                args=(shm.name, t * stride, M // p)
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError("Worker failed")

    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    p = 8

    # Quick test
    M = 800_000

    # Full lab: M = 24_000_000
    strides = [1, 2, 4, 8, 16, 32]

    with open("phase1_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "stride", "median_ms",
            "iqr_ms", "throughput_mops"
        ])

        for stride in strides:
            print(f"Testing stride = {stride}...", flush=True)

            stats = run_benchmark(
                lambda: test_stride(p, stride, M),
                warmup_runs=2,
                timed_runs=7
            )

            throughput = M / (stats.median_ms * 1000)

            writer.writerow([
                stride,
                stats.median_ms,
                stats.iqr_ms,
                throughput
            ])

            f.flush()

            print(
                f"Stride {stride}: "
                f"{stats.median_ms:.2f} ms"
            )

    print("All tests completed!")
