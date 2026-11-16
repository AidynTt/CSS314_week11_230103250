
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes

from harness import run_benchmark, emit_verify_tag


def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    acc = arr[slot_idx]

    for i in range(iters):
        acc = (
            acc * 6364136223846793005 + (i | 1)
        ) & 0xFFFFFFFFFFFFFFFF

        arr[slot_idx] = acc

    del arr
    shm.close()


def run_test(p, stride, total_iters=16_000_000):
    shm = shared_memory.SharedMemory(create=True, size=4096)

    try:
        iters = total_iters // p

        def execute():
            processes = []

            for i in range(p):
                process = mp.Process(
                    target=fs_worker,
                    args=(shm.name, i * stride, iters)
                )
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

                if process.exitcode != 0:
                    raise RuntimeError("Worker process failed")

        stats = run_benchmark(execute)
        throughput = total_iters / (stats.median_ms * 1000)

        return stats, throughput

    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    student_id = 230103250
    physical_cores = 8

    phase1_times = {}

    print("PHASE 1: STRIDE SWEEP")

    for stride in [1, 2, 4, 8, 16, 32]:
        stats, throughput = run_test(physical_cores, stride)

        print(
            f"Stride={stride} | "
            f"Median={stats.median_ms:.2f} ms | "
            f"IQR={stats.iqr_ms:.2f} ms | "
            f"Throughput={throughput:.2f} Mops/s"
        )

    print("\nPHASE 1: PROCESS SCALING")

    for stride in [1, 16]:
        print(f"\nStride {stride}")
        baseline = None

        for p in [1, 2, 4, 8]:
            stats, throughput = run_test(p, stride)

            if p == 1:
                baseline = stats.median_ms

            speedup = baseline / stats.median_ms
            efficiency = speedup / p

            if p > 1:
                karp_flatt = (
                    (1 / speedup - 1 / p)
                    / (1 - 1 / p)
                )
            else:
                karp_flatt = 0.0

            if p == physical_cores:
                phase1_times[stride] = stats.median_ms

            print(
                f"P={p} | "
                f"Time={stats.median_ms:.2f} ms | "
                f"Speedup={speedup:.2f} | "
                f"Efficiency={efficiency:.2f} | "
                f"Karp-Flatt={karp_flatt:.4f}"
            )

    print("\nVERIFICATION")

    print(
        emit_verify_tag(
            "PHASE1",
            student_id,
            phase1_times[1],
            phase1_times[16]
        )
    )
