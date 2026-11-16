import multiprocessing as mp
import math
import time

from harness import emit_verify_tag

STUDENT_ID = 230103250
P_PHYS = 8
N = 8000
CHUNK_SIZES = [1, 8, 64, 512]


def heavy_work(i):
    k = 1 + (i // 40) ** 2
    result = 0.0

    for j in range(k):
        x = (i + j + 1) * 0.00001
        result += (
            math.sin(x)
            + math.cos(x)
            + math.sqrt(x + 1.0)
        )

    return result


def static_worker(worker_id, p, strategy, runtimes):
    start = time.perf_counter()

    result = 0.0

    if strategy == "block":
        block_size = (N + p - 1) // p
        begin = worker_id * block_size
        end = min(begin + block_size, N)

        for i in range(begin, end):
            result += heavy_work(i)

    elif strategy == "cyclic":
        for i in range(worker_id, N, p):
            result += heavy_work(i)

    runtimes[worker_id] = time.perf_counter() - start


def dynamic_worker(worker_id, counter, chunk_size, runtimes):
    start = time.perf_counter()
    result = 0.0

    while True:
        with counter.get_lock():
            begin = counter.value
            counter.value += chunk_size

        if begin >= N:
            break

        end = min(begin + chunk_size, N)

        for i in range(begin, end):
            result += heavy_work(i)

    runtimes[worker_id] = time.perf_counter() - start


def run_test(strategy, chunk_size=None):
    runtimes = mp.Array("d", P_PHYS)
    counter = mp.Value("i", 0)

    processes = []

    start = time.perf_counter()

    for worker_id in range(P_PHYS):
        if strategy == "dynamic":
            proc = mp.Process(
                target=dynamic_worker,
                args=(
                    worker_id,
                    counter,
                    chunk_size,
                    runtimes
                )
            )
        else:
            proc = mp.Process(
                target=static_worker,
                args=(
                    worker_id,
                    P_PHYS,
                    strategy,
                    runtimes
                )
            )

        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()

        if proc.exitcode != 0:
            raise RuntimeError("Worker process failed")

    wall_time = time.perf_counter() - start

    worker_times = list(runtimes)

    average_time = sum(worker_times) / P_PHYS

    imbalance = (
        max(worker_times) / average_time
        if average_time > 0 else 0
    )

    return wall_time, imbalance


def main():
    print("PHASE 4: DYNAMIC LOAD BALANCING")

    print("\nSTATIC BLOCK")
    block_time, block_lambda = run_test("block")

    print(f"Wall Time: {block_time:.3f}s")
    print(f"Load Imbalance: {block_lambda:.3f}")

    print("\nSTATIC CYCLIC")
    cyclic_time, cyclic_lambda = run_test("cyclic")

    print(f"Wall Time: {cyclic_time:.3f}s")
    print(f"Load Imbalance: {cyclic_lambda:.3f}")

    best_chunk = None
    best_time = float("inf")
    best_lambda = None

    print("\nDYNAMIC QUEUE")

    for chunk_size in CHUNK_SIZES:
        wall_time, imbalance = run_test(
            "dynamic", chunk_size
        )

        print(
            f"Chunk={chunk_size} | "
            f"Wall Time={wall_time:.3f}s | "
            f"Load Imbalance={imbalance:.3f}"
        )

        if wall_time < best_time:
            best_time = wall_time
            best_chunk = chunk_size
            best_lambda = imbalance

    print("\nRESULTS")

    print(f"Best Chunk Size: {best_chunk}")
    print(f"Best Dynamic Time: {best_time:.3f}s")
    print(f"Best Dynamic Imbalance: {best_lambda:.3f}")

    print("\nVERIFICATION")

    print(
        emit_verify_tag(
            "PHASE4",
            STUDENT_ID,
            block_time,
            best_time
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
