import multiprocessing as mp
import math
import time

from harness import emit_verify_tag


STUDENT_ID = 230103250
P_LOG = mp.cpu_count()

STRONG_N = 1_200_000
WEAK_N0 = 200_000


def stage_a(n):
    start = time.perf_counter()

    result = 0.0
    count = int(n * 0.05)

    for i in range(count):
        x = (i + 1) * 0.0001

        for _ in range(10):
            x = math.sin(x) + math.sqrt(abs(x) + 1.0)

        result += x

    return time.perf_counter() - start


def stage_b_worker(args):
    start_idx, end_idx = args

    result = 0.0

    for i in range(start_idx, end_idx):
        x = (i + 1) * 0.00001

        for _ in range(15):
            x = math.sin(x) + math.sqrt(abs(x) + 1.0)
            x = x * 1.000001 + 0.000001

        result += x

    return result


def run_pipeline(pool, p, n):
    chunk = n // p

    tasks = []

    for i in range(p):
        start_idx = i * chunk

        if i == p - 1:
            end_idx = n
        else:
            end_idx = (i + 1) * chunk

        tasks.append((start_idx, end_idx))

    start = time.perf_counter()

    stage_a_time = stage_a(n)

    pool.map(stage_b_worker, tasks)

    total_time = time.perf_counter() - start

    return total_time, stage_a_time


def main():
    print("PHASE 2: STRONG SCALING")

    strong_results = {}
    strong_stage_a = {}

    for p in range(1, P_LOG + 1):
        with mp.Pool(processes=p) as pool:
            total, serial = run_pipeline(pool, p, STRONG_N)

        strong_results[p] = total
        strong_stage_a[p] = serial

        speedup = strong_results[1] / total
        efficiency = speedup / p

        print(
            f"P={p} | "
            f"Time={total:.2f}s | "
            f"Speedup={speedup:.2f} | "
            f"Efficiency={efficiency:.2f}"
        )

    serial_fraction = (
        strong_stage_a[1] / strong_results[1]
    )

    amdahl_limit = 1 / serial_fraction

    print(f"\nSerial fraction: {serial_fraction:.4f}")
    print(f"Amdahl limit: {amdahl_limit:.2f}")

    print("\nPHASE 2: WEAK SCALING")

    weak_results = {}
    gustafson_results = {}

    for p in range(1, P_LOG + 1):
        n = p * WEAK_N0

        with mp.Pool(processes=p) as pool:
            total, serial = run_pipeline(pool, p, n)

        weak_results[p] = total

        weak_efficiency = weak_results[1] / total

        alpha = serial / total

        gustafson = p - alpha * (p - 1)

        gustafson_results[p] = gustafson

        print(
            f"P={p} | "
            f"N={n} | "
            f"Time={total:.2f}s | "
            f"Weak Efficiency={weak_efficiency:.2f} | "
            f"Gustafson={gustafson:.2f}"
        )

    print("\nARCHITECTURAL ANALYSIS")

    if P_LOG > 4:
        speedup_4 = (
            strong_results[1] / strong_results[4]
        )

        speedup_5 = (
            strong_results[1] / strong_results[5]
        )

        print(
            f"Speedup change P=4 to P=5: "
            f"{speedup_5 - speedup_4:.4f}"
        )

    strong_speedup = (
        strong_results[1] / strong_results[P_LOG]
    )

    gustafson_speedup = gustafson_results[P_LOG]

    print("\nVERIFICATION")

    print(
        emit_verify_tag(
            "PHASE2",
            STUDENT_ID,
            strong_speedup,
            gustafson_speedup
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
