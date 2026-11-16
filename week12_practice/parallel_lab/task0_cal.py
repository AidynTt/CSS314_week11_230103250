import multiprocessing as mp
import time

from harness import emit_verify_tag


def main():
    student_id = 230103250

    cores = mp.cpu_count()
    print(f"Logical CPU cores: {cores}")

    iterations = 1_000_000

    start = time.perf_counter_ns()

    for _ in range(iterations):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    avg_latency_ns = (end - start) / iterations

    print(f"Average latency: {avg_latency_ns:.2f} ns")

    verify_hash = emit_verify_tag(
        "PHASE0",
        student_id,
        float(cores),
        avg_latency_ns
    )

    print(verify_hash)


if __name__ == "__main__":
    main()