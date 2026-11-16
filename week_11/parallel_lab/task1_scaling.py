
import csv
from harness import run_benchmark
from task1_falsesharing import test_stride

M = 24_000_000
process_counts = [1, 2, 4, 8]
strides = [1, 16]

if __name__ == "__main__":
    with open("phase1_scaling.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "stride", "p", "median_ms",
            "speedup", "efficiency", "karp_flatt"
        ])

        for stride in strides:
            baseline = None

            for p in process_counts:
                print(f"Testing stride={stride}, p={p}...", flush=True)

                stats = run_benchmark(
                    lambda: test_stride(p, stride, M),
                    warmup_runs=2,
                    timed_runs=7
                )

                t = stats.median_ms

                if p == 1:
                    baseline = t

                speedup = baseline / t
                efficiency = speedup / p

                karp = (
                    (1 / speedup - 1 / p) / (1 - 1 / p)
                    if p > 1 else None
                )

                writer.writerow([
                    stride, p, t,
                    speedup, efficiency, karp
                ])
                f.flush()

                print(f"Time: {t:.2f} ms, Speedup: {speedup:.2f}")
