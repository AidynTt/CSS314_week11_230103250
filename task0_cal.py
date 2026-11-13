from harness import emit_verify_tag
import multiprocessing as mp
import time

N = 2_000_000

start = time.perf_counter_ns()

for i in range(N):
    time.perf_counter_ns()

end = time.perf_counter_ns()

average_ns = (end - start) / N

print("CPU count:", mp.cpu_count())
print("Average timer latency:", average_ns, "ns")


student_id = int(input("Enter your Student ID: "))

print(emit_verify_tag(
    "PHASE0",
    student_id,
    average_ns,
    float(mp.cpu_count())
))