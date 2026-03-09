import torch
import sys
# from torch.profiler import profile, ProfilerActivity
from sgl_kernel import fp8_scaled_mm


def flush_gpu_cache(size_mb=256):
    x = torch.empty(size_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    x += 1
    torch.cuda.synchronize()


# def measure_with_profiler(fn, *args, **kwargs):
#     # Warm-up
#     for _ in range(5):
#         fn(*args, **kwargs)
#     torch.cuda.synchronize()

#     # Cold-cache run
#     flush_gpu_cache()

#     # High-resolution CUPTI timing
#     with profile(activities=[ProfilerActivity.CUDA]) as prof:
#         fn(*args, **kwargs)
#         torch.cuda.synchronize()

#     # Extract CUDA kernel events
#     cuda_events = [evt for evt in prof.events() if evt.device_type == torch.profiler.ProfilerActivity.CUDA]
#     print(len(cuda_events))

#     # Print all CUDA events
#     for evt in prof.events():
#         if evt.device_type == ProfilerActivity.CUDA:
#             print(f"Kernel: {evt.name}")
#             print(f"  Time (us): {evt.cuda_time:.2f}")
#             print(f"  Start (ns): {evt.start_time}")
#             print(f"  Duration (ns): {evt.duration}")
#             print(f"  Stream: {evt.stream}")
#             print(f"  Correlation ID: {evt.correlation_id}")
#             print("-" * 60)


data = torch.load(sys.argv[1])

a = data['mat_a'].cuda()
b = data['mat_b'].cuda()
sa = data['scales_a'].cuda()
sb = data['scales_b'].cuda()
o = data['out_dtype']
bias = data['bias']

# warm up
fp8_scaled_mm(a, b, sa, sb, o, bias)
fp8_scaled_mm(a, b, sa, sb, o, bias)
fp8_scaled_mm(a, b, sa, sb, o, bias)
fp8_scaled_mm(a, b, sa, sb, o, bias)
fp8_scaled_mm(a, b, sa, sb, o, bias)
flush_gpu_cache()
fp8_scaled_mm(a, b, sa, sb, o, bias)
