import torch

num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    print("No GPUs detected!")
else:
    print(f"Detected {num_gpus} GPUs\n")
    used_count = 0
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**2
        reserved = torch.cuda.memory_reserved(i) / 1024**2
        print(f"GPU {i}: {name}")
        print(f"  Allocated memory: {allocated:.1f} MB")
        print(f"  Reserved memory:  {reserved:.1f} MB")
        # 判断是否正在使用
        if allocated > 0 or reserved > 0:
            used_count += 1
    print("\nSummary:")
    print(f"  GPUs currently used by processes: {used_count} / {num_gpus}")
