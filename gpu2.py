import torch

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()  
    print(f"Number of GPUs: {gpu_count}")
    

    for gpu_id in range(gpu_count):
        print(f"\nDetails for GPU {gpu_id}:")
        
        
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"GPU Name: {gpu_name}")
        
        
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3)
        print(f"Total Memory: {total_memory:.2f} GB")
        
        
        props = torch.cuda.get_device_properties(gpu_id)
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Number of CUDA Cores: {props.multi_processor_count * 128}")  # Assuming 128 cores per SM
        print(f"Clock Rate: {props.clock_rate / 1000} MHz")
        print(f"Memory Bandwidth: {props.memory_clock_rate * 2 * props.memory_bus_width / 8 / 1e6} GB/s")
else:
    print("CUDA (GPU) is not available.")
