import torch

def bytes_to_gb(bytes):
    """ バイトをギガバイトに変換 """
    return bytes / 1024**3

def display_gpu(display="gpu使用料を表示"):
    print("===============[", display, "]===============")
    # GPUメモリの割り当てられた量をGB単位で取得
    memory_allocated_gb = bytes_to_gb(torch.cuda.memory_allocated())
    print(f"Memory allocated: {memory_allocated_gb:.2f} GB")

    """# GPUメモリの予約された量をGB単位で取得
    memory_reserved_gb = bytes_to_gb(torch.cuda.memory_reserved())
    print(f"Memory reserved: {memory_reserved_gb:.2f} GB")"""
