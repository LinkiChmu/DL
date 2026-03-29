import torch

print(f"Версия PyTorch: {torch.__version__}")

# Проверка для процессоров Apple Silicon (M1/M2/M3)
if torch.backends.mps.is_available():
    print("✅ MPS доступно (ускорение на GPU Apple)")
    device = torch.device("mps")
elif torch.cuda.is_available():
    print("✅ CUDA доступно (редко для Mac)")
    device = torch.device("cuda")
else:
    print("⚠️ Только CPU")
    device = torch.device("cpu")
