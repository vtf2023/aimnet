import torch
from aimnet.calculators import AIMNet2Calculator

# 1. 決定計算裝置 (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"將使用 {device} 進行計算")

# 2. 初始化計算器 (會透過剛剛裝的 [hf] 功能自動從 Hugging Face 下載模型)
#    並將模型移至指定的計算裝置
calc = AIMNet2Calculator("aimnet2", device=device)

# 3. 準備你的分子資料（以水分子 H2O 為例，座標單位：埃 Ångstrom）
# 這裡需要提供：原子序 (numbers)、原子座標 (coord)
#    同時將張量 (tensor) 移至指定的計算裝置
input_data = {
    "numbers": torch.tensor([8, 1, 1], device=device),  # 8是氧(O), 1是氫(H)
    "coord": torch.tensor([
        [0.0000, 0.0000, 0.1194],   # O
        [0.0000, 0.7572, -0.4778],  # H
        [0.0000, -0.7572, -0.4778]  # H
    ], dtype=torch.float32, device=device),
    "charge": 0.0  # 淨電荷
}

# 4. 進行預測，並要求計算受力 (forces)
result = calc(input_data, forces=True)

# 5. 印出預測結果
print("--- 計算結果 ---")
#    從 GPU 將結果移回 CPU 以便使用 .numpy()
print("分子總能量 (eV):", result["energy"].item()) # .item() 會自動處理裝置
print("原子受力 (eV/Å):\n", result["forces"].detach().cpu().numpy())
print("原子淨電荷 (e):\n", result["charges"].detach().cpu().numpy())