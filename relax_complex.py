import torch
import numpy as np
from Bio.PDB import PDBParser, PDBIO
from aimnet.calculators import AIMNet2Calculator

# ==========================================
# 配置區域
# ==========================================
INPUT_PDB = "4J0P_cleaned_with_H.pdb"  # 含有立體衝突的 PDB 檔案
OUTPUT_PDB = "4J0P_relaxed.pdb"      # 輸出能量最小化後的 PDB 檔案
TOTAL_CHARGE = 0.0                     # 整個系統的總電荷 (蛋白質+配體)

# 優化參數
MINIMIZATION_STEPS = 200
LEARNING_RATE = 0.01

# ==========================================
# 工具函式
# ==========================================
periodic_table = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'CL': 17, 'BR': 35, 'I': 53}

def get_aimnet_inputs_from_biopython(atoms, device="cpu"):
    """將 Biopython 的 Atom List 轉換為 AIMNet2 所需的 Tensor 格式"""
    if not atoms:
        return None
    
    numbers = []
    coords = []
    
    for atom in atoms:
        # 清理元素符號，移除數字和空格
        elem = ''.join([i for i in atom.element if not i.isdigit()]).upper().strip()
        if elem in periodic_table:
            numbers.append(periodic_table[elem])
            coords.append(atom.get_coord())
        else:
            print(f"警告：腳本不支援此原子 {elem}，將會跳過。")
            
    return {
        "numbers": torch.tensor(numbers, dtype=torch.long, device=device),
        "coord": torch.tensor(np.array(coords), dtype=torch.float32, device=device),
        "charge": torch.tensor([TOTAL_CHARGE], dtype=torch.float32, device=device)
    }

def minimize_complex(structure, calc, device, steps, lr):
    """
    使用 AIMNet2 和 PyTorch optimizer 對整個複合物結構進行能量最小化。
    """
    print(f"\n--- 開始對整個複合物進行能量最小化 ---")
    
    # 1. 收集所有原子
    all_atoms = list(structure.get_atoms())
    print(f"總原子數: {len(all_atoms)}")
    
    # 2. 準備 AIMNet2 輸入
    input_data = get_aimnet_inputs_from_biopython(all_atoms, device)
    
    # 3. 設置優化器
    coords = input_data['coord'].clone().detach().requires_grad_(True)
    numbers = input_data['numbers']
    charge = input_data['charge']
    optimizer = torch.optim.Adam([coords], lr=lr)

    # 4. 計算初始能量
    initial_energy = calc({"numbers": numbers, "coord": coords, "charge": charge})["energy"].item()
    print(f"初始能量: {initial_energy:.4f} eV")

    # 5. 執行最小化迴圈
    for step in range(steps):
        optimizer.zero_grad()
        results = calc({"numbers": numbers, "coord": coords, "charge": charge}, forces=True)
        energy = results["energy"]
        energy.backward()
        optimizer.step()
        
        if (step + 1) % 20 == 0:
            print(f"步驟 {step+1}/{steps}, 當前能量: {energy.item():.4f} eV")
            
    final_energy = energy.item()
    print(f"最小化完成。最終能量: {final_energy:.4f} eV")
    
    # 6. 更新 Biopython structure 物件中的原子座標
    minimized_coords_np = coords.detach().cpu().numpy()
    for i, atom in enumerate(all_atoms):
        atom.set_coord(minimized_coords_np[i])
        
    return structure

def main():
    print("--- 結構弛豫 (Relaxation) 腳本 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    calc = AIMNet2Calculator("aimnet2", device=device)
    
    print(f"正在讀取 PDB 檔案: {INPUT_PDB}...")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", INPUT_PDB)

    relaxed_structure = minimize_complex(structure, calc, device, steps=MINIMIZATION_STEPS, lr=LEARNING_RATE)

    print(f"\n正在儲存弛豫後的結構至: {OUTPUT_PDB}")
    io = PDBIO()
    io.set_structure(relaxed_structure)
    io.save(OUTPUT_PDB)
    
    print("\n✅ 處理完成！")
    print(f"您現在可以使用 '{OUTPUT_PDB}' 進行後續的結合能計算。")

if __name__ == "__main__":
    main()