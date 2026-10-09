import torch
import numpy as np
from Bio.PDB import PDBParser
from aimnet.calculators import AIMNet2Calculator

# 1. 決定計算裝置並載入 AIMNet2 計算器
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"將使用 {device} 進行計算...")
calc = AIMNet2Calculator("aimnet2", device=device)

# 2. 定義一個輔助函式：將 Bio.PDB 的原子轉換為 AIMNet2 所需的 Tensor 格式
def get_aimnet_inputs(atoms, device="cpu"):
    if not atoms:
        return None

    # 獲取原子序 (Atomic numbers)
    # Bio.PDB 的 element 是字串 (例如 'C', 'O', 'H')
    periodic_table = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'CL': 17, 'BR': 35, 'I': 53}
    
    numbers = []
    coords = []
    
    for atom in atoms:
        elem = atom.element.upper().strip()
        if elem in periodic_table:
            numbers.append(periodic_table[elem])
            coords.append(atom.get_coord())

    return {
        "numbers": torch.tensor(numbers, dtype=torch.long, device=device),
        "coord": torch.tensor(np.array(coords), dtype=torch.float32, device=device),
        "charge": 0.0 # 簡化假設淨電荷為 0
    }

# 新增：能量最小化函式
def minimize_structure(atoms, calc, device, steps=100, lr=0.01):
    """
    使用 AIMNet2 和 PyTorch optimizer 對結構進行能量最小化。
    """
    if not atoms:
        return None, 0.0

    print(f"\n開始對 {len(atoms)} 個原子進行能量最小化...")
    
    # 準備輸入
    input_data = get_aimnet_inputs(atoms, device=device)
    
    # 讓座標可以被優化
    coords = input_data['coord'].clone().detach().requires_grad_(True)
    numbers = input_data['numbers']
    charge = input_data['charge']
    
    # 使用 Adam 優化器
    optimizer = torch.optim.Adam([coords], lr=lr)

    # 最小化前的能量
    initial_energy = calc({"numbers": numbers, "coord": coords, "charge": charge})["energy"].item()
    print(f"初始能量: {initial_energy:.4f} eV")

    for step in range(steps):
        optimizer.zero_grad()
        
        # 計算能量和梯度
        results = calc({"numbers": numbers, "coord": coords, "charge": charge}, forces=True)
        energy = results["energy"]
        
        # 反向傳播以計算梯度
        energy.backward()
        
        # 更新座標
        optimizer.step()
        
        if (step + 1) % 10 == 0:
            print(f"步驟 {step+1}/{steps}, 能量: {energy.item():.4f} eV")
            
    final_energy = energy.item()
    print(f"最小化完成。最終能量: {final_energy:.4f} eV")
    
    return coords.detach(), final_energy

# 3. 解析 4J0P.pdb 檔案
parser = PDBParser(QUIET=True)
structure = parser.get_structure("4J0P", "4J0P_H.pdb")

# 分離 Target (BACE-1 蛋白質) 與 Ligand (在 4J0P 中配體殘基名稱為 '1H8')
protein_atoms = []
ligand_atoms = []

for model in structure:
    for chain in model:
        for residue in chain:
            # 判斷是否為配體 (Heteroatom 且非水分子)
            if residue.id[0].startswith('H_') and residue.resname != 'HOH':
                if residue.resname == "1H8": # 4J0P 的主要抑制劑配體
                    ligand_atoms.extend(residue.get_atoms())
            elif residue.id[0] == " ": # 標準氨基酸 (蛋白質主體)
                protein_atoms.extend(residue.get_atoms())

print(f"蛋白質原子數: {len(protein_atoms)}, 配體原子數: {len(ligand_atoms)}")

# 4. 對蛋白質進行能量最小化
#    注意：這裡假設 PDB 檔案已包含氫原子。
minimized_protein_coords, energy_protein_minimized = minimize_structure(protein_atoms, calc, device, steps=100)

# 5. 準備用於最終計算的資料
#    - 創建一個新的複合物，包含最小化後的蛋白質和原始配體
input_ligand = get_aimnet_inputs(ligand_atoms, device=device)

if minimized_protein_coords is not None and input_ligand is not None:
    protein_numbers = get_aimnet_inputs(protein_atoms, device=device)['numbers']
    new_complex_coords = torch.cat([minimized_protein_coords, input_ligand['coord']], dim=0)
    new_complex_numbers = torch.cat([protein_numbers, input_ligand['numbers']], dim=0)
    input_complex_minimized_protein = {
        "numbers": new_complex_numbers,
        "coord": new_complex_coords,
        "charge": 0.0
    }
else:
    print("錯誤：蛋白質或配體原子為空，無法繼續。")
    exit()

# 6. 進行最終的能量評估
with torch.no_grad():
    # 計算包含最小化蛋白質的新複合物能量
    energy_complex_new = calc(input_complex_minimized_protein)["energy"].item()
    # 配體能量保持不變
    energy_ligand_original = calc(input_ligand)["energy"].item()

# 7. 計算交互作用能
interaction_energy_ev = energy_complex_new - (energy_protein_minimized + energy_ligand_original)
interaction_energy_kcal = interaction_energy_ev * 23.0605 # 1 eV = 23.0605 kcal/mol

print("\n=== AIMNet2 能量評估結果 (使用最小化後的蛋白質) ===")
print(f"新複合物總能量 (New Complex Energy) : {energy_complex_new:.4f} eV")
print(f"最小化蛋白質總能量 (Minimized Protein): {energy_protein_minimized:.4f} eV")
print(f"原始配體總能量   (Original Ligand)  : {energy_ligand_original:.4f} eV")
print("---------------------------------------------")
print(f"結合交互作用能 (Interaction Energy): {interaction_energy_ev:.4f} eV")
print(f"結合交互作用能 (Interaction Energy): {interaction_energy_kcal:.2f} kcal/mol")