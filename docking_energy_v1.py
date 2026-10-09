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
        "coord": torch.tensor(coords, dtype=torch.float32, device=device),
        "charge": 0.0 # 簡化假設淨電荷為 0
    }

# 3. 解析 4J0P.pdb 檔案
parser = PDBParser(QUIET=True)
residue_name = "LIG"
#structure = parser.get_structure("4J0P", "4J0P_H.pdb")
#structure = parser.get_structure("4J0P", "4J0P_cleaned_with_H.pdb")
#structure = parser.get_structure("4J0P", "4J0P_cleaned_Noaddres_with_H_wo_Na_DMS_Water.pdb")
#structure = parser.get_structure("4J0V", "4J0V_cleaned_Noaddres_with_H_wo_Na_DMS_Water.pdb")
#structure = parser.get_structure("4J1H", "4J1H_cleaned_Noaddres_with_H_wo_Na_DMS_Water.pdb")
#structure = parser.get_structure("4J1F", "4J1F_cleaned_Noaddres_with_H_wo_Na_DMS_Water.pdb")
structure = parser.get_structure("2NNQ", "./2nnq_data/complex_2nnq.pdb")

# 分離 Target (BACE-1 蛋白質) 與 Ligand (在 4J0P 中配體殘基名稱為 '1H8')
protein_atoms = []
ligand_atoms = []

for model in structure:
    for chain in model:
        for residue in chain:
            # 判斷是否為配體 (Heteroatom 且非水分子)
            if residue.id[0].startswith("H_") and residue.resname != "HOH":
                if residue.resname == residue_name: # 4J0P 的主要抑制劑配體
                    ligand_atoms.extend(residue.get_atoms())
            elif residue.id[0] == " ": # 標準氨基酸 (蛋白質主體)
                protein_atoms.extend(residue.get_atoms())

# 結合兩者得到 Complex
complex_atoms = protein_atoms + ligand_atoms

print(f"蛋白質原子數: {len(protein_atoms)}, 配體原子數: {len(ligand_atoms)}")

# 4. 將資料轉換為 AIMNet2 格式
input_complex = get_aimnet_inputs(complex_atoms, device=device)
input_protein = get_aimnet_inputs(protein_atoms, device=device)
input_ligand = get_aimnet_inputs(ligand_atoms, device=device)

if not all([input_complex, input_protein, input_ligand]):
    print("錯誤：一個或多個分子群組中沒有原子，無法進行計算。")
    exit()

# 5. 進行能量評估 (單點能計算，單位：eV)
with torch.no_grad():
    energy_complex = calc(input_complex)["energy"].item()
    energy_protein = calc(input_protein)["energy"].item()
    energy_ligand = calc(input_ligand)["energy"].item()

# 6. 計算交互作用能 (Interaction Energy)
# $E_{interaction} = E_{complex} - (E_{protein} + E_{ligand})$
interaction_energy_ev = energy_complex - (energy_protein + energy_ligand)
interaction_energy_kcal = interaction_energy_ev * 23.0605 # 1 eV = 23.0605 kcal/mol

print("\n=== AIMNet2 能量評估結果 ===")
print(f"複合物總能量 (Complex Energy)   : {energy_complex:.4f} eV")
print(f"蛋白質總能量 (Target Energy)    : {energy_protein:.4f} eV")
print(f"配體總能量   (Ligand Energy)    : {energy_ligand:.4f} eV")
print("---------------------------------------------")
print(f"結合交互作用能 (Interaction Energy): {interaction_energy_ev:.4f} eV")
print(f"結合交互作用能 (Interaction Energy): {interaction_energy_kcal:.2f} kcal/mol")