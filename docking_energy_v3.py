import torch
import numpy as np
from rdkit import Chem
from Bio.PDB import PDBParser
from aimnet.calculators import AIMNet2Calculator

# ==========================================
# 配置區域
# ==========================================
# 請修改以下檔名為您的實際檔案
PROTEIN_PDB_FILE = "4J0P_H70_2.pdb"
LIGAND_MOL2_FILE = "ligand_1H8_H_3.mol2"

# 設定總淨電荷 (Net Charge)
# AIMNet2 需要知道整個系統的總電荷。
# 如果配體在 pH 7.0 下帶電，請在此修改。
PROTEIN_NET_CHARGE = 0.0  # 蛋白質總電荷 (通常假設中性，除非您確定)
LIGAND_NET_CHARGE  = 0.0  # 配體總電荷 (中性設0, 若去質子化羧酸設-1, 質子化胺設+1)

# ==========================================
# 工具函式
# ==========================================
periodic_table = {
    'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'P': 15, 'S': 16, 'CL': 17, 'BR': 35, 'I': 53, 'PD': 46
}

def rdkit_mol_to_aimnet_input(mol, net_charge):
    """將 RDKit Molecule 物件轉換為 AIMNet2 格式"""
    numbers = []
    coords = []
    
    # 確保 RDKit 有讀到 3D 座標
    if mol.GetNumConformers() == 0:
        raise ValueError("Ligand MOL2 file does not contain 3D coordinates!")
        
    conf = mol.GetConformer()
    
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol().upper()
        if symbol in periodic_table:
            numbers.append(periodic_table[symbol])
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])
        else:
            print(f"Warning: Element {symbol} not supported by this script.")

    return {
        "numbers": torch.tensor(numbers, dtype=torch.long),
        "coord": torch.tensor(coords, dtype=torch.float32),
        "charge": torch.tensor([net_charge], dtype=torch.float32)
    }

def biopython_pdb_to_aimnet_input(structure, net_charge):
    """將 Biopython PDB Structure 物件轉換為 AIMNet2 格式"""
    numbers = []
    coords = []
    
    for atom in structure.get_atoms():
        # 取得元素符號 (Biopython有時會包含數字，需要清理)
        element = ''.join([i for i in atom.element if not i.isdigit()]).upper().strip()
        
        if element in periodic_table:
            numbers.append(periodic_table[element])
            coords.append(atom.get_coord())
            
    return {
        "numbers": torch.tensor(numbers, dtype=torch.long),
        "coord": torch.tensor(coords, dtype=torch.float32),
        "charge": torch.tensor([net_charge], dtype=torch.float32)
    }

def merge_aimnet_inputs(input1, input2, new_net_charge):
    """合併兩個 AIMNet2 輸入 Tensor"""
    return {
        "numbers": torch.cat([input1["numbers"], input2["numbers"]]),
        "coord": torch.cat([input1["coord"], input2["coord"]]),
        "charge": torch.tensor([new_net_charge], dtype=torch.float32)
    }

# ==========================================
# 主程式流程
# ==========================================
def main():
    print("--- AIMNet2 結合能評估腳本 ---")
    
    # 1. 初始化 AIMNet2 計算器 (使用 GPU 若可用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    calc = AIMNet2Calculator("aimnet2") # 載入一般有機分子模型
    
    # 2. 讀取配體 (MOL2) - 使用 RDKit
    print(f"正在讀取配體: {LIGAND_MOL2_FILE}...")
    ligmol = Chem.MolFromMol2File(LIGAND_MOL2_FILE, removeHs=False) # 關鍵: removeHs=False 保留氫
    if ligmol is None:
        print("錯誤: 無法讀取 MOL2 檔案。請檢查格式。")
        return
        
    input_ligand = rdkit_mol_to_aimnet_input(ligmol, LIGAND_NET_CHARGE)
    input_ligand = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in input_ligand.items()}

    # 3. 讀取蛋白質 (PDB) - 使用 Biopython
    print(f"正在讀取蛋白質: {PROTEIN_PDB_FILE}...")
    parser = PDBParser(QUIET=True)
    struct_prot = parser.get_structure("protein", PROTEIN_PDB_FILE)
    
    input_protein = biopython_pdb_to_aimnet_input(struct_prot, PROTEIN_NET_CHARGE)
    input_protein = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in input_protein.items()}

    # 4. 建立複合物 (Complex)
    total_complex_charge = PROTEIN_NET_CHARGE + LIGAND_NET_CHARGE
    input_complex = merge_aimnet_inputs(input_protein, input_ligand, total_complex_charge)
    input_complex = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in input_complex.items()}

    print(f"結構讀取完成。Protein原子數: {len(input_protein['numbers'])}, Ligand原子數: {len(input_ligand['numbers'])}")

    # 5. 使用 AIMNet2 計算能量 (eV)
    print("正在呼叫 AIMNet2 進行能量計算...")
    with torch.no_grad():
        # 計算單點能
        e_complex = calc(input_complex)["energy"].item()
        e_protein = calc(input_protein)["energy"].item()
        e_ligand  = calc(input_ligand)["energy"].item()

    # 6. 計算交互作用結合能 (Interaction/Binding Energy)
    # E_binding = E_complex - (E_protein + E_ligand)
    e_binding_ev = e_complex - (e_protein + e_ligand)
    
    # 單位換算: eV -> kcal/mol
    EV_TO_KCALMOL = 23.0605
    e_binding_kcal = e_binding_ev * EV_TO_KCALMOL

    # 7. 輸出結果
    print("\n" + "="*40)
    print("       AIMNet2 計算結果 (eV)")
    print("="*40)
    print(f"Complex 總能量:  {e_complex:15.4f} eV")
    print(f"Protein 總能量:  {e_protein:15.4f} eV")
    print(f"Ligand  總能量:  {e_ligand:15.4f} eV")
    print("-" * 40)
    print(f"結合交互作用能:  {e_binding_ev:15.4f} eV")
    print(f"結合交互作用能:  {e_binding_kcal:15.2f} kcal/mol")
    print("="*40)
    print("\n* 註：負值越低代表結合越穩定。")
    print("* 註：計算基於您提供的固定幾何結構，未進行結構優化。")

if __name__ == "__main__":
    main()