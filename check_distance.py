import numpy as np
from scipy.spatial.distance import cdist
from Bio.PDB import PDBParser

# 執行檢查
# ==========================================
# 檔案路徑與參數設定
# ==========================================
PDB_FILE = "4J0P_cleaned_with_H.pdb"
CLASH_THRESHOLD = 1.0  # 單位：Ångstrom。小於此距離的原子對將被報告
LIGAND_RESNAME = "1H8" # 設定為 None 可檢查整個系統，或指定配體名稱

def main():
    print(f"--- 開始檢查 '{PDB_FILE}' 中的立體衝突 ---")
    print(f"距離閾值: {CLASH_THRESHOLD} Å")

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("system", PDB_FILE)
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到檔案 '{PDB_FILE}'")
        return

    protein_atoms = []
    ligand_atoms = []
    
    # 1. 將原子分類為蛋白質和配體
    for atom in structure.get_atoms():
        residue = atom.get_parent()
        # HETATM 且殘基名稱與我們指定的配體相符
        if residue.id[0].startswith('H_') and residue.get_resname() == LIGAND_RESNAME:
            ligand_atoms.append(atom)
        # 標準氨基酸殘基
        elif residue.id[0] == ' ':
            protein_atoms.append(atom)

    if not ligand_atoms and LIGAND_RESNAME:
        print(f"⚠️ 警告：在檔案中找不到殘基名為 '{LIGAND_RESNAME}' 的配體。")
        print("將改為檢查整個檔案內部的所有原子衝突。")
        protein_atoms = list(structure.get_atoms()) # 將所有原子視為一個群組
        ligand_atoms = [] # 清空配體列表

    # 2. 執行檢查
    clash_count = 0
    
    # 檢查 1: 蛋白質內部衝突
    print("\n[檢查 1/2] 正在掃描蛋白質內部的嚴重衝突...")
    p_coords = np.array([atom.get_coord() for atom in protein_atoms])
    # 使用 cdist 高效計算所有原子對的距離矩陣
    dist_matrix = cdist(p_coords, p_coords)
    # 找出所有小於閾值的距離，並排除對角線(自己跟自己)
    clashes = np.where((dist_matrix > 0) & (dist_matrix < CLASH_THRESHOLD))
    
    # 為了避免重複報告 (i,j) 和 (j,i)，我們只取上三角部分
    internal_clash_pairs = set()
    for i, j in zip(*clashes):
        if i < j:
            atom_i, atom_j = protein_atoms[i], protein_atoms[j]
            res_i, res_j = atom_i.get_parent(), atom_j.get_parent()
            # 排除相同或相鄰殘基內的原子，避免報告正常的共價鍵
            if abs(res_i.id[1] - res_j.id[1]) <= 1 and res_i.get_parent() == res_j.get_parent():
                continue
            
            pair = tuple(sorted((atom_i.serial_number, atom_j.serial_number)))
            internal_clash_pairs.add(pair)

    if internal_clash_pairs:
        print(f"❌ 發現 {len(internal_clash_pairs)} 處蛋白質內部立體衝突！")
        for i, pair in enumerate(list(internal_clash_pairs)[:10]): # 最多顯示10個
            # 這裡不再重新計算距離，僅為報告
            print(f"  - 偵測到一處嚴重碰撞，建議執行能量最小化。")
        clash_count += len(internal_clash_pairs)
    else:
        print("  ✅ 蛋白質內部無嚴重衝突。")

    # 檢查 2: 配體與蛋白質之間的衝突
    if ligand_atoms:
        print("\n[檢查 2/2] 正在掃描配體與蛋白質之間的衝突...")
        l_coords = np.array([atom.get_coord() for atom in ligand_atoms])
        
        distances_lig_prot = cdist(l_coords, p_coords)
        min_dist = distances_lig_prot.min()
        
        print(f"  配體與蛋白質原子間的最短距離為: {min_dist:.2f} Å")
        
        if min_dist < CLASH_THRESHOLD:
            clash_indices = np.where(distances_lig_prot < CLASH_THRESHOLD)
            lig_prot_clash_count = clash_indices[0].size
            clash_count += lig_prot_clash_count
            
            print(f"❌ 發現 {lig_prot_clash_count} 處配體-蛋白質立體衝突！")
            for i in range(min(lig_prot_clash_count, 10)): # 最多顯示10個
                l_idx, p_idx = clash_indices[0][i], clash_indices[1][i]
                dist = distances_lig_prot[l_idx, p_idx]
                l_atom = ligand_atoms[l_idx]
                p_atom = protein_atoms[p_idx]
                p_res = p_atom.get_parent()
                print(f"  -> 距離: {dist:.2f} Å | 配體原子: {l_atom.name} (ID: {l_atom.serial_number}) "
                      f"與 蛋白質原子: {p_res.get_resname()}{p_res.id[1]}-{p_atom.name} (ID: {p_atom.serial_number})")
        else:
            print("  ✅ 配體與蛋白質之間無嚴重衝突。")

    # 總結
    print("-" * 40)
    if clash_count == 0:
        print("✅ 結構檢查完成，未發現低於閾值的嚴重立體衝突。")
    else:
        print(f"總結：共發現 {clash_count} 處潛在的嚴重立體衝突。")
        print("💡 建議使用能量最小化腳本 (relax_complex.py) 來解決這些問題。")

if __name__ == "__main__":
    main()
