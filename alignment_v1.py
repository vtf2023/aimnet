import numpy as np
from Bio.PDB import PDBParser, PDBIO, Vector, Matrix

def get_shared_ca_coords(struct_old, struct_new):
    """提取新舊結構中對應的 Alpha-carbon (CA) 座標"""
    coords_old = []
    coords_new = []
    
    # 建立舊結構的 CA 字典 {(chain, res_id): coord}
    old_ca = {}
    for model in struct_old:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " " and "CA" in residue: # 標準氨基酸且有CA
                    old_ca[(chain.id, residue.id)] = residue["CA"].get_coord()
                    
    # 比對新結構，找出共同存在的 CA
    for model in struct_new:
        for chain in model:
            for residue in chain:
                key = (chain.id, residue.id)
                if key in old_ca and "CA" in residue:
                    coords_old.append(old_ca[key])
                    coords_new.append(residue["CA"].get_coord())
                    
    return np.array(coords_old), np.array(coords_new)

def align_ligand_to_pqr(old_pdb_path, pqr_path, ligand_with_h_path, output_ligand_path):
    parser = PDBParser(QUIET=True)
    
    # 載入結構
    struct_old = parser.get_structure("old", old_pdb_path)
    struct_new = parser.get_structure("pqr", pqr_path)
    struct_ligand = parser.get_structure("ligand", ligand_with_h_path)
    
    # 1. 取得新舊蛋白質的對應 CA 座標
    ca_old, ca_new = get_shared_ca_coords(struct_old, struct_new)
    print(f"找到 {len(ca_old)} 個對應的共用 CA 原子作為對齊基準。")
    
    # 2. 計算 Kabsch 演算法所需的質心
    centroid_old = np.mean(ca_old, axis=0)
    centroid_new = np.mean(ca_new, axis=0)
    
    ca_old_centered = ca_old - centroid_old
    ca_new_centered = ca_new - centroid_new
    
    # 3. 計算旋轉矩陣 (H = old^T * new)
    H = np.dot(ca_old_centered.T, ca_new_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    # 修正反射矩陣問題
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
        
    # 4. 將旋轉與平移套用到配體的所有原子上
    # Biopython 的 Vector 和 Matrix 需要特定的格式
    biopython_R = Matrix.RawMatrixForm(R)
    
    print("正在將配體座標旋轉並平移至 PQR 空間...")
    for model in struct_ligand:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coord = atom.get_coord()
                    # 先減去舊質心 -> 旋轉 -> 加上新質心
                    coord_centered = coord - centroid_old
                    coord_rotated = np.dot(coord_centered, R.T) # 矩陣乘法
                    coord_final = coord_rotated + centroid_new
                    
                    # 更新配體原子的新座標
                    atom.set_coord(coord_final)
                    
    # 5. 儲存對齊後的配體
    io = PDBIO()
    io.set_structure(struct_ligand)
    io.save(output_ligand_path)
    print(f"🎉 對齊完成！新配體已儲存至: {output_ligand_path}")

# --- 執行對齊 ---
align_ligand_to_pqr(
    old_pdb_path="4J0P.pdb",                # 原始下載的 PDB
    pqr_path="4J0P_H74.pqr",          # PDB2PQR 產生的 PQR
    ligand_with_h_path="ligand_1H8_H_2.pdb", # 你單獨加完氫的配體
    output_ligand_path="ligand_aligned.pdb" # 最終對齊好、可在 PQR 空間使用的配體
)