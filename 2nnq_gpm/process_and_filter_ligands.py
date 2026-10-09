import os
import sys
import argparse
import pandas as pd

# 為了能從父目錄導入 'check_distance_2'
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 導入需要的函式
from merge_pdbs import merge_structures
from check_distance_3 import check_steric_clashes

# --- 組態設定 ---
# 基礎資料夾路徑
BASE_DIR = '/home/kuoweihuang/aimnet/2nnq_gpm'

# 輸入檔案與資料夾
TOP_LIGANDS_FILE = os.path.join(BASE_DIR, 'top_500_ligands.txt')
PROTEIN_PDB_PATH = os.path.join(BASE_DIR, 'processed_2nnq_with_H.pdb')
DECOYS_DIR = os.path.join(BASE_DIR, 'workdir_gpm_decoys_dxdydz20')
ACTIVES_DIR = os.path.join(BASE_DIR, 'workdir_gpm_actives_dxdydz20')

# 輸出檔案與資料夾
COMPLEXES_DIR = os.path.join(BASE_DIR, 'complexes')
FILTERED_LIGANDS_FILE = os.path.join(BASE_DIR, 'top_ligands_no_clash.txt')

def get_ligand_paths(top_ligands_file, num_ligands):
    """
    讀取 top_ligands.txt 檔案，並建構出對應 .mol2 檔案的完整路徑。
    """
    print(f"正在從 {os.path.basename(top_ligands_file)} 讀取前 {num_ligands} 個配體...")
    
    # 使用 pandas 讀取以空白分隔的檔案
    df = pd.read_csv(top_ligands_file, delim_whitespace=True, header=0)
    
    ligand_info_list = []
    
    for i, row in df.head(num_ligands).iterrows():
        energy_file = row['index']
        label = row['label']
        
        # 將 'ligand_XXXX_match_energies.txt' 轉換為 'ligand_XXXX.mol2'
        ligand_name = energy_file.replace('_match_energies.txt', '.mol2')
        
        if label == 0:
            # Decoy 配體
            mol2_path = os.path.join(DECOYS_DIR, ligand_name)
        elif label == 1:
            # Active 配體
            mol2_path = os.path.join(ACTIVES_DIR, ligand_name)
        else:
            print(f"警告：發現未知標籤 '{label}' 於 {energy_file}，將跳過。")
            continue
            
        if not os.path.exists(mol2_path):
            print(f"警告：找不到配體檔案，將跳過：{mol2_path}")
            continue
            
        # 儲存原始行內容與完整路徑
        original_line = f"{row['index']:<35} {row['energy']:<10.3f} {row['label']}"
        ligand_info_list.append({
            "path": mol2_path,
            "original_line": original_line,
            "ligand_name": ligand_name
        })
        
    print(f"成功找到 {len(ligand_info_list)} 個有效的配體路徑。")
    return ligand_info_list

def main(num_ligands):
    """
    主工作流程：處理配體、建立複合物、檢查碰撞並過濾列表。
    """
    # --- 步驟 0: 環境設定 ---
    if not os.path.exists(COMPLEXES_DIR):
        print(f"正在建立複合物輸出資料夾: {COMPLEXES_DIR}")
        os.makedirs(COMPLEXES_DIR)

    # --- 步驟 1: 獲取配體路徑 ---
    ligand_infos = get_ligand_paths(TOP_LIGANDS_FILE, num_ligands)
    
    good_ligands_lines = []
    
    # --- 步驟 2 & 3: 迴圈處理、合併、檢查 ---
    for i, info in enumerate(ligand_infos):
        ligand_path = info['path']
        print(ligand_path)
        ligand_name_base = os.path.splitext(info['ligand_name'])[0]
        
        print(f"\n--- 正在處理配體 {i+1}/{len(ligand_infos)}: {info['ligand_name']} ---")
        
        # (2) 定義輸出路徑並呼叫 merge_pdbs 的邏輯
        output_complex_path = os.path.join(COMPLEXES_DIR, f"complex_{ligand_name_base}.pdb")
        merge_structures(protein_path=PROTEIN_PDB_PATH, ligand_path=ligand_path, output_path=output_complex_path)
        
        # (3) 呼叫 check_distance_2 的邏輯
        clash_count = check_steric_clashes(output_complex_path, threshold=1.0)
        
        if clash_count == 0:
            print(f"-> 結果: OK (無碰撞)。保留此配體。")
            good_ligands_lines.append(info['original_line'])
            
    # --- 步驟 4: 產生過濾後的新列表 ---
    print(f"\n--- 工作流程結束 ---")
    print(f"寫入過濾後的列表至: {FILTERED_LIGANDS_FILE}")
    with open(FILTERED_LIGANDS_FILE, 'w') as f:
        f.write(f"{'index':<35} {'energy':<10} {'label'}\n")
        for line in good_ligands_lines:
            f.write(line + "\n")
    print("✅ 完成。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="處理頂級配體，建立複合物，檢查碰撞並過濾。")
    parser.add_argument('--num_ligands', type=int, default=500, help='要處理的頂級配體數量。預設為 500。')
    args = parser.parse_args()
    main(args.num_ligands)