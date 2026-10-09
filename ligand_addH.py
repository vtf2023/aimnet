import subprocess
from rdkit import Chem

# 1. 讀取原始分子並使用 RDKit 進行加氫（確保重原子與氫原子連續編號）
mol = Chem.MolFromPDBFile('ligand_1H8.pdb', removeHs=True)
mol_h = Chem.AddHs(mol, addCoords=True)

# 2. 先寫出成暫存的 PDB 檔案（此時拓撲順序已經在 RDKit 內固定）
temp_pdb = 'temp_reordered.pdb'
Chem.MolToPDBFile(mol_h, temp_pdb)

# 3. 使用 Python 呼叫系統的 obabel 指令，將 PDB 完美轉換為 .mol2
output_mol2 = 'ligand_1H8_H_3.mol2'
try:
    # 執行 obabel 轉換命令
    subprocess.run(['obabel', temp_pdb, '-O', output_mol2], check=True)
    print(f"成功將排序後的結構轉換並輸出至：{output_mol2}")
except subprocess.CalledProcessError as e:
    print(f"Open Babel 轉換失敗: {e}")
except FileNotFoundError:
    print("錯誤：系統未安裝 Open Babel，請先確保環境中可執行 'obabel' 指令。")