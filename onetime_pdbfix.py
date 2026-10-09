#import numpy as np
from pdbfixer import PDBFixer
from openmm.app import PDBFile
#import torch
from openmm.app import Modeller

#input_filename = '4J0P.pdb'
#output_filename = '4J0P_cleaned_Noaddres_with_H_wo_Na_DMS_Water.pdb'

#input_filename = '4J0V.pdb'
#output_filename = '4J0V_cleaned_Noaddres_with_H_wo_Na_DMS_Water.pdb'

#input_filename = '4J1H.pdb'
#output_filename = '4J1H_cleaned_Noaddres_with_H_wo_Na_DMS_Water.pdb'

#input_filename = '4J1F.pdb'
#output_filename = '4J1F_cleaned_Noaddres_with_H_wo_Na_DMS_Water.pdb'

input_filename = '/home/kuoweihuang/aimnet/2nnq_data/processed_2nnq.pdb'
output_filename = '/home/kuoweihuang/aimnet/2nnq_data/processed_2nnq_with_H.pdb'


# 1. 自動前處理（免手動）# 1. 使用 PDBFixer 讀取 PDB 檔案並進行初步修復
fixer = PDBFixer(filename=input_filename)


# 3. 補齊缺失結構與自動加氫 您也可以視需求啟用以下功能來補齊缺失的重原子
#fixer.findMissingResidues()
#fixer.findMissingAtoms()
#fixer.addMissingAtoms() # 自動補齊缺失原子
fixer.addMissingHydrogens(7.0) # 在 pH 7.0 下自動加氫
print("已成功加入氫原子。")

# 2. 將修復好的結構交給 OpenMM 的 Modeller 進行精細編輯
modeller = Modeller(fixer.topology, fixer.positions)

# 新增：刪除指定的 HETATM (NA 和 DMS)
residues_to_delete = []
for res in modeller.topology.residues():
    # 檢查殘基名稱是否為 NA 或 DMS
    if res.name in ['NA', 'DMS']:
        residues_to_delete.append(res)

modeller.delete(residues_to_delete)
print(f"已成功移除 {len(residues_to_delete)} 個 NA/DMS 殘基。")

# 3. 刪除水分子 (Modeller 有內建專門且安全的方法，絕對不會動到配體)
modeller.deleteWater()
print("已成功使用 Modeller 移除所有水分子，並完整保留 Ligand。")

# 4. 儲存最終乾淨的 PDB 檔案
with open(output_filename, 'w') as f:
    PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)

print(f"儲存成功！新檔案為: {output_filename}")

print("檔案儲存成功！現在您可以隨時將此檔案送入 AIMNet2 進行計算，或用 PyMOL 開啟檢查。")
