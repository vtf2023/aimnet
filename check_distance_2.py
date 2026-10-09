import numpy as np

def check_steric_clashes(pdb_file, threshold=1.0):
    """掃描 PDB 檔，找出距離小於 threshold (預設 1.0 Å) 的原子對"""
    atoms = []
    
    # 1. 讀取 PDB 座標
    # 使用 enumerate 來同時獲取行號 (從 1 開始)
    with open(pdb_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # 簡單解析 PDB 格式
                chain = line[21].strip()
                res_name = line[17:20].strip()
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atoms.append({'name': atom_name, 'res': res_name, 'chain': chain, 'coords': np.array([x, y, z]), 'line': line_num})
    
    clash_count = 0
    print(f"開始掃描 {pdb_file} (閾值: {threshold} Å)...")
    
    # 2. 計算兩兩距離 (這是一個簡化的 O(N^2) 寫法，適合小分子或局部檢查)
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            # 排除同一個殘基內的原子 (它們本來就連在一起)
            if atoms[i]['res'] == atoms[j]['res'] and atoms[i]['chain'] == atoms[j]['chain']:
                continue
                
            dist = np.linalg.norm(atoms[i]['coords'] - atoms[j]['coords'])
            
            if dist < threshold:
                clash_count += 1
                print(f"⚠️ 發現碰撞! 距離: {dist:.2f} Å | "
                      f"原子 A: {atoms[i]['res']}-{atoms[i]['name']} (行: {atoms[i]['line']}) 與 "
                      f"原子 B: {atoms[j]['res']}-{atoms[j]['name']} (行: {atoms[j]['line']})")
                
    if clash_count == 0:
        print("✅ 結構很乾淨，沒有發現明顯的立體障礙！")
    else:
        print(f"❌ 總共發現 {clash_count} 處嚴重的立體障礙。")
    
    return clash_count

# 執行檢查
#check_steric_clashes('4J0P_cleaned_Noaddres_with_H_wo_Na_DMS_Water.pdb', threshold=1.0)
#check_steric_clashes('4J0V_cleaned_Noaddres_with_H_wo_Na_DMS_Water.pdb', threshold=1.0)
#check_steric_clashes('4J1H_cleaned_Noaddres_with_H_wo_Na_DMS_Water.pdb', threshold=1.0)
#check_steric_clashes('4J1F_cleaned_Noaddres_with_H_wo_Na_DMS_Water.pdb', threshold=1.0)
#check_steric_clashes('4J0P.pdb', threshold=1.0)
# check_steric_clashes('./2nnq_data/complex_2nnq.pdb', threshold=1.0)