import os
import subprocess

def prepare_protein_with_pqr(pdb_id):
    # 1. 下載原始 PDB
    #os.system(f"wget https://files.rcsb.org/download/{pdb_id}.pdb")
    
    # 2. 呼叫 pdb2pqr 進行結構修復與加氫
    input_pdb = f"{pdb_id}.pdb"
    output_pqr = f"{pdb_id}_H70_1.pdb" # 雖然副檔名用 pdb，但內容是加好氫的結構
    
    cmd = ["pdb2pqr", "--ff=AMBER", "--with-ph=7.0", input_pdb, output_pqr]
    
    print(f"正在為 {pdb_id} 補全氫原子與修復結構...")
    subprocess.run(cmd, check=True)
    print(f"優化完成！新檔案已儲存為: {output_pqr}")

# 執行測試
prepare_protein_with_pqr("4J0P")