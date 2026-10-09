import numpy as np
import scipy.stats as stats

# 假設這是您用 AIMNet2 算出的結合能 (越負越好)
# 這裡隨機給定 5 個分子的預測能量
predicted_scores = np.array([-9.5, -8.2, -6.1, -11.4, -5.0])

# 這是實驗測得的真實結合力 (例如 pIC50, 數值越大代表結合力越強)
# 對應上面 5 個分子
experimental_affinities = np.array([8.1, 7.0, 5.5, 9.2, 4.8])

# 1. 計算皮爾森相關係數 (線性相關)
# 注意：預測能量是越小越好，實驗 pIC50 是越大越好，所以預期會得到一個負相關
pearson_r, p_value_p = stats.pearsonr(predicted_scores, experimental_affinities)

# 2. 計算斯皮爾曼等級相關係數 (排序能力)
spearman_rho, p_value_s = stats.spearmanr(predicted_scores, experimental_affinities)

print(f"--- 排序能力評估報告 ---")
print(f"皮爾森相關係數 (Pearson R): {pearson_r:.3f}")
# 因為我們預期是負相關，所以可以取絕對值來看強度
print(f"  > 線性相關強度 (|R|): {abs(pearson_r):.3f}")

print(f"斯皮爾曼相關係數 (Spearman Rho): {spearman_rho:.3f}")
print(f"  > 純排序能力 (|Rho|): {abs(spearman_rho):.3f}")

if abs(spearman_rho) > 0.8:
    print("\n結論：您的打分方法展現出【極其優異】的排序能力！")
elif abs(spearman_rho) > 0.6:
    print("\n結論：您的打分方法具有【良好】的排序能力。")
else:
    print("\n結論：排序能力較弱，建議檢查模型或進行進一步的結構優化。")