import os
import torch
import random
import numpy as np
from minimize_contrastive_loss_text_utils import plot_heatmap

random.seed(43)

N=4

# ETF
etf_solution = torch.full((N, N), -1 / (N - 1))
for i in range(N):
    etf_solution[i, i] = 1.0

# Cross-polytope
crp_solution = torch.zeros((N, N))
for i in range(N):
    crp_solution[i, i] = 1.0
    random_index = random.choice([x for x in range(N) if x != i])
    crp_solution[i, random_index] = -1.0


# base_dir = "output_rarng/2023_05_10-05_31_26_N8_d4_lr0.5_accumFalse_s20000x1_u=vFalse_Bs[2]"
# target_path = f"{base_dir}/z_NcB_mini_batch_B2_19999.pt"
# source_base_path = f"{base_dir}/N8_d4_lr0.5_s20000_z_fixed_mini_batch_B2_NcB"
# base_dir = "output_rarng/2023_05_10-07_00_46_N8_d4_lr0.5_accumFalse_s1000x1_u=vFalse_Bs[2]"
base_dir = "output/2023_05_15-01_27_49_N4_d2_lr0.5_accumFalse_s50000x1_u=vFalse_Bs[2]_seed44"
target_path = f"{base_dir}/z_NcB_mini_batch_B2_49999.pt"
source_base_path = f"{base_dir}/N4_d2_lr0.5_s50000_z_fixed_mini_batch_B2_NcB"
if not os.path.isfile(target_path):
    raise ValueError
z = torch.load(target_path)
solution = crp_solution

# 행별로 재배열하기
def rearranging(z, solution):
    for i in range(z.shape[0]):
        # 두 번째 텐서에서 각 값에 대해 첫 번째 텐서의 모든 값과의 차이를 계산합니다.
        diffs = torch.abs(solution[i].unsqueeze(-1) - z[i])
        # numpy로 변환합니다.
        diffs_np = diffs.numpy()
        selected = set()
        result = []
        for row in diffs_np:
            # 아직 선택되지 않은 가장 작은 요소를 찾습니다.
            for idx in np.argsort(row):
                if idx not in selected:
                    # 선택된 요소를 추가하고 다음 행으로 넘어갑니다.
                    selected.add(idx)
                    result.append(z[i][idx].item())
                    break
        z[i] = torch.tensor(result)
    return z
print("rearranging..")
z = rearranging(z, solution)
print(z)
import pdb; pdb.set_trace()
#print(torch.round(z, decimals=3))
plot_heatmap(z, filename=os.path.join(source_base_path + "_w_cbar_rarng"), plot_cbar=True)
plot_heatmap(z, filename=os.path.join(source_base_path + "_wo_cbar_rarng"), plot_cbar=False)