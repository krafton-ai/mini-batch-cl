import os
import torch
import random
import numpy as np
from minimize_contrastive_loss_text_utils import plot_heatmap

random.seed(44)

N=16

# ETF
etf_solution = torch.full((N, N), -1 / (N - 1))
for i in range(N):
    etf_solution[i, i] = 1.0

# Cross-polytope
crp_solution = torch.zeros((N, N))
for i in range(N):
    crp_solution[i, i] = 1.0
    if i % 2 == 0:
        j = i+1
    else:
        j = i-1
    crp_solution[i, j] = -1.0

# base_dir = "output_rarng/2023_05_10-05_31_26_N8_d4_lr0.5_accumFalse_s20000x1_u=vFalse_Bs[2]"
# target_path = f"{base_dir}/z_NcB_mini_batch_B2_19999.pt"
# source_base_path = f"{base_dir}/N8_d4_lr0.5_s20000_z_fixed_mini_batch_B2_NcB"

# N = 8
# base_dir = "output/2023_05_15-01_29_43_N8_d4_lr0.5_accumFalse_s50000x1_u=vFalse_Bs[2]_seed44"
# target_path = f"{base_dir}/z_full_mini_batch_B2_49999.pt"
# source_base_path = f"{base_dir}/N8_d4_lr0.5_s50000_z_fixed_mini_batch_B2_full"
# solution_path = f"{base_dir}/N8_d4_B2_CP"

# base_dir = "output/2023_05_15-01_30_52_N8_d4_lr0.5_accumFalse_s50000x1_u=vFalse_Bs[2]_seed44"
# target_path = f"{base_dir}/z_f_mini_batch_B2_49999.pt"
# source_base_path = f"{base_dir}/N8_d4_lr0.5_s50000_z_fixed_mini_batch_B2_f"
# solution_path = f"{base_dir}/N8_d4_B2_CP"

# base_dir = "output/2023_05_15-01_34_10_N8_d4_lr0.5_accumFalse_s50000x1_u=vFalse_Bs[2]_seed44"
# target_path = f"{base_dir}/z_NcB_mini_batch_B2_49999.pt"
# source_base_path = f"{base_dir}/N8_d4_lr0.5_s50000_z_fixed_mini_batch_B2_NcB"
# solution_path = f"{base_dir}/N8_d4_B2_CP"

# N = 8 ETF
# base_dir = "output/2023_05_15-01_30_53_N8_d16_lr0.5_accumFalse_s50000x1_u=vFalse_Bs[2]_seed44"
# # base_dir = "output/2023_05_14-03_12_01_N8_d16_lr0.5_accumFalse_s50000x1_u=vFalse_Bs[2]_seed42"
# target_path = f"{base_dir}/z_f_mini_batch_B2_49999.pt"
# source_base_path = f"{base_dir}/N8_d16_lr0.5_s50000_z_fixed_mini_batch_B2_f"
# solution_path = f"{base_dir}/N8_d16_B2_ETF"


# N = 16
# base_dir = "output/2023_05_15-01_31_37_N16_d8_lr0.5_accumFalse_s500000x1_u=vFalse_Bs[2]_seed44"
# target_path = f"{base_dir}/z_full_mini_batch_B2_499999.pt"
# source_base_path = f"{base_dir}/N16_d8_lr0.5_s500000_z_fixed_mini_batch_B2_full"
# solution_path = f"{base_dir}/N16_d8_B2_CP"

# base_dir = "output/2023_05_15-01_36_26_N16_d8_lr0.5_accumFalse_s500000x1_u=vFalse_Bs[2]_seed44"
# target_path = f"{base_dir}/z_f_mini_batch_B2_499999.pt"
# source_base_path = f"{base_dir}/N16_d8_lr0.5_s500000_z_fixed_mini_batch_B2_f"
# solution_path = f"{base_dir}/N16_d8_B2_CP"

# base_dir = "output/2023_05_15-01_57_39_N16_d8_lr0.5_accumFalse_s500000x1_u=vFalse_Bs[2]_seed44"
# target_path = f"{base_dir}/z_NcB_mini_batch_B2_499999.pt"
# source_base_path = f"{base_dir}/N16_d8_lr0.5_s500000_z_fixed_mini_batch_B2_NcB"
# solution_path = f"{base_dir}/N16_d8_B2_CP"

base_dir = "output/2023_05_11-14_39_54_N16_d32_lr0.5_accumFalse_s50000x1_u=vFalse_Bs[2]"
target_path = f"{base_dir}/z_f_mini_batch_B2_49999.pt"
source_base_path = f"{base_dir}/N16_d32_lr0.5_s50000_z_fixed_mini_batch_B2_f"
solution_path = f"{base_dir}/N16_d32_B2_ETF"

# N = 4
# base_dir = "output/2023_05_15-01_27_48_N4_d2_lr0.5_accumFalse_s50000x1_u=vFalse_Bs[2]_seed44"
# target_path = f"{base_dir}/z_full_mini_batch_B2_49999.pt"
# source_base_path = f"{base_dir}/N4_d2_lr0.5_s50000_z_fixed_mini_batch_B2_full"
# solution_path = f"{base_dir}/N4_d2_B2_CP"

# base_dir = "output/2023_05_15-01_27_49_N4_d2_lr0.5_accumFalse_s50000x1_u=vFalse_Bs[2]_seed44"
# target_path = f"{base_dir}/z_f_mini_batch_B2_49999.pt"
# source_base_path = f"{base_dir}/N4_d2_lr0.5_s50000_z_fixed_mini_batch_B2_f"
# solution_path = f"{base_dir}/N4_d2_B2_CP"


# base_dir = "output/2023_05_15-01_27_49_N4_d2_lr0.5_accumFalse_s50000x1_u=vFalse_Bs[2]_seed44"
# target_path = f"{base_dir}/z_NcB_mini_batch_B2_49999.pt"
# source_base_path = f"{base_dir}/N4_d2_lr0.5_s50000_z_fixed_mini_batch_B2_NcB"
# solution_path = f"{base_dir}/N4_d2_B2_CP"


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
# print(z)
plot_heatmap(z, filename=os.path.join(source_base_path + "_w_cbar_rarng"), plot_cbar=True)
plot_heatmap(z, filename=os.path.join(source_base_path + "_wo_cbar_rarng"), plot_cbar=False)
plot_heatmap(solution, filename=os.path.join(solution_path + "_w_cbar_rarng"), plot_cbar=True)
plot_heatmap(solution, filename=os.path.join(solution_path + "_wo_cbar_rarng"), plot_cbar=False)


