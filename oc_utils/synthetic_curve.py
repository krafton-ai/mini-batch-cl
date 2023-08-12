## New plotting ##

import wandb
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.font_manager as font_manager

from matplotlib import rc
import os 
 
rc('text', usetex=True)  # 외부 LaTeX 사용

# 프로젝트 이름, 엔티티 이름, 그룹 이름, 실행 이름을 설정합니다.
project_name = "unimodal_synthetic"
entity_name = "krafton_clap"
# group_name = "2023_05_11-04_19_59_N8_d4_lr0.5_accumFalse_s500x1_u=vFalse_Bs[2]"
# group_name = "2023_05_17-08_44_29_N8_d3_lr0.5_accumFalse_s500x1_u=vFalse_Bs[2]_seed44"

group_names = [
    ["2023_05_11-10_16_24_N4_d2_lr0.5_accumFalse_s500x1_u=vFalse_Bs[2]", 'legend_off'],
    ["2023_05_11-14_37_36_N4_d8_lr0.5_accumFalse_s500x1_u=vFalse_Bs[2]", 'legend_on'],
    ["2023_05_11-04_19_59_N8_d4_lr0.5_accumFalse_s500x1_u=vFalse_Bs[2]", 'legend_off'],
    ["2023_05_11-02_04_06_N8_d16_lr0.5_accumFalse_s500x1_u=vFalse_Bs[2]", 'legend_on'],
    ["2023_05_11-14_39_11_N16_d8_lr0.5_accumFalse_s500x1_u=vFalse_Bs[2]", 'legend_off'],
    ["2023_05_11-14_39_07_N16_d32_lr0.5_accumFalse_s500x1_u=vFalse_Bs[2]", 'legend_on']
]

for group_name, legend_2 in group_names:

    N = group_name.split("_N")[1].split("_d")[0]
    d = group_name.split("_d")[1].split("_lr")[0]
    print(f"N {N} d {d}")

    run_names = ["osgd_NcB_B2", "sc_even_B2",  "s_B2"]

    legend = {"osgd_NcB_B2": "OSGD", "s_B2": "SGD", "sc_even_B2":"Spectral Clustering Method"}
    linestyle = {"osgd_NcB_B2": "--", "s_B2": ":", "sc_even_B2":"-"}

    c = ['#EF476F','#118AB2']
    # color = {"osgd_NcB_B2": "red", "s_B2": "blue", "sc_even_B2":"green"}
    color = {"osgd_NcB_B2": '#EF476F', "s_B2": '#118AB2', "sc_even_B2":'#228B22'}

    graph_dict = {}

    # Wandb 프로젝트를 로드합니다.
    api = wandb.Api()
    runs = api.runs(path="{}/{}".format(entity_name, project_name))
    fig, ax = plt.subplots(figsize=(7,3))

    # plt.figure(figsize=(14,6))

    # 로그 데이터를 불러옵니다.
    for run in runs:
        for run_name in run_names:
            if run.state == "finished" and run.group == group_name and run.name == run_name:
                run_history = run.history()
                if "solution_norm" in run_history:
                    print(run_name)
                    loss_list = run_history["solution_norm"]
                    line, = ax.plot(loss_list, 
                                    linewidth=3.5, 
                                    linestyle=linestyle[run_name], 
                                    color=color[run_name])
                    graph_dict[run_name] = line

    fontsize = 20
    rc('text', usetex=True) 
    plt.ylabel(r'$\|U^{*T}V^*-U^TV\|_F$', fontsize=fontsize, labelpad=20)
    plt.yticks(fontsize=fontsize)
    if legend_2 == 'legend_off':
        plt.xlabel("Update steps", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)

    font = font_manager.FontProperties(variant='small-caps', size=16)
    # handles, labels = ax.get_legend_handles_labels()
    handles=[graph_dict["osgd_NcB_B2"], graph_dict["sc_even_B2"], graph_dict["s_B2"]]
    labels = ["OSGD", "Spectral Clustering Method", "SGD"]
    # fig.legend(handles, labels, bbox_to_anchor=(0.6, 0.98), ncol=1, loc='upper left', prop=font)
    if legend_2 == 'legend_on':
        fig.legend(handles, labels, bbox_to_anchor=(0.49, 0.98), ncol=1, loc='upper left', prop=font)

    ax.grid(True)

    fig.tight_layout()

    # plt.legend(handles=[graph_dict["osgd_NcB_B2"], graph_dict["sc_even_B2"], graph_dict["s_B2"]], fontsize=25)

    dir_name = f"synthetic_curve_output/{group_name}"
    print(f"{group_name}")
    os.makedirs(dir_name, exist_ok=True)

    plt.savefig(f"{dir_name}/norm_diff_N{N}_d{d}.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{dir_name}/norm_diff_N{N}_d{d}.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()