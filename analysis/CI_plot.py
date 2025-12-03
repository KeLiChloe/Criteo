import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D
import warnings

# ==========================================
# 数据加载与处理
# ==========================================
with open("exp_results/main/exp6.pkl", "rb") as f:
    data_exp = pickle.load(f)
    
# 忽略不必要的警告
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 1. 核心修复：强制使用 STIX/Times 风格
# ==========================================
sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks", {'axes.grid': True})

plt.rcParams.update({
    'font.family': 'STIXGeneral', 
    'font.serif': ['STIXGeneral', 'Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'axes.labelweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'pdf.fonttype': 42,
    'figure.dpi': 300
})


# 获取参数
if "params" in data_exp:
    params = data_exp["params"]
    print("Experiment Params:", params) 
    # 输出: {'sample_frac': 0.01, 'pilot_frac': 0.5, 'train_frac': 0.7, 'N_sim': 100}

    # 获取结果列表
    results_list = data_exp["results"]
else:
    results_list = data_exp
n_sims = len(results_list)


baselines = ['all_treat', 'all_control', 'random', 'kmeans', 'gmm', 'clr', 'mst']  # 'mst' 可选
target = 'dast'
records = []
raw_data_map = {m: [] for m in baselines + [target]}

for i, run in enumerate(results_list):
    val_target = run[target]
    raw_data_map[target].append(val_target)
    for base in baselines:
        val_base = run[base]
        raw_data_map[base].append(val_base)
        lift = ((val_target - val_base) / val_base) * 100
        records.append({'Run': i, 'Baseline': base, 'Lift': lift})

df = pd.DataFrame(records)

# 【关键修改 1】：增加 "vs." 前缀，明确对比关系
label_map = {
    'all_treat': 'vs. All Treat', 
    'all_control': 'vs. All Control',
    'random': 'vs. Random', 
    'kmeans': 'vs. K-Means',
    'gmm': 'vs. GMM', 
    'clr': 'vs. CLR',
    'mst': 'vs. MST',
    # 'policytree': 'vs. Policytree',
}
df['Baseline_Label'] = df['Baseline'].map(label_map)

# ==========================================
# 3. 统计计算 (P-value & CI)
# ==========================================
def get_sig_star(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return None

p_values = {label_map[b]: stats.ttest_rel(raw_data_map[target], raw_data_map[b])[1] for b in baselines}

constant_order = [
    'vs. All Treat', 
    'vs. All Control',
    'vs. Random', 
    'vs. K-Means',
    'vs. GMM', 
    'vs. CLR',
    'vs. MST',
    # 'vs. Policytree',
]

# ✅ 颜色与 comparator label 一一绑定（不再依赖顺序）
palette = {
    'vs. All Treat':   "#4C72B0DD",
    'vs. All Control': "#55A868DD",
    'vs. Random':      "#C44E52DD",
    'vs. K-Means':     "#8172B2DD",
    'vs. GMM':         "#CCB974DD",
    'vs. CLR':         "#64B5CDDD",
    'vs. MST':         "#8C613CDD",
    # 'Policytree':  "#937860",

}




summary_stats = []
for label in constant_order:
    subset = df[df['Baseline_Label'] == label]['Lift']
    mean = subset.mean()
    sem = subset.sem()
    ci = sem * stats.t.ppf(0.975, len(subset)-1)
    p_star = get_sig_star(p_values[label])
    summary_stats.append({
        'Label': label,
        'Mean': mean,
        'CI': ci,
        'P_Star': p_star,
    })

stats_df = pd.DataFrame(summary_stats)
# ==========================================
# 按 mean 动态排序（从大到小）
# ==========================================


# ==========================================
# 4. 绘图 (Drawing)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = range(len(stats_df))

ax.grid(axis='x', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

for i, row in stats_df.iterrows():
    color = palette[row['Label']]
    ax.errorbar(x=row['Mean'], y=i, xerr=row['CI'], fmt='o',
                color=color, ecolor=color,
                capsize=4, elinewidth=2, markersize=9)

ytick_labels = [f"{row['Label']}" + (f" ({row['P_Star']})" if row['P_Star'] is not None else '') for _, row in stats_df.iterrows()]
ax.set_yticks(y_pos)
ax.set_yticklabels(ytick_labels, fontweight='bold', fontsize=14)
ax.tick_params(axis='y', length=0)

# 0% 基准线
ax.axvline(0, color="#E40606", linestyle='--', linewidth=1.6, alpha=0.8)

# 轴标题
ax.set_xlabel('Averaged DAST Improvement (%) on Conversion Rate Over Comparators', fontweight='bold', labelpad=12)

ax.invert_yaxis()
sns.despine(left=True, top=True, right=True)

# 标题
ax.set_title(f'Averaged DAST Improvement (%) on Conversion Rate Over Comparators with 95% CI (Runs={n_sims})', 
             fontweight='bold', 
             pad=20,
             fontsize=16, 
             y=1.12)

# 说明文字
ax.annotate('Positive values (>0%) indicate DAST outperforms comparators', 
            xy=(0.5, 1.06),
            xycoords='axes fraction',
            fontsize=12, fontweight='bold', color='#333333',
            ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray", lw=0.5, alpha=0.8))

# ==========================================
# 5. 优化图例 (Legend)
# ==========================================
legend_handles = [
    Line2D([0], [0], color='black', marker='o', linestyle='-', 
           linewidth=2, markersize=8, label='Mean ± 95% CI'),
    Line2D([0], [0], color='none', label='*** p < 0.001\n**   p < 0.01\n*     p < 0.05'),
    Line2D([0], [0], color="#E40606", linestyle='--', linewidth=1.6, label='No Improvement (0%)')
]
ax.legend(handles=legend_handles, 
          loc='best',
          frameon=True, 
          framealpha=0.95, 
          edgecolor='#E0E0E0', 
          fancybox=False,      
          fontsize=11,
          borderpad=1)

plt.tight_layout()

# ==========================================
# 6. 导出
# ==========================================
plt.savefig('figures/Fig2_UTD_Style.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/Fig2_UTD_Style.png', dpi=300, bbox_inches='tight')
