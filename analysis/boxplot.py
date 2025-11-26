import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import warnings
from matplotlib.patches import Patch

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 1. 核心风格设置 (与 Forest Plot 保持 100% 统一)
# ==========================================
# 先设置 seaborn
sns.set_context("paper", font_scale=1.4)
# 开启网格，但后续我们只保留 Y 轴网格
sns.set_style("ticks", {'axes.grid': True}) 

plt.rcParams.update({
    'font.family': 'STIXGeneral', 
    'font.serif': ['STIXGeneral', 'Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.pad': 8,
    'ytick.major.pad': 8,
    'pdf.fonttype': 42,
    'figure.dpi': 300
})

# 沿用 NPG 学术配色
academic_colors = ['#FF6B6B', '#4ECDC4', "#7F8893", '#C7F464', '#FFCC5C', "#79CBFB",  "#DA7BFF"]

# ==========================================
# 2. 数据处理
# ==========================================

with open("exp_results/exp3.pkl", "rb") as f:
    data_exp = pickle.load(f)

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

baselines = ['all_treat', 'all_control', 'random', 'kmeans', 'gmm', 'clr']

target = 'dast'
records = []

for i, run in enumerate(results_list):
    for base in baselines:
        lift = ((run[target] - run[base]) / run[base]) * 100
        records.append({
            'Run': i,
            'Baseline': base,
            'Lift': lift
        })

df = pd.DataFrame(records)
label_map = {
    'all_treat': 'vs. All Treat', 
    'all_control': 'vs. All Control',
    'random': 'vs. Random', 
    'kmeans': 'vs. K-Means',
    'gmm': 'vs. GMM', 
    'clr': 'vs. CLR',
    # 'mst': 'vs. MST',
}
df['Baseline_Label'] = df['Baseline'].map(label_map)

# 排序 + 颜色映射
# median_order = df.groupby('Baseline_Label')['Lift'].median().sort_values(ascending=False).index.tolist()
constant_order = [
    'vs. All Treat', 
    'vs. All Control',
    'vs. Random', 
    'vs. K-Means',
    'vs. GMM', 
    'vs. CLR',
    # 'vs. MST',
]
palette = {label: color for label, color in zip(constant_order, academic_colors)}

# ==========================================
# 3. 绘图 (Drawing)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

# 3.1 添加水平网格线 (Grid) - 放在最底层
ax.grid(axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

# 3.2 绘制 Boxplot
# showmeans=True: 显示均值点，这是顶刊常用的信息补充
# meanprops: 定义均值点的样式（白色菱形，黑边）
sns.boxplot(
    x='Baseline_Label', y='Lift', data=df, order=constant_order,
    palette=palette, showfliers=False, width=0.55, linewidth=1.2, ax=ax,
    showmeans=True, 
    meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":6},
    boxprops=dict(edgecolor='black', linewidth=1.2, alpha=0.9),  
    whiskerprops=dict(color='black', linewidth=1.2),
    capprops=dict(color='black', linewidth=1.2),
    medianprops=dict(color="#C0392B", linewidth=1.5)
)

# 3.3 绘制散点 (Swarm/Strip)
# jitter=True: 让点散开，避免重叠成一条线
# alpha=0.4: 半透明，避免抢眼
sns.stripplot(
    x='Baseline_Label', y='Lift', data=df, order=constant_order,
    color='#333333', alpha=0.4, size=4, jitter=False, ax=ax
)

# 3.4 辅助线
ax.axhline(0, color="#2332F9", linestyle='--', linewidth=1.6, alpha=0.8)

# 3.5 标签优化
ax.set_ylabel('DAST Improvement Over Comparators (%)', fontweight='bold', labelpad=12)
ax.set_xlabel('Benchmarks and Comparators', fontweight='bold', labelpad=12)

# 设置 Y 轴范围 (可选，根据数据调整，留出图例空间)
ax.set_ylim(-50, 150) 

# X 轴标签微调
ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold', rotation=0)

# 去除多余边框 (Despine)
sns.despine(trim=True, offset=10) # offset让轴稍微分离，更有透气感

ax.set_title(f'Distribution of DAST Improvement (%) Over Comparators across {n_sims} Runs', 
             fontweight='bold', 
             pad=20,          # 如果用了 y，pad 可以稍微改小或者保持，主要靠 y 控制绝对位置
             fontsize=16, 
             y=1.12)          # <--- 【核心修改】：添加 y 参数，值越大越往上

# 2. 调整 "Higher is Better" 标注位置
# 将 xy 的 y 坐标从原来的 1.02 改为 1.06 (或者更高)，让它跟着标题一起往上走
ax.annotate('Positive values (>0%) indicate DAST outperforms baselines', 
            xy=(0.5, 1.06),             # <--- 【核心修改】：将 1.02 改为 1.06 (往上挪)
            xycoords='axes fraction',
            fontsize=12, fontweight='bold', color='#333333',
            ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray", lw=0.5, alpha=0.8))

# 我们手动构建图例句柄 (Handles)，以便精确控制每一个符号的含义
legend_elements = [
    # 1. Box: 代表 IQR
    # 使用灰色 Patch 代表箱体，显得中立
    Patch(facecolor='#E0E0E0', edgecolor='black', linewidth=1.2, 
          label='Box: IQR (25% – 75% quantile)'),

    # 2. Line: 代表 Median
    Line2D([0], [0], color='#C0392B', linewidth=1.5, 
           label='Red Line: Median'),

    # 3. Diamond: 代表 Mean
    # linestyle='None' 确保只显示菱形，不显示贯穿线
    Line2D([0], [0], marker='D', color='w', markerfacecolor='white', 
           markeredgecolor='black', markersize=6, linestyle='None',
           label='Diamond: Mean'),

    # 4. Whiskers: non-outlier range (核心修改为 "工" 字形)
    Line2D([0], [0], color='black', linewidth=1.2,
           marker='_',              # 使用下划线 '_' 作为 marker，它本身就是一条横线
           markersize=15,           # 调大 marker 尺寸，让它看起来像 Whiskers 的 cap
           markeredgewidth=1.2,     # marker 边缘宽度与线宽保持一致
           markerfacecolor='black', # marker 的填充色（使横线实心）
           label=r'Whiskers: [$Q_1 - 1.5 \cdot \mathrm{IQR}, \ Q_3 + 1.5 \cdot \mathrm{IQR}$]'),
    
    
        # 使用 marker='o' 和 stripplot 的颜色（#333333）
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#333333', alpha=0.4,
           markersize=5, linestyle='None', # 尺寸略微调大到 5，在图例中更清晰
           label='Dot: Single run result'),
]

# 绘制图例
leg = ax.legend(handles=legend_elements, 
                loc='upper right', 
                frameon=True, 
                framealpha=0.95, 
                edgecolor='#B0B0B0', 
                fancybox=False, 
                fontsize=10, 
                borderpad=0.8, 
                labelspacing=0.6, 
                handlelength=1.5)

# 可选：让图例左对齐 (默认通常是居中或自动，左对齐阅读体验更好)
leg._legend_box.align = "left"

plt.tight_layout()

# ==========================================
# 5. 导出
# ==========================================
plt.savefig('figures/Fig1_Boxplot_UTD_Style.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/Fig1_Boxplot_UTD_Style.png', dpi=300, bbox_inches='tight')
