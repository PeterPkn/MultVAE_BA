import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df_bias = pd.read_csv('adv_multvae_gs.csv')
df_da = pd.read_csv('ADV_DA_club1_gs.csv')
df_adv = pd.read_csv('ADV_gs_adv2300.csv')

df_bias['Model'] = 'Baseline\nMultVAE'
df_da['Model'] = 'Debiased\n(CLUB = 1.0)'
df_adv['Model'] = 'Debiased\n(Adversarial\nα = 2300)'

df_combined = pd.concat([df_bias, df_da, df_adv], ignore_index=True)

plt.figure(figsize=(10, 6), dpi=300)
sns.set_theme(style="whitegrid")

sns.violinplot(
    x='Model',
    y='balanced_accuracy',
    data=df_combined,
    hue='Model',
    palette=['#d62728', '#1f77b4', '#2ca02c'],
    inner='quartile',
    cut=0,
    legend=False,
)

plt.title(
    f'LFM DemoBias: Adversarial Gender Predictability Across {len(df_bias)} Configurations',
    fontsize=15,
    fontweight='bold',
    pad=15,
)
plt.ylabel('Balanced Accuracy (Bias)', fontsize=13, fontweight='bold')
plt.xlabel('Recommender Model', fontsize=13, fontweight='bold')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.axhline(
    y=0.5,
    color='gray',
    linestyle='--',
    linewidth=1.8,
    label='Perfect Fairness (0.50)',
)

plt.legend(loc='lower left', frameon=True, shadow=True, fontsize=11)
plt.tight_layout()

plt.savefig(
    'lfmdemobias_bias_distribution_3models.pdf',
    format='pdf',
    bbox_inches='tight',
)
plt.savefig(
    'lfmdemobias_bias_distribution_3models.png',
    format='png',
    bbox_inches='tight',
)
plt.show()
