import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_bias = pd.read_csv('adv_search_biasVAE.csv', skiprows=1) # row 1 comment
df_da = pd.read_csv('adv_search_DA_club_1.0.csv', skiprows=1) # row 1 comment
df_adv = pd.read_csv('adv_search_ADV_alpha_2300.csv', skiprows=1) # row 1 comment

df_bias['Model'] = 'Baseline MultVAE'
df_da['Model'] = 'Debiased (CLUB = 1.0)'
df_adv['Model'] = 'Adversarial (α = 2300)'
df_combined = pd.concat([df_bias, df_da, df_adv])

plt.figure(figsize=(10, 6), dpi=300)
sns.set_theme(style="whitegrid")

sns.violinplot(
    x='Model', 
    y='balanced_accuracy', 
    data=df_combined, 
    inner='quartile', 
    palette=['#d62728', '#1f77b4', '#2ca02c'],
    cut=0
)

plt.title('Adversarial Gender Predictability Across 162 Configurations', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Balanced Accuracy (Bias)', fontsize=12, fontweight='bold')
plt.xlabel('Recommender Model', fontsize=12, fontweight='bold')

plt.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='Perfect Fairness (0.50)')

plt.legend(loc='lower left', frameon=True, shadow=True)
plt.tight_layout()

plt.savefig('bias_distribution_violin.pdf', format='pdf', bbox_inches='tight')
plt.savefig('bias_distribution.png', format='png', bbox_inches='tight')
plt.show()