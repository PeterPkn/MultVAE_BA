import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the 5 grid search results
df_bias = pd.read_csv('adv_ekstrabladet_mulutvae_gs.csv')
df_da = pd.read_csv('adv_ekstrabladet_DA_gs.csv')
df_adv = pd.read_csv('adv_ekstrabladet_ADV_gs.csv')
df_club5 = pd.read_csv('adv_club_5_gs.csv')      # Your newly provided CLUB 5.0 data
df_club10 = pd.read_csv('adv_club_10_gs.csv')

# 2. Assign labels
df_bias['Model'] = 'Baseline\nMultVAE'
df_da['Model'] = 'Debiased\n(CLUB = 1.0)'
df_adv['Model'] = 'Debiased\n(Adversarial)'
df_club5['Model'] = 'Debiased\n(CLUB = 5.0)'
df_club10['Model'] = 'Debiased\n(CLUB = 10.0)'

# 3. Combine into a single DataFrame
df_combined = pd.concat([df_bias, df_da, df_adv, df_club5, df_club10])

# 4. Setup the plot
plt.figure(figsize=(14, 7), dpi=300) 
sns.set_theme(style="whitegrid")

# 5. Create the Violin Plot with a 5-color palette
sns.violinplot(
    x='Model', 
    y='balanced_accuracy', 
    data=df_combined, 
    inner='quartile', 
    palette=['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd'], 
    cut=0
)

# 6. Formatting
plt.title('Ekstrabladet: Adversarial Gender Predictability Across 162 Configurations', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Balanced Accuracy (Bias)', fontsize=14, fontweight='bold')
plt.xlabel('Recommender Model', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 7. Add the baseline of perfect fairness
plt.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='Perfect Fairness (0.50)')

plt.legend(loc='lower left', frameon=True, shadow=True, fontsize=12)
plt.tight_layout()

# 8. Save the figures for the thesis
plt.savefig('ekstrabladet_bias_distribution_5models.pdf', format='pdf', bbox_inches='tight')
plt.savefig('ekstrabladet_bias_distribution_5models.png', format='png', bbox_inches='tight')
plt.show()
