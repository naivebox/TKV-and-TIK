import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Načtení dat ze Spin-Lock Optimizeru
df = pd.read_csv('tcd_spin_lock_results.csv', delimiter=';')

# Nalezení maxima a minima
max_val = df['Hustota_Mostu'].max()
min_val = df['Hustota_Mostu'].min()
best_spin = df.loc[df['Hustota_Mostu'] == max_val, 'Spin_Uhel'].values[0]

print(f"=== ANALÝZA GEOMETRICKÉ KOHERENCE ===")
print(f"Absolutní špička (Zlatý Spin): {best_spin}°")
print(f"Maximální hustota mostu: {max_val:.4f}")
print(f"Minimální hustota mostu (disonance): {min_val:.4f}")
print(f"Dynamický rozsah rezonance: {max_val/min_val:.1f}x")

# Vizualizace "Zlatého Zámku"
plt.figure(figsize=(10, 6))
plt.plot(df['Spin_Uhel'], df['Hustota_Mostu'], color='#38bdf8', linewidth=2, label='Fázový most')
plt.fill_between(df['Spin_Uhel'], df['Hustota_Mostu'], color='#38bdf8', alpha=0.1)
plt.axvline(x=best_spin, color='#e879f9', linestyle='--', label=f'Resonance Lock ({best_spin}°)')

plt.title("TCD: Geometrický podpis fázového zámku", color='white', fontsize=14)
plt.xlabel("Relativní Spinový úhel [stupně]", color='gray')
plt.ylabel("Hustota fázového mostu", color='gray')
plt.grid(color='white', alpha=0.05)
plt.legend()
plt.tight_layout()
plt.show()
