import sys
import subprocess

def install_and_import(package, import_name=None):
    """Pokusí se importovat modul. Pokud neexistuje, nainstaluje ho."""
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
    except ImportError:
        print(f"[*] Modul '{package}' nenalezen. Instaluji...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
            print(f"[+] '{package}' úspěšně nainstalován.")
        except Exception as e:
            print(f"[!] CHYBA instalace '{package}': {e}")
            sys.exit(1)

# --- 1. KONTROLA ZÁKLADNÍCH KNIHOVEN (Bez scikit-image!) ---
print("[*] Kontroluji základní knihovny...")
install_and_import('numpy')
install_and_import('matplotlib')

import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# TKV VIZUÁLNÍ SONDA: HOLOGRAFICKÝ RENDER PROTONU
# =====================================================================

# --- 2. NAČTENÍ DAT ---
try:
    print("\n[*] Načítám data z Levelu D (z disku)...")
    quarks_amp = np.load("proton_quarks_amplitude.npy")
    gluon_tube = np.load("proton_gluon_tube.npy")
    N = quarks_amp.shape[0]
except FileNotFoundError:
    print("[!] CHYBA: Soubory .npy chybí. Spusť nejprve exportní skript.")
    sys.exit()

# --- 3. PŘÍPRAVA HOLOGRAFICKÝCH DAT (Point Cloud) ---
print("[*] Zpracovávám 3D hologram...")

# Filtrujeme pouze pixely, které mají dostatečnou sílu (odstraníme prázdné vakuum)
# Experimentuj s těmito čísly (0.01 pro gluony a 2.5 pro kvarky)
mask_gluons = (gluon_tube > 0.005)
mask_quarks = (quarks_amp > 2.5)

# Abychom nekreslili gluony uvnitř kvarků (ušetříme výkon)
mask_gluons = mask_gluons & ~mask_quarks

# Získáme souřadnice [Z, Y, X] pro každý bod
z_g, y_g, x_g = np.where(mask_gluons)
z_q, y_q, x_q = np.where(mask_quarks)

# --- 4. NASTAVENÍ GRAFIKY ---
fig = plt.figure(figsize=(10, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black') # Vesmírné pozadí

ax.set_title("TKV Proton: Hologram (Kvarky a Gluonová trubice)", color='white')
ax.set_xlim(0, N); ax.set_ylim(0, N); ax.set_zlim(0, N)

# Skrytí os pro hezčí čistý pohled
ax.axis('off')

# --- 5. VYKRESLENÍ ---
print(f"-> Vykresluji Gluonovou trubici ({len(x_g)} částic)...")
# c='cyan' je barva, alpha=0.05 je průhlednost, s=5 je velikost bodu
ax.scatter(x_g, y_g, z_g, c='cyan', alpha=0.05, s=2, marker='o')

print(f"-> Vykresluji Kvarky ({len(x_q)} částic)...")
ax.scatter(x_q, y_q, z_q, c='red', alpha=0.8, s=15, marker='o')

# Nastavení úhlu kamery
ax.view_init(elev=20, azim=45)
print("[*] Vykresleno. Zobrazuji interaktivní okno!")

plt.tight_layout()
plt.show()
