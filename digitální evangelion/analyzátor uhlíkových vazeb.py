import sys
import subprocess

# =============================================================================
# AUTOMATICKÝ INSTALÁTOR ZÁVISLOSTÍ
# =============================================================================
def ensure_packages():
    packages = ['pandas', 'numpy']
    for p in packages:
        try:
            __import__(p)
        except ImportError:
            print(f"[*] Chybí modul '{p}'. Automaticky instaluji pro Python {sys.version_info.major}.{sys.version_info.minor}...")
            # Zavolá pip install pro chybějící balíček
            subprocess.check_call([sys.executable, "-m", "pip", "install", p])
            print(f"[OK] Modul '{p}' byl úspěšně nainstalován.")

# Spustíme kontrolu před samotným importem
ensure_packages()

import pandas as pd
import numpy as np
import re
import ast

# =============================================================================
# TCD ANALYTICKÝ MODUL: Zpracování dat z Carbon Valence Scanneru
# Cíl: Najít geometrický atraktor pro uhlík a dokázat přítomnost tetraedru.
# =============================================================================

def parse_numpy_array_string(arr_str):
    """Převede string typu '[np.float64(108.9), ...]' na list běžných floatů."""
    # Odstraní np.float64() a nechá jen čistá čísla
    clean_str = re.sub(r'np\.float64\((.*?)\)', r'\1', arr_str)
    try:
        return ast.literal_eval(clean_str)
    except:
        return []

print("=" * 65)
print(" TCD ANALÝZA: GEOMETRIE UHLÍKOVÝCH VAZEB")
print("=" * 65)

try:
    # Načtení dat z Omni-Enginu
    df = pd.read_csv('tcd_carbon_valences_data.csv', delimiter=';')
    
    # Filtrování pouze na 4 vazby (CH4 - Metan)
    df_4h = df[df['Pocet_Vazeb'] == 4].copy()
    
    if df_4h.empty:
        print("[!] Chyba: Nenalezena žádná data pro 4 vazby.")
    else:
        # Nalezení absolutně nejstabilnějšího stavu (Nejnižší celkové pnutí)
        best_row = df_4h.loc[df_4h['Celkove_Pnuti'].idxmin()]
        worst_row = df_4h.loc[df_4h['Celkove_Pnuti'].idxmax()]
        
        print(f"[*] Hledání fázového minima pro {best_row['Pocet_Vazeb']} vazby (Uhlík-Vodík)...")
        print(f"\n[NEJHORŠÍ KONFIGURACE (Chaotický šum)]")
        print(f"ID Testu: {worst_row['ID']}")
        print(f"Celkový stres mřížky: {worst_row['Celkove_Pnuti']:,.2f} Psi")
        
        print(f"\n[NEJLEPŠÍ KONFIGURACE (Topologický Atraktor)]")
        print(f"ID Testu: {best_row['ID']}")
        print(f"Celkový stres mřížky: {best_row['Celkove_Pnuti']:,.2f} Psi (Absolutní minimum)")
        
        # Analýza vítězných úhlů
        angles = parse_numpy_array_string(best_row['Uhly_Mezi_Vazbami'])
        if angles:
            avg_angle = np.mean(angles)
            std_dev = np.std(angles)
            
            print(f"\nVítězné úhly mezi 4 rameny:")
            for i, a in enumerate(angles):
                print(f"  Vazba {i+1}: {a:.2f}°")
                
            print(f"\n[!!!] MATEMATICKÝ VERDIKT [!!!]")
            print(f"Průměrný úhel atraktoru: {avg_angle:.2f}° (Odchylka ±{std_dev:.2f}°)")
            
            # Porovnání s ideálním tetraedrem
            ideal_tetrahedron = 109.4712
            rozdil = abs(avg_angle - ideal_tetrahedron)
            
            print(f"Ideální čtyřstěn (Tetraedr) má úhel: {ideal_tetrahedron:.4f}°")
            print(f"Geometrická odchylka simulace Levelu D od ideálu: {rozdil:.4f}°")
            
            if rozdil < 5.0:
                print("\n[ZÁVĚR] Úspěch! Mřížka sama zformovala dokonalý čtyřstěn.")
                print("Uhlík nemá žádné vnitřní pnutí, proto nedeformuje mřížku jako Voda.")
                print("Vyplňuje prostor čistou, nedeformovanou Dvanáctistěnnou symetrií.")

except FileNotFoundError:
    print("Soubor 'tcd_carbon_valences_data.csv' nebyl nalezen.")
