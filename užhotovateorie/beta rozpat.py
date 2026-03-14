import time
import math
import numpy as np

# =============================================================================
# OMNI-ENGINE: ALGORITMUS GENEZE - FÁZE 2
# Úkol: Simulace fázové inverze a stabilizace Neutronu (Beta rozpad naruby).
# =============================================================================

class TCDLatticePhase2:
    def __init__(self, resolution=64):
        self.res = resolution
        self.PHI = 1.61803398875
        self.PROTON_TARGET = 275.4258  # Homeostáza z Knihy II
        self.NEUTRON_TARGET = 278.1560 # Odhadovaná hustota Neutronu (p + e + pnutí)
        print(f"[*] Inicializuji mřížku Level D pro Fázi 2 ({resolution}^3 buněk)...")

    def spawn_proton(self):
        """Vyvolá základní Trefoil uzel."""
        print("[*] Stav: Detekován Proton (Trefoil uzel) v homeostáze.")
        return {"type": "proton", "density": self.PROTON_TARGET, "lock": True}

    def initiate_topological_inversion(self, particle):
        """
        Simuluje pohlcení fázové slupky a transformaci v Neutron.
        V TCD jde o 'přehuštění' uzlu nad limit Harmoniky.
        """
        print("[!] ZAHÁJENÍ FÁZOVÉ INVERZE (Slabá interakce)")
        print("-" * 50)
        
        # Simulace pohlcení elektronu (injekce 0.511 MeV ekvivalentu pnutí)
        current_density = particle["density"]
        steps = 8
        
        for i in range(1, steps + 1):
            # Nárůst pnutí směrem k neutronovému limitu
            delta_psi = (self.NEUTRON_TARGET - self.PROTON_TARGET) / steps
            current_density += delta_psi
            
            # Výpočet fázového pnutí (vrtule se začíná 'zadrhávat' v mřížce)
            tension = math.tanh(current_density / self.PROTON_TARGET)
            
            print(f"Krok {i:02d}: Lokální hustota: {current_density:8.4f} | Fázové pnutí: {tension:.6f}")
            time.sleep(0.5)

        # Uzamčení neutronu
        particle["type"] = "neutron"
        particle["density"] = current_density
        particle["lock"] = False # Neutron je v TCD vně mřížky metastabilní
        
        print("-" * 50)
        print(f"[OK] Vznikl NEUTRON (Metastabilní uzel).")
        return particle

    def calculate_energy_defect(self, neutron):
        """Vypočítá energii 'uskladněnou' v uzlu (rozdíl Psi)."""
        # Převodní poměr z Kapitoly 1 Knihy III: 1 unit = 3.406 MeV
        energy_unit = 3.406
        mass_diff = neutron["density"] - self.PROTON_TARGET
        energy_mev = mass_diff * energy_unit
        
        print(f"[INFO] Uskladněná fázová energie: {energy_mev:.4f} MeV")
        print(f"[INFO] Uzel vykazuje vnitřní asymetrii (příprava na Sifon).")
        return energy_mev

if __name__ == "__main__":
    engine = TCDLatticePhase2()
    
    # KROK 1: Vezmeme náš zkalibrovaný proton
    p_node = engine.spawn_proton()
    
    # KROK 2: Vynutíme pohlcení fáze (Vznik neutronu)
    n_node = engine.initiate_topological_inversion(p_node)
    
    # KROK 3: Záznam dat pro Fázi 3 (Deuterium)
    engine.calculate_energy_defect(n_node)
    
    print("\n[VÝSLEDEK FÁZE 2]:")
    print(f"Máme v paměti stabilní 'přetížený' uzel.")
    print("Mřížka je připravena na spojení Proton + Neutron.")
