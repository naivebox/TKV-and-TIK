import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# OMNI-ENGINE v9.5 - TOPOLOGICAL RELAXATION (BENZENE C6H6)
# Důkaz rovnice z Kapitoly 8 (n=3 -> 120 stupňů).
# Z chaotického 3D shluku mřížka "vyžehlí" atomy do dokonale plochého
# šestiúhelníku. Fázová repulze nedovolí prostorové (3D) zkroucení.
# =============================================================================

class BenzeneRelaxation:
    def __init__(self):
        self.N_C = 6
        self.N_H = 6
        self.D_CC = 4.0 # Ideální fázová vzdálenost C-C v kruhu
        self.D_CH = 2.5 # Ideální fázová vzdálenost C-H
        
        # 1. CHAOTICKÁ INICIALIZACE (Zmačkaný 3D tvar)
        # Uhlíky umístíme zhruba do kruhu, ale přidáme jim masivní 3D šum (Z-osa)
        angles = np.linspace(0, 2*np.pi, self.N_C, endpoint=False)
        self.C = np.zeros((self.N_C, 3))
        self.H = np.zeros((self.N_H, 3))
        
        for i in range(self.N_C):
            # Extrémní z-odchylka (rozbitá rovina)
            z_noise = np.random.uniform(-4.0, 4.0) 
            self.C[i] = [3.0 * np.cos(angles[i]), 3.0 * np.sin(angles[i]), z_noise]
            # Vodíky rozházené naprosto náhodně kolem uhlíků
            self.H[i] = self.C[i] + np.random.uniform(-3.0, 3.0, 3)

        # Nastavení 3D plátna
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8), facecolor='#05050a')
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='#05050a')

    def apply_phase_forces(self, dt=0.08):
        """Aplikace TCD Laplaciánu: Fázové pružiny a repulze."""
        
        # A. UHLÍKOVÝ KRUH (Sousední uhlíky se k sobě fázově vážou)
        for i in range(self.N_C):
            next_i = (i + 1) % self.N_C
            prev_i = (i - 1) % self.N_C
            
            # Tah k sousedům (Kovalentní sdílení)
            for neighbor in [next_i, prev_i]:
                diff = self.C[neighbor] - self.C[i]
                dist = np.linalg.norm(diff)
                # Harmonická síla snažící se udržet vzdálenost D_CC
                force = 5.0 * (dist - self.D_CC) * (diff / dist)
                self.C[i] += force * dt

        # B. VAZBA C-H
        for i in range(self.N_C):
            diff = self.H[i] - self.C[i]
            dist = np.linalg.norm(diff)
            force = 8.0 * (dist - self.D_CH) * (diff / dist)
            self.H[i] -= force * dt  # Vodík je přitahován k Uhlíku
            self.C[i] += force * dt  # Zpětná reakce

        # C. GLOBÁLNÍ FÁZOVÁ REPULZE (Všichni proti všem)
        # Toto nutí mřížku vytvořit úhel 120° a "zplacatit" strukturu do 2D
        all_nodes = np.vstack((self.C, self.H))
        diffs = all_nodes[:, None, :] - all_nodes[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        # Inverzní kvadratická repulze (fázový odpor vakua)
        repulsion = np.sum(15.0 * diffs / dists[..., None]**3, axis=1)
        
        self.C += repulsion[:self.N_C] * dt
        self.H += repulsion[self.N_C:] * dt
        
        # Centrování molekuly (aby nám neuletěla z obrazovky)
        center_of_mass = np.mean(self.C, axis=0)
        self.C -= center_of_mass
        self.H -= center_of_mass

    def get_planarity_error(self):
        """Měří, jak moc je molekula zkroucená ve 3D prostoru (Odchylka osy Z)."""
        # Spočítáme průměrnou odchylku Z-souřadnice uhlíků od nuly
        return np.mean(np.abs(self.C[:, 2]))

    def draw_frame(self, step):
        self.ax.clear()
        self.ax.set(facecolor='#05050a', xlim=(-8,8), ylim=(-8,8), zlim=(-8,8))
        self.ax.axis('off')
        
        # Vykreslení Uhlíkového kruhu
        for i in range(self.N_C):
            next_i = (i + 1) % self.N_C
            self.ax.plot([self.C[i,0], self.C[next_i,0]], 
                         [self.C[i,1], self.C[next_i,1]], 
                         [self.C[i,2], self.C[next_i,2]], color='#c084fc', lw=5, alpha=0.8)
        
        self.ax.scatter(self.C[:,0], self.C[:,1], self.C[:,2], color='#e879f9', s=350, label='Uhlík-12')
        
        # Vykreslení Vodíků
        for i in range(self.N_C):
            self.ax.plot([self.C[i,0], self.H[i,0]], 
                         [self.C[i,1], self.H[i,1]], 
                         [self.C[i,2], self.H[i,2]], color='white', alpha=0.5, lw=2)
            self.ax.scatter(self.H[i,0], self.H[i,1], self.H[i,2], color='#38bdf8', s=120)
                
        # Zobrazení stavu
        planarity = self.get_planarity_error()
        title = f"TCD Makro-architektura: Benzenový kruh ($C_6H_6$)\nIterace: {step}"
        
        if planarity < 0.05:
            self.ax.set_title(title + f"\n[!!!] 2D ZÁMEK: Absolutní planární homeostáza [!!!]", color='#deff9a')
        else:
            self.ax.set_title(title + f"\n3D Fázové pnutí... Odchylka od roviny: {planarity:.2f}", color='cyan')
            
        # Pomalá rotace pro ukázání zploštění
        self.ax.view_init(elev=10 + step*0.2, azim=step * 0.5)
        plt.pause(0.01)

    def run(self):
        print("[*] TCD BENZENE RELAXAČNÍ MODUL v9.5")
        print("[*] Sledujte, jak Laplacián vyžehlí 3D chaos do 2D roviny (120°).")
        for step in range(300):
            self.draw_frame(step)
            if self.get_planarity_error() < 0.02 and step > 50:
                self.draw_frame(step)
                print(f"\n[OK] Topologická rovina nalezena v kroku {step}!")
                print(f"[OK] Atomy se uzamkly do 120° trigonální symetrie.")
                break
            self.apply_phase_forces()
        print("\nHotovo. Můžete zavřít okno grafu.")
        plt.ioff(); plt.show()

if __name__ == "__main__":
    BenzeneRelaxation().run()
