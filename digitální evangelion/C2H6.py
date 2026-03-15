import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# OMNI-ENGINE v9.4.1 - TOPOLOGICAL RELAXATION (ETHANE C2H6) - ZJEDNODUŠENO
# Využití maticové vektorizace Numpy pro čistší a rychlejší výpočet.
# Fyzika zůstává stejná, ale kód je o 50 % kratší a elegantnější.
# =============================================================================

class EthaneRelaxation:
    def __init__(self):
        self.C_dist, self.H_dist = 6.0, 4.0
        self.C1, self.C2 = np.array([-self.C_dist/2, 0, 0]), np.array([self.C_dist/2, 0, 0])
        
        # Inicializace náhodných pozic vodíků okolo uhlíků
        self.H1 = self._random_sphere_points(3, self.H_dist) + self.C1
        self.H2 = self._random_sphere_points(3, self.H_dist) + self.C2
        
        # Nastavení 3D plátna
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8), facecolor='#05050a')
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='#05050a')

    def _random_sphere_points(self, n, r):
        """Vygeneruje N bodů na sféře pomocí maticových operací."""
        phi = np.arccos(np.random.uniform(-1, 1, n))
        theta = np.random.uniform(0, 2*np.pi, n)
        return r * np.column_stack((np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)))

    def apply_phase_repulsion(self, dt=0.15):
        """Vektorizovaný výpočet TCD Laplaciánu (odpuzování všech H navzájem)."""
        all_H = np.vstack((self.H1, self.H2))
        
        # Matice rozdílů a vzdáleností všech bodů proti všem (Broadcasting)
        diffs = all_H[:, None, :] - all_H[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf) # Prevence dělení nulou vůči sobě samému
        
        # Výpočet síly: F = 50 / dist^2 * (směrový vektor)
        forces = np.sum(50.0 * diffs / dists[..., None]**3, axis=1)
        all_H += forces * dt
        
        # Přitáhnutí vodíků zpět na přesnou oběžnou dráhu (Kovalentní vazba C-H)
        for C, H_group in zip([self.C1, self.C2], [all_H[:3], all_H[3:]]):
            vecs = H_group - C
            H_group[:] = C + vecs * (self.H_dist / np.linalg.norm(vecs, axis=1)[:, None])
            
        self.H1, self.H2 = all_H[:3], all_H[3:]

    def get_dihedral_angle(self):
        """Měří torzní úhel pomocí průmětu do roviny YZ (pohled podél C-C osy)."""
        v1 = self.H1[:, 1:] / np.linalg.norm(self.H1[:, 1:], axis=1)[:, None]
        v2 = self.H2[:, 1:] / np.linalg.norm(self.H2[:, 1:], axis=1)[:, None]
        # Maticové násobení pro nalezení minimálního úhlu mezi všemi H1 a H2
        return np.min(np.rad2deg(np.arccos(np.clip(v1 @ v2.T, -1.0, 1.0))))

    def draw_frame(self, step):
        self.ax.clear()
        self.ax.set(facecolor='#05050a', xlim=(-10,10), ylim=(-10,10), zlim=(-10,10))
        self.ax.axis('off')
        
        # Vykreslení uhlíků a centrální vazby
        self.ax.plot(*zip(self.C1, self.C2), color='#e879f9', lw=4, alpha=0.8)
        self.ax.scatter(*zip(self.C1, self.C2), color='#e879f9', s=300)
        
        # Vykreslení vodíků
        for C, H_group in zip([self.C1, self.C2], [self.H1, self.H2]):
            self.ax.scatter(*H_group.T, color='#38bdf8', s=100)
            for H in H_group:
                self.ax.plot(*zip(C, H), color='white', alpha=0.5, lw=2)
                
        # Rotace a zobrazení stavu
        offset = self.get_dihedral_angle()
        title = f"TCD Makro-architektura (Etan $C_2H_6$)\nIterace: {step}"
        
        if abs(offset - 60.0) < 1.0:
            self.ax.set_title(title + f"\n[!!!] TORZNÍ ZÁMEK: Střídavá konf. (60°) [!!!]", color='#deff9a')
        else:
            self.ax.set_title(title + f"\nTorzní pnutí: {offset:.1f}°", color='cyan')
            
        self.ax.view_init(elev=20, azim=step % 360)
        plt.pause(0.01)

    def run(self):
        print("[*] TCD ETHANE RELAXAČNÍ MODUL v9.4.1 (Zjednodušený Numpy kód)")
        for step in range(200):
            self.draw_frame(step)
            if abs(self.get_dihedral_angle() - 60.0) < 0.2:
                self.draw_frame(step)
                print(f"\n[OK] Topologická homeostáza nalezena v kroku {step}!")
                break
            self.apply_phase_repulsion()
        print("\nHotovo. Můžete zavřít okno grafu.")
        plt.ioff(); plt.show()

if __name__ == "__main__":
    EthaneRelaxation().run()
