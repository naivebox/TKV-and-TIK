import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# =============================================================================
# OMNI-ENGINE v9.3 - TOPOLOGICAL RELAXATION (METHANE CH4)
# Tento skript simuluje fázovou repulzi mezi uzly vodíku v orbitalu uhlíku.
# Atomy začínají v náhodných pozicích a mřížka je dynamicky "odstrká"
# do stavu nejmenšího pnutí -> Dokonalý Tetraedr (109.47°).
# =============================================================================

class TopologicalRelaxation:
    def __init__(self, num_nodes=4, radius=10.0):
        self.num_nodes = num_nodes
        self.R = radius
        # Generování náhodných počátečních pozic na sféře
        self.nodes = self._random_sphere_points(num_nodes, radius)
        
        # Nastavení vizualizace
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.patch.set_facecolor('#05050a')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#05050a')
        
    def _random_sphere_points(self, n, r):
        points = []
        for _ in range(n):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.arccos(np.random.uniform(-1, 1))
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            points.append(np.array([x, y, z]))
        return np.array(points)

    def calculate_angles(self):
        """Vypočítá všech 6 úhlů mezi 4 vazbami."""
        angles = []
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                v1 = self.nodes[i] / np.linalg.norm(self.nodes[i])
                v2 = self.nodes[j] / np.linalg.norm(self.nodes[j])
                dot_prod = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.rad2deg(np.arccos(dot_prod))
                angles.append(angle)
        return angles

    def apply_phase_repulsion(self, dt=0.1):
        """
        Simuluje TCD Laplacián. Fázové uzly (vodíky) se navzájem odpuzují,
        protože překryv jejich vln tvoří v mřížce Levelu D destruktivní napětí.
        """
        forces = np.zeros_like(self.nodes)
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j: continue
                
                # Vektor od uzlu j k uzlu i
                diff = self.nodes[i] - self.nodes[j]
                dist = np.linalg.norm(diff)
                
                if dist > 0:
                    # Topologická repulze (čím blíž, tím větší tlak mřížky)
                    force_mag = 100.0 / (dist**2)
                    forces[i] += force_mag * (diff / dist)
                    
        # Aplikace síly a udržení uzlů na oběžné dráze (sférický orbital)
        for i in range(self.num_nodes):
            self.nodes[i] += forces[i] * dt
            # Normalizace zpět na poloměr orbitalu
            self.nodes[i] = (self.nodes[i] / np.linalg.norm(self.nodes[i])) * self.R

    def draw_frame(self, step, angles):
        self.ax.clear()
        self.ax.set_facecolor('#05050a')
        self.ax.axis('off')
        self.ax.set_xlim([-self.R, self.R])
        self.ax.set_ylim([-self.R, self.R])
        self.ax.set_zlim([-self.R, self.R])
        
        # Vykreslení centrálního Uhlíku
        self.ax.scatter([0], [0], [0], color='#e879f9', s=300, label='Uhlík-12')
        
        # Vykreslení Vodíků a vazeb
        for i in range(self.num_nodes):
            # Vodík
            self.ax.scatter(*self.nodes[i], color='#38bdf8', s=150)
            # Kovalentní vazba (fázový tunel)
            self.ax.plot([0, self.nodes[i][0]], [0, self.nodes[i][1]], [0, self.nodes[i][2]], color='white', alpha=0.5, lw=2)
            
        # Výpočet odchylky od dokonalého tetraedru (109.47°)
        avg_angle = np.mean(angles)
        deviation = abs(avg_angle - 109.4712)
        
        title = f"TCD Dynamická Relaxace Mřížky (Metan $CH_4$)\nIterace: {step} | Průměrný úhel vazby: {avg_angle:.2f}°"
        
        if deviation < 0.1:
            title += "\n[!!!] GEOMETRICKÝ ZÁMEK DOSAŽEN [!!!]"
            title_color = '#deff9a'
        else:
            title += f"\nFázový stres... Odchylka: {deviation:.2f}°"
            title_color = 'cyan'
            
        self.ax.set_title(title, color=title_color, pad=20, fontsize=12)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self, max_steps=150):
        print("[*] TCD CH4 RELAXAČNÍ MODUL v9.3")
        print("[*] 4 fázové uzly (Vodíky) vstříknuty v náhodném chaosu.")
        print("[*] Sledujte, jak je mřížka sama vytlačí do tetraedru...")
        
        for step in range(max_steps):
            angles = self.calculate_angles()
            self.draw_frame(step, angles)
            
            # Pokud je odchylka minimální, zastavíme simulaci
            if abs(np.mean(angles) - 109.4712) < 0.01 and np.std(angles) < 0.1:
                self.draw_frame(step, angles)
                print(f"\n[OK] Absolutní topologická homeostáza nalezena v kroku {step}!")
                print(f"[OK] Finální úhel mřížky: {np.mean(angles):.4f}°")
                break
                
            self.apply_phase_repulsion(dt=0.2)
            time.sleep(0.05)
            
        print("\nHotovo. Můžete zavřít okno grafu.")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    sim = TopologicalRelaxation()
    sim.run()
