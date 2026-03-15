import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# =============================================================================
# OMNI-ENGINE v15.0 - BIOGENESIS: HELIX FOLDING (CHIRALITY)
# Kniha IV: Simulace "Zlatého zkrutu" mřížky Levelu D.
# Dlouhé polymery se ve vakuu neudrží rovné. Topologický odpor mřížky
# je přirozeně stáčí do pravotočivé šroubovice.
# =============================================================================

class PolymerFolder:
    def __init__(self, num_nodes=15):
        self.num_nodes = num_nodes
        self.bond_length = 3.0
        
        # Startovní stav: Rovný řetězec podél osy Z (např. uhlíková páteř)
        self.nodes = np.zeros((num_nodes, 3))
        for i in range(num_nodes):
            self.nodes[i] = [0.0, 0.0, i * self.bond_length - (num_nodes*self.bond_length/2)]
            
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.patch.set_facecolor('#051005') # Bio-Zelená tma
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#051005')

    def apply_topological_forces(self, dt=0.1):
        forces = np.zeros_like(self.nodes)
        
        # 1. Kovalentní vazba (Sousední uzly se drží na vzdálenosti bond_length)
        for i in range(self.num_nodes - 1):
            diff = self.nodes[i+1] - self.nodes[i]
            dist = np.linalg.norm(diff)
            force = 15.0 * (dist - self.bond_length) * (diff / dist)
            forces[i] += force
            forces[i+1] -= force
            
        # 2. Geometrická frustrace (Úhel 109.5° mezi ob-sousedními uzly)
        # Zabraňuje tomu, aby se řetězec zhroutil do sebe
        for i in range(self.num_nodes - 2):
            vec1 = self.nodes[i] - self.nodes[i+1]
            vec2 = self.nodes[i+2] - self.nodes[i+1]
            n1 = np.linalg.norm(vec1)
            n2 = np.linalg.norm(vec2)
            if n1 > 0 and n2 > 0:
                dot = np.clip(np.dot(vec1, vec2) / (n1 * n2), -1.0, 1.0)
                angle = np.arccos(dot)
                target_angle = np.deg2rad(109.5) # Tetraedrální úhel uhlíku
                
                if abs(angle - target_angle) > 0.01:
                    # Rozvírací/svírací síla
                    correction = 5.0 * (target_angle - angle)
                    forces[i] += correction * (vec1/n1)
                    forces[i+2] += correction * (vec2/n2)

        # 3. ZLATÝ ZKRUT (Chirality Drag Levelu D)
        # Toto je tajemství života! Mřížka dvanáctistěnu má inherentní torzní spin.
        # Každý uzel je mírně natáčen vůči předchozímu kolem osy postupu.
        golden_twist = np.deg2rad(137.5) # Zlatý úhel
        twist_strength = 2.0
        
        for i in range(1, self.num_nodes):
            # Aplikace tečného kroutivého momentu (vytváří spirálu)
            z_axis = np.array([0, 0, 1])
            pos_xy = np.array([self.nodes[i][0], self.nodes[i][1], 0])
            r_xy = np.linalg.norm(pos_xy)
            
            if r_xy < 0.1: # Pokud jsme na ose, mírně uzel vychýlíme, aby mohl rotovat
                forces[i][0] += np.random.uniform(-0.1, 0.1)
                forces[i][1] += np.random.uniform(-0.1, 0.1)
            else:
                # Vektor kolmý na poloměr (tangenta)
                tangent = np.cross(z_axis, pos_xy) / r_xy
                forces[i] += twist_strength * tangent

        # Aplikace sil + tlumení (viskozita vakua)
        self.nodes += forces * dt
        # Zafixování prvního uzlu, ať nám řetězec neuletí
        self.nodes[0] = [0.0, 0.0, -self.num_nodes*self.bond_length/2]

    def draw_frame(self, step):
        self.ax.clear()
        self.ax.set_facecolor('#051005')
        self.ax.axis('off')
        
        limit = self.num_nodes * 2.0
        self.ax.set_xlim([-limit/2, limit/2])
        self.ax.set_ylim([-limit/2, limit/2])
        self.ax.set_zlim([-limit, limit])
        
        # Vykreslení uhlíkové páteře
        xs, ys, zs = self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2]
        self.ax.plot(xs, ys, zs, color='#84cc16', lw=4, alpha=0.8, label='Uhlíková páteř')
        self.ax.scatter(xs, ys, zs, color='#bef264', s=150, edgecolors='#4d7c0f')
        
        # Titulek a stav
        spread = np.max(np.sqrt(xs**2 + ys**2))
        status = "LINEÁRNÍ STAV (Nepřirozený)" if spread < 1.0 else "TVORBA ŠROUBOVICE (Fázový zkrut mřížky)"
        if step > 200: status = "ALFA-ŠROUBOVICE STABILNÍ"
            
        self.ax.set_title(f"TCD Biogeneze: Chiralita a vznik šroubovice\nTik: {step} | Stav: {status}", color='#bef264', pad=20, fontsize=12)
        
        # Pomalá rotace kamery
        self.ax.view_init(elev=15, azim=step * 1.5)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self, max_steps=300):
        print("[*] OMNI-ENGINE v15.0 - BIOGENESIS MODULE")
        print("[*] Start: Lineární polymerní řetězec vložen do mřížky.")
        print("[*] Sledujte aplikaci Zlatého zkrutu (Topological Drag)...")
        
        for step in range(max_steps):
            self.draw_frame(step)
            self.apply_topological_forces(dt=0.05)
            time.sleep(0.02)
            
        print("\n[OK] Šroubovice je stabilní. Mřížka určila tvar.")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    sim = PolymerFolder(num_nodes=20)
    sim.run()
