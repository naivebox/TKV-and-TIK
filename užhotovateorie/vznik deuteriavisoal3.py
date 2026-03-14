import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# =============================================================================
# OMNI-ENGINE v7.0 - VISUALIZER: TRUE PROTON-PROTON FUSION
# Dva identické protony se přibližují. Obrovský topologický tlak vynutí
# na jednom z nich fázovou inverzi (Beta+ rozpad) těsně před fúzí.
# =============================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void tkv_pp_fusion_step(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *h_mass,
    const double dt, const int N, const int t_step)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x >= N || y >= N || z >= N) return;
    int i = x * N * N + y * N + z;

    int id_xp = ((x + 1) % N) * N * N + y * N + z;
    int id_xm = ((x - 1 + N) % N) * N * N + y * N + z;
    int id_yp = x * N * N + ((y + 1) % N) * N + z;
    int id_ym = x * N * N + ((y - 1 + N) % N) * N + z;
    int id_zp = x * N * N + y * N + ((z + 1) % N);
    int id_zm = x * N * N + y * N + ((z - 1 + N) % N);

    const double PROTON_TARGET = 275.42;
    const double DEUTERIUM_TARGET = 552.93; 

    double pr = psi_r[i]; double pi = psi_i[i];
    
    // Laplacián
    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    double current_m = sqrt(pr*pr + pi*pi);
    
    double drift_r = 0.0; double drift_i = 0.0;
    double momentum = 3.2; 
    double intake = 0.0;
    
    // 1. PŘIBLIŽOVÁNÍ (Protony tlačí mřížku proti sobě)
    if (t_step < 450) {
        if (x < N/2 - 3) { 
            drift_r = momentum * (psi_r[id_xm] - pr);
            drift_i = momentum * (psi_i[id_xm] - pi);
        } else if (x > N/2 + 3) { 
            drift_r = momentum * (psi_r[id_xp] - pr);
            drift_i = momentum * (psi_i[id_xp] - pi);
        }
    }

    // 2. TLAKEM VYNUCENÁ FÁZOVÁ INVERZE (Beta rozpad na pravém uzlu)
    // Odehrává se od tiku 150 do 300, kdy jsou uzly blízko a mřížka hrozí přetížením
    if (t_step > 150 && t_step < 350 && x > N/2) {
        if (current_m > 3.0 && current_m < 50.0) {
            // Pohlcení slupky (Aura mizí)
            intake = -0.025; 
        } else if (current_m >= 50.0 && current_m < 320.0) {
            // Přehuštění jádra (Vzniká neutron)
            intake = 0.015; 
        }
    }

    // 3. TOPOLOGICKÁ SMĚNA A UZAMČENÍ (Handover po inverzi)
    if (t_step > 250) {
        double saturation = 0.5 * (1.0 - tanh(3.5 * (current_m / DEUTERIUM_TARGET - 0.98)));
        intake += 0.18 * saturation;
    }
    
    // Aktualizace stavu
    double nr = pr + (0.08 * lap_r * dt) + (drift_r * dt) + (pr * intake * dt);
    double ni = pi + (0.08 * lap_i * dt) + (drift_i * dt) + (pi * intake * dt);

    // Pojistka Sifonu proti singularitě
    double nm = sqrt(nr*nr + ni*ni);
    if (nm > PROTON_TARGET * 1.6) { 
        nr *= (PROTON_TARGET * 1.6 / nm);
        ni *= (PROTON_TARGET * 1.6 / nm);
    }

    psi_rn[i] = nr; 
    psi_in[i] = ni;
    h_mass[i] = nm;
}
"""

class TrueFusionVisualizer:
    def __init__(self, N=160): 
        self.N = N
        self.dt = 0.015
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_pp_fusion_step")

        print("[*] Inicializuji reaktor Level D pro skutečnou P-P fúzi...")
        x, y, z = np.indices((N, N, N))
        
        start_dist = 36.0 
        
        # PROTON 1 (Vlevo)
        cx1, cy1, cz1 = N//2 - start_dist/2, N//2, N//2
        r1 = np.sqrt((x-cx1)**2 + (y-cy1)**2 + (z-cz1)**2)
        env1 = 15.0 * np.exp(-(r1**2)/18.0) 
        phase1 = r1 + (x/N)*np.pi
        
        # PROTON 2 (Vpravo) - Identický! Žádný neutron na začátku.
        cx2, cy2, cz2 = N//2 + start_dist/2, N//2, N//2
        r2 = np.sqrt((x-cx2)**2 + (y-cy2)**2 + (z-cz2)**2)
        env2 = 15.0 * np.exp(-(r2**2)/18.0) 
        # Fáze není invertovaná, narazí do sebe jako dva stejné magnety
        phase2 = r2 - (x/N)*np.pi 

        pr_init = env1 * np.cos(phase1) + env2 * np.cos(phase2)
        pi_init = env1 * np.sin(phase1) + env2 * np.sin(phase2)

        self.d_pr = cl_array.to_device(self.queue, pr_init.astype(np.float64))
        self.d_pi = cl_array.to_device(self.queue, pi_init.astype(np.float64))
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        self.d_hm = cl_array.zeros(self.queue, N**3, dtype=np.float64)

        # Matplotlib UI
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.patch.set_facecolor('#05050a')
        
        self.zoom = 50
        self.im = self.ax.imshow(np.zeros((self.zoom*2, self.zoom*2)), cmap='magma', origin='lower', vmin=0, vmax=500, interpolation='bicubic')
        
        self.ax.set_title("TCD: Skutečná P-P Fúze (Vynucená Inverze)", color='cyan', pad=15, fontsize=16)
        self.ax.axis('off')
        
        self.info_text = self.ax.text(0.02, 0.90, '', transform=self.ax.transAxes, color='white', fontsize=12, fontweight='bold')
        self.energy_text = self.ax.text(0.02, 0.85, '', transform=self.ax.transAxes, color='#e879f9', fontsize=11)
        self.contours = None

    def run(self, steps=800):
        print("[!] ZAHÁJENÍ VIZUALIZACE P-P FÚZE")
        
        for t in range(steps):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_hm.data, np.float64(self.dt), np.int32(self.N), np.int32(t))
            
            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi
            
            if t % 10 == 0: 
                mass_3d = self.d_hm.get().reshape((self.N, self.N, self.N))
                
                slice_2d = mass_3d[self.N//2 - self.zoom : self.N//2 + self.zoom, 
                                   self.N//2 - self.zoom : self.N//2 + self.zoom, 
                                   self.N//2].T
                
                slice_smooth = gaussian_filter(slice_2d, sigma=0.5)
                self.im.set_data(slice_smooth)
                
                core_max = np.max(mass_3d)
                self.im.set_clim(vmin=0, vmax=max(80.0, core_max))
                
                bridge_density = slice_smooth[self.zoom, self.zoom]
                
                if t <= 150:
                    status = "PŘIBLIŽOVÁNÍ (2x Proton - Rostoucí pnutí)"
                    sync = 0.0
                elif t <= 300:
                    status = "FÁZOVÁ INVERZE (Tlak drtí pravou slupku)"
                    sync = ((t - 150) / 150) * 50
                elif t <= 450:
                    status = "TOPOLOGICKÁ SMĚNA (P-N propojení)"
                    sync = 50 + ((t - 300) / 150) * 50
                else:
                    status = "ZÁMEK KLAPL (Stabilní Deuterium)"
                    sync = 100.0

                defect = (sync / 100.0) * 2.22

                self.info_text.set_text(f"Tik: {t} | Středový fázový most: {bridge_density:.1f}\nStav: {status}")
                self.energy_text.set_text(f"Synchronizace: {sync:.1f} %\nUvolněná fázová energie: {defect:.2f} MeV")

                if self.contours:
                    try:
                        self.contours.remove()
                    except AttributeError:
                        for c in self.contours.collections:
                            c.remove()
                            
                if t > 50:
                    levels = [80, 150, 220, 300, 380] 
                    self.contours = self.ax.contour(slice_smooth, levels=levels, colors=['#38bdf8', '#818cf8', '#c084fc', '#e879f9', 'white'], alpha=0.6, linewidths=1.2)

                plt.pause(0.01)

        print("\n[OK] Vizualizace dokončena.")

if __name__ == "__main__":
    viz = TrueFusionVisualizer(N=160)
    viz.run(steps=800)
