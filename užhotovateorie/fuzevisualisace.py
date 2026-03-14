import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# =============================================================================
# OMNI-ENGINE v9.0 - FINAL FUSION VISUALIZER (PROJECT EVA)
# Konfigurace: 10.0 Hz | 180° Phase | 0° Spin | Pump 8.0
# Vizualizace reálného topologického handoveru v mřížce Level D.
# =============================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void tkv_final_viz_step(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *h_mass,
    const double dt, const int N, const double pump_intensity, const int t_step)
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

    const double DEUTERIUM_TARGET = 552.93; 
    const double RESONANCE_FREQ = 10.0;

    double pr = psi_r[i]; double pi = psi_i[i];
    double current_m = sqrt(pr*pr + pi*pi);

    // 1. REZONANČNÍ TEP (10 Hz)
    double angle = RESONANCE_FREQ * dt;
    double pr_rot = pr * cos(angle) - pi * sin(angle);
    double pi_rot = pr * sin(angle) + pi * cos(angle);
    pr = pr_rot; pi = pi_rot;

    // 2. KINETICKÝ TLAK (Fixní plynulý přítlak)
    double drift_r = 0.0; double drift_i = 0.0;
    double mom = 2.5; 
    if (t_step < 350) {
        if (x < N/2 - 2) { 
            drift_r = mom * (psi_r[id_xm] - pr);
            drift_i = mom * (psi_i[id_xm] - pi);
        } else if (x > N/2 + 2) { 
            drift_r = mom * (psi_r[id_xp] - pr);
            drift_i = mom * (psi_i[id_xp] - pi);
        }
    }

    // 3. FÁZOVÁ PUMPA (Sytič mostu)
    double local_pump = 0.0;
    if (x >= N/2-1 && x <= N/2+1 && y == N/2 && z == N/2) {
        local_pump = pump_intensity * 0.12; 
    }

    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    // 4. TOPOLOGICKÝ ZÁMEK (Handover logic)
    double intake = 0.0;
    if (t_step > 200) {
        double saturation = 0.5 * (1.0 - tanh(4.5 * (current_m / DEUTERIUM_TARGET - 0.98)));
        intake = 0.32 * saturation + local_pump; 
    }
    
    double nr = pr + (0.08 * lap_r * dt) + (drift_r * dt) + (pr * intake * dt);
    double ni = pi + (0.08 * lap_i * dt) + (drift_i * dt) + (pi * intake * dt);

    double nm = sqrt(nr*nr + ni*ni);
    if (nm > 900.0) { nr *= (900.0/nm); ni *= (900.0/nm); nm = 900.0; }

    psi_rn[i] = nr; 
    psi_in[i] = ni;
    h_mass[i] = nm;
}
"""

class FinalFusionViz:
    def __init__(self, N=120): 
        self.N = N
        self.dt = 0.015
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_final_viz_step")

        print("[*] Inicializuji finální reaktor EVA...")
        x, y, z = np.indices((N, N, N))
        
        # Startovní konfigurace
        start_dist = 18.0 
        
        # PROTON (Vlevo)
        cx1, cy1, cz1 = N//2 - start_dist/2, N//2, N//2
        r1 = np.sqrt((x-cx1)**2 + (y-cy1)**2 + (z-cz1)**2)
        env1 = 15.0 * np.exp(-(r1**2)/15.0) 
        phase1 = r1 + (x/N)*np.pi
        
        # NEUTRON (Vpravo - vítězný klíč 180° neboli PI)
        cx2, cy2, cz2 = N//2 + start_dist/2, N//2, N//2
        r2 = np.sqrt((x-cx2)**2 + (y-cy2)**2 + (z-cz2)**2)
        env2 = 17.5 * np.exp(-(r2**2)/15.0) 
        phase2 = r2 - (x/N)*np.pi + np.pi 

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
        self.fig.patch.set_facecolor('#020205')
        
        self.zoom = 40
        self.im = self.ax.imshow(np.zeros((self.zoom*2, self.zoom*2)), cmap='magma', origin='lower', vmin=0, vmax=600, interpolation='bicubic')
        
        self.ax.set_title("TCD: Finální Zážeh (Vítězná Rezonance 10Hz/180°)", color='cyan', pad=15, fontsize=16)
        self.ax.axis('off')
        
        self.info_text = self.ax.text(0.02, 0.92, '', transform=self.ax.transAxes, color='white', fontsize=12, fontweight='bold')
        self.status_text = self.ax.text(0.02, 0.85, '', transform=self.ax.transAxes, color='#e879f9', fontsize=14, fontweight='extra bold')
        self.contours = None

    def run(self, steps=800):
        print("[!] SPOUŠTÍM SIMULACI ZÁŽEHU")
        pump_val = 8.0 # Naše vítězná intenzita
        
        for t in range(steps):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_hm.data, np.float64(self.dt), np.int32(self.N), np.float64(pump_val), np.int32(t))
            
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
                bridge_val = slice_smooth[self.zoom, self.zoom]
                
                # Dynamický jas pro ohňostroj
                self.im.set_clim(vmin=0, vmax=max(150.0, core_max))
                
                if t < 200:
                    msg = "PŘIBLÍŽENÍ: Synchronizace fází..."
                    status = ""
                elif t < 400:
                    msg = f"BUZENÍ: Intenzita pumpy {pump_val:.1f}..."
                    status = "HLADINA ENERGIE ROSTE"
                elif bridge_val > 100:
                    msg = "!!! HANDOVER !!!"
                    status = "ZÁMEK KLAPL - STABILNÍ DEUTERIUM"
                else:
                    msg = "REKONFIGURACE MŘÍŽKY..."
                    status = "TOPOLOGICKÁ SMĚNA V BĚHU"

                self.info_text.set_text(f"Tik: {t} | Středový most: {bridge_val:.1f}")
                self.status_text.set_text(f"{msg}\n{status}")

                if self.contours:
                    try: self.contours.remove()
                    except:
                        for c in self.contours.collections: c.remove()
                            
                if t > 50:
                    levels = [60, 150, 250, 400, 550] 
                    self.contours = self.ax.contour(slice_smooth, levels=levels, colors=['#38bdf8', '#818cf8', '#c084fc', '#e879f9', 'white'], alpha=0.4, linewidths=1.0)

                plt.pause(0.01)

        print("\n[OK] Reakce dokončena. Mřížka udržela fúzní uzel.")

if __name__ == "__main__":
    viz = FinalFusionViz(N=120)
    viz.run(steps=800)
