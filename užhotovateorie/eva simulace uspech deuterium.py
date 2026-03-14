import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# =============================================================================
# OMNI-ENGINE v12.1 - EVA REACTOR INTEGRATION (FUSION FIX)
# Oprava: Zvýšen koeficient vazby a implementován vzájemný přítlak jader.
# Cíl: Vynutit topologický Handover a stabilizaci Deuteria.
# =============================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void tkv_eva_cycle_step(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *h_mass,
    const double dt, const int N, const double field_A, 
    const double freq, const int t_step)
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

    double pr = psi_r[i]; double pi = psi_i[i];
    double current_m = sqrt(pr*pr + pi*pi);

    // 1. DYNAMICKÁ REZONANCE (Ladění frekvence)
    double angle = freq * dt;
    double pr_rot = pr * cos(angle) - pi * sin(angle);
    double pi_rot = pr * sin(angle) + pi * cos(angle);

    // 2. KINETICKÝ DRIFT (Vzájemné přitahování jader)
    double drift_r = 0.0; double drift_i = 0.0;
    double center_x = (double)N / 2.0;
    
    // Ve Fázi B (Handover) aktivujeme kinetické "slisování"
    if (freq < 10.1 && t_step < 700) {
        double momentum = 2.8; 
        if (x < center_x - 1) { 
            drift_r = momentum * (psi_r[id_xm] - pr_rot); 
            drift_i = momentum * (psi_i[id_xm] - pi_rot);
        } else if (x > center_x + 1) { 
            drift_r = momentum * (psi_r[id_xp] - pr_rot); 
            drift_i = momentum * (psi_i[id_xp] - pi_rot);
        }
    }

    // 3. FÁZOVÉ SÁNÍ (INTEGRACE VAZBY)
    double intake = 0.0;
    if (freq > 10.1) {
        // Mód Neutronizace (Pohlcování slupky)
        double r_dist = sqrt((double)((x-center_x)*(x-center_x) + (y-center_x)*(y-center_x)));
        if (r_dist < 4.0) intake = field_A * 0.08; 
    } else {
        // Mód Handover (Svařování mřížky)
        // Cílíme na Deuterium Target (552.0)
        double saturation = 0.5 * (1.0 - tanh(4.5 * (current_m / 552.0 - 0.96)));
        intake = 0.42 * saturation; // ZVÝŠENO PRO PRŮRAZ BARIÉRY
    }

    double lap_r = -6.0*pr_rot + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi_rot + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    double nr = pr_rot + (0.07 * lap_r * dt) + (pr_rot * intake * dt) + (drift_r * dt);
    double ni = pi_rot + (0.07 * lap_i * dt) + (pi_rot * intake * dt) + (drift_i * dt);

    double nm = sqrt(nr*nr + ni*ni);
    if (nm > 700.0) { nr *= (700.0/nm); ni *= (700.0/nm); nm = 700.0; }

    psi_rn[i] = nr; psi_in[i] = ni;
    h_mass[i] = nm;
}
"""

class EVAFinalIntegrator:
    def __init__(self, N=100):
        self.N = N
        self.dt = 0.02
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_eva_cycle_step")

        # Inicializace: Dva atomy vodíku blíž u sebe (distance 14)
        x, y, z = np.indices((N, N, N))
        cx1, cy1 = N//2 - 7, N//2
        cx2, cy2 = N//2 + 7, N//2
        r1 = np.sqrt((x-cx1)**2 + (y-cy1)**2 + (z-N//2)**2)
        r2 = np.sqrt((x-cx2)**2 + (y-cy2)**2 + (z-N//2)**2)
        
        # Jádra
        env = 18.0 * (np.exp(-(r1**2)/8.0) + np.exp(-(r2**2)/8.0))
        # Slupky (vytvářejí onen modrý mrak)
        shell = 1.2 * (np.exp(-((r1-8.0)**2)/5.0) + np.exp(-((r2-8.0)**2)/5.0))
        
        total_env = env + shell
        # Fázové zrcadlení pro fúzní zámek
        phase = (x/N)*np.pi
        
        self.d_pr = cl_array.to_device(self.queue, (total_env * np.cos(phase)).astype(np.float64))
        self.d_pi = cl_array.to_device(self.queue, (total_env * np.sin(phase)).astype(np.float64))
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        self.d_hm = cl_array.zeros(self.queue, N**3, dtype=np.float64)

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.patch.set_facecolor('black')
        # Změna mapy na 'inferno' a úprava vmax, aby vyniklo jádro nad mrakem
        self.im = self.ax.imshow(np.zeros((N, N)), cmap='inferno', origin='lower', vmax=450)
        self.ax.axis('off')
        self.text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, color='white', fontweight='bold', fontsize=12)

    def run_cycle(self):
        print("[*] START REAKTORU EVA v12.1 - INTEGROVANÝ CYKLUS")
        
        for t in range(1200):
            # FÁZE 1: NEUTRONIZACE (0-450) - 10.15 Hz
            if t < 450:
                current_freq = 10.15
                field_A = 5.0
                status = "FÁZE A: VÝROBA NEUTRONŮ (10.15 Hz)"
            # FÁZE 2: HANDOVER (450-1200) - 10.00 Hz
            else:
                current_freq = 10.00
                field_A = 1.0 
                status = "FÁZE B: FÚZNÍ ZÁŽEH (10.00 Hz)"

            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_hm.data, np.float64(self.dt), np.int32(self.N), 
                     np.float64(field_A), np.float64(current_freq), np.int32(t))
            
            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi

            if t % 15 == 0:
                # Vizualizace centrálního řezu
                mass_3d = self.d_hm.get().reshape((self.N, self.N, self.N))
                mass_2d = gaussian_filter(mass_3d[self.N//2, :, :], sigma=0.4)
                
                self.im.set_data(mass_2d)
                
                core_val = np.max(mass_2d)
                msg = f"Tik: {t} | Max Hustota: {core_val:.1f} | {status}"
                if core_val > 450:
                    msg += " !!! DEUTERIUM STABILNÍ !!!"
                
                self.text.set_text(msg)
                plt.pause(0.01)

if __name__ == "__main__":
    eva = EVAFinalIntegrator()
    eva.run_cycle()
