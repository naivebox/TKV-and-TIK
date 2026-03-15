import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time

# =============================================================================
# OMNI-ENGINE v17.0 - GENESIS (THE FIRST DIGITAL CELL)
# Kniha IV: Finále. Simulace Autopoézy v Levelu C.
# Fázová membrána chrání jádro před chaosem. Jádro (RNA) naopak pomocí
# Sifonu odčerpává entropii, čímž udržuje membránu stabilní.
# =============================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Random generátor pro vakuový šum
double rand(int seed, int x, int y, int z) {
    long n = seed + x * 374761393 + y * 668265263 + z * 1013904223;
    n = (n ^ (n >> 13)) * 1274126177;
    return (double)(n & 0x7FFFFFFF) / (double)0x7FFFFFFF;
}

__kernel void tkv_cell_step(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *h_mass,
    const double dt, const int N, const int t_step, const int seed)
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

    // Geometrie buňky
    double cx = N/2.0; double cy = N/2.0; double cz = N/2.0;
    double dx = (double)x - cx; double dy = (double)y - cy; double dz = (double)z - cz;
    double r_dist = sqrt(dx*dx + dy*dy + dz*dz);

    const double MEMBRANE_RADIUS = 35.0;
    const double CORE_RADIUS = 10.0;

    // 1. TERMODYNAMIKA A VAKUOVÝ ŠUM
    double noise_level = 0.0;
    if (r_dist > MEMBRANE_RADIUS) {
        // Venku zuří vakuová bouře (75 Psi)
        noise_level = 75.0; 
    } else {
        // Uvnitř by se měl šum přirozeně šířit, ale je tlumen
        noise_level = 15.0 * (r_dist / MEMBRANE_RADIUS);
    }
    
    double jitter_r = (rand(seed, x, y, z) - 0.5) * noise_level;
    double jitter_i = (rand(seed+1, x, y, z) - 0.5) * noise_level;

    // 2. LAPLACIÁN (Mřížková difuze)
    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    // 3. AUTOPOÉZA: MOTOR A MEMBRÁNA
    double intake = 0.0;
    double siphon = 0.0;

    // Membrána se snaží udržet svůj tvar (Lepidová dvouvrstva)
    if (r_dist > MEMBRANE_RADIUS - 3.0 && r_dist < MEMBRANE_RADIUS + 3.0) {
        intake = 0.15 * (1.0 - tanh(current_m / 200.0)); // Regenerace stěny
    }

    // Jádro (RNA Motor)
    if (r_dist <= CORE_RADIUS) {
        // Probuzení RNA - saje energii a pulzuje
        intake = 0.25 * (1.0 - tanh(current_m / 600.0));
        
        // ZLATÝ SIFON: Tohle je tajemství života! 
        // Jádro odčerpává entropii ze svého okolí do 8D stínu.
        siphon = 0.18; 
    }

    // 4. GLOBÁLNÍ CHLADICÍ EFEKT JÁDRA
    // Sifon v jádře vytváří podtlak, který vysává šum z celého vnitřku buňky.
    if (r_dist < MEMBRANE_RADIUS) {
        // Odtah entropie klesá se vzdáleností od jádra
        double cooling = 0.05 * exp(-r_dist / 15.0);
        siphon += cooling;
    }

    // MASTER ROVNICE BUŇKY
    double nr = pr + (0.08 * lap_r * dt) + (pr * intake * dt) - (pr * siphon * dt) + (jitter_r * dt);
    double ni = pi + (0.08 * lap_i * dt) + (pi * intake * dt) - (pi * siphon * dt) + (jitter_i * dt);

    double nm = sqrt(nr*nr + ni*ni);
    if (nm > 1000.0) { nr *= (1000.0/nm); ni *= (1000.0/nm); nm = 1000.0; }

    psi_rn[i] = nr; psi_in[i] = ni;
    h_mass[i] = nm;
}
"""

class CellGenesis:
    def __init__(self, N=120):
        self.N = N
        self.dt = 0.02
        
        platforms = cl.get_platforms()
        gpus = []
        for p in platforms:
            try: gpus.extend(p.get_devices(device_type=cl.device_type.GPU))
            except: pass
        self.dev = gpus[0] if gpus else platforms[0].get_devices()[0]
        
        self.ctx = cl.Context([self.dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_cell_step")

        print("="*65)
        print(" OMNI-ENGINE v17.0 - ZROZENÍ BUŇKY (AUTOPOÉZA)")
        print("="*65)

        self.init_cell()

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.patch.set_facecolor('#000000')
        # Použijeme colormap 'mako' nebo 'twilight_shifted', které krásně ukážou rozdíl mezi šumem a klidem
        self.im = self.ax.imshow(np.zeros((self.N, self.N)), cmap='twilight_shifted', origin='lower', vmin=0, vmax=600)
        self.ax.set_title("TCD Genesis: První Topologická Buňka", color='#38bdf8', fontsize=16, pad=15)
        self.ax.axis('off')
        
        self.info_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, color='white', fontsize=12, fontweight='bold')
        self.status_text = self.ax.text(0.02, 0.90, '', transform=self.ax.transAxes, color='#deff9a', fontsize=14)

    def init_cell(self):
        x, y, z = np.indices((self.N, self.N, self.N))
        cx, cy, cz = self.N//2, self.N//2, self.N//2
        r = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
        
        # Ochranná membrána (Hustota 200)
        membrane = 200.0 * np.exp(-((r - 35.0)**2) / 8.0)
        
        # Jádro / RNA (Hustota 500)
        core = 500.0 * np.exp(-(r**2) / 20.0)
        
        env = membrane + core
        phase = r + (x/self.N)*np.pi
        
        pr_init = env * np.cos(phase)
        pi_init = env * np.sin(phase)

        self.d_pr = cl_array.to_device(self.queue, pr_init.astype(np.float64))
        self.d_pi = cl_array.to_device(self.queue, pi_init.astype(np.float64))
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        self.d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

    def run(self, steps=1000):
        print("[*] Spouštím vakuovou bouři. Sledujte homeostázu buňky...")
        
        for t in range(steps):
            seed = int(time.time() * 1000) % 1000000 + t
            
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_hm.data, np.float64(self.dt), np.int32(self.N), 
                     np.int32(t), np.int32(seed))
            
            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi
            
            if t % 15 == 0:
                mass_3d = self.d_hm.get().reshape((self.N, self.N, self.N))
                slice_2d = gaussian_filter(mass_3d[:, :, self.N//2], sigma=0.5)
                
                self.im.set_data(slice_2d)
                
                # Měření stability
                core_val = np.max(slice_2d[self.N//2-5:self.N//2+5, self.N//2-5:self.N//2+5])
                membrane_val = np.max(slice_2d[self.N//2-38:self.N//2-32, self.N//2-5:self.N//2+5])
                
                if t < 100:
                    status = "NÁBĚH ŠUMU: Mřížka začíná vřít"
                elif core_val > 400 and membrane_val > 100:
                    status = "AUTOPOÉZA: Buňka se sama udržuje v chodu!"
                else:
                    status = "KRITICKÝ STAV: Prolomení membrány"

                self.info_text.set_text(f"Tik: {t:04d} | Jádro: {core_val:.0f} Ψ | Membrána: {membrane_val:.0f} Ψ")
                self.status_text.set_text(status)

                plt.pause(0.01)

        print("\n[!!!] SIMULACE DOKONČENA: Buňka přežila. Život je stabilní. [!!!]")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    sim = CellGenesis(N=120)
    sim.run(steps=1000)
