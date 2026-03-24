import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter

# =====================================================================
# OMNI-ENGINE v6.0: PROJEKT VODÍK (PURE AXIOMATIC EMERGENCE)
# Architekt Levelu D | Matematika Dodatku G+ a O
# =====================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void tkv_hydrogen_step(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *h_mass,
    const double dt, const int N)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x >= N || y >= N || z >= N) return;
    int i = x * N * N + y * N + z;

    // Periodické okrajové podmínky (Topologie Dvanáctistěnné mřížky)
    int id_xp = ((x + 1) % N) * N * N + y * N + z;
    int id_xm = ((x - 1 + N) % N) * N * N + y * N + z;
    int id_yp = x * N * N + ((y + 1) % N) * N + z;
    int id_ym = x * N * N + ((y - 1 + N) % N) * N + z;
    int id_zp = x * N * N + y * N + ((z + 1) % N);
    int id_zm = x * N * N + y * N + ((z - 1 + N) % N);

    // --- FUNDAMENTÁLNÍ KONSTANTY (Z LUT_CORE a Dodatku O) ---
    const double PSI_H = 138.20;           // Základní fázový bit (Vodík)
    const double DELTA_THETA = 0.179856;   // Dihedrální deficit 10.305° v radiánech
    const double PHI = 1.618034;           // Zlatý řez
    const double ALPHA_G = 1.1999816;      // Kalibrační vazba sítě

    double pr = psi_r[i];
    double pi = psi_i[i];
    double M = sqrt(pr*pr + pi*pi);

    // --- 1. OPERÁTOR GEOMETRICKÉ FRUSTRACE (G_hat) ---
    // Pnutí úměrné poměru aktuální hmoty vůči ideálnímu Vodíku
    double G = DELTA_THETA * (M / PSI_H);

    // --- 2. CHRONONOVÝ ROTÁTOR (Časový skluz) ---
    // Zpomalení času vlivem nutnosti uzavírat 10.305° mezeru
    double local_t = 1.0 / (1.0 + G);
    double eff_dt = dt * local_t;

    // --- 3. KOVARIANTNÍ LAPLACIÁN (Čisté šíření pole) ---
    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    // Schrödingerova-topodynamická rovnice (dPsi = i * Laplacian * dt)
    double dr = -lap_i * ALPHA_G; 
    double di =  lap_r * ALPHA_G; 

    double next_r = pr + dr * eff_dt;
    double next_i = pi + di * eff_dt;

    // --- 4. TORZNÍ PNUTÍ (Vnitřní Spin & DNA Helix Basis) ---
    // Mřížka kompenzuje chybu rotací fáze
    double alpha_torsion = (DELTA_THETA / PHI) * (M / PSI_H); 
    double theta = alpha_torsion * eff_dt;
    
    double cos_t = cos(theta);
    double sin_t = sin(theta);

    double final_r = next_r * cos_t - next_i * sin_t;
    double final_i = next_r * sin_t + next_i * cos_t;

    // --- 5. TOPOLOGICKÉ DNO VODÍKU (Geometrické ukotvení) ---
    // Kolem 138.20 vzniká přirozená rezonanční jamka
    double new_M = sqrt(final_r*final_r + final_i*final_i);
    if (new_M > 5.0) { // Aplikujeme pouze na jádro, vakuum necháme volné
        // Jemná přitažlivost k ideálnímu pnutí Vodíku
        double pull = (PSI_H - new_M) * 0.02 * eff_dt;
        double factor = 1.0 + (pull / new_M);
        final_r *= factor;
        final_i *= factor;
        new_M = sqrt(final_r*final_r + final_i*final_i);
    }

    psi_rn[i] = final_r;
    psi_in[i] = final_i;
    h_mass[i] = new_M;
}
"""

class HydrogenSimulator:
    def __init__(self, size=64):
        self.N = size
        self.dt = 0.005

        dev = self._choose_device()
        self.ctx = cl.Context([dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_hydrogen_step")

        self.d_pr = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_pi = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        self.d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.patch.set_facecolor('#050505')
        
        # Omezení barvy na míru Vodíku (max 150)
        self.im = self.ax.imshow(np.zeros((self.N, self.N)), 
                                 cmap='magma', 
                                 origin='lower',
                                 norm=colors.PowerNorm(gamma=0.5, vmin=0.01, vmax=150),
                                 interpolation='bicubic')
        self.ax.axis('off')
        self.contours = None

    def _choose_device(self):
        platforms = cl.get_platforms()
        gpus = []
        cpus = []
        for p in platforms:
            try:
                gpus.extend(p.get_devices(device_type=cl.device_type.GPU))
            except cl.LogicError: pass
            try:
                cpus.extend(p.get_devices(device_type=cl.device_type.CPU))
            except cl.LogicError: pass

        best_dev = gpus[0] if gpus else (cpus[0] if cpus else platforms[0].get_devices()[0])
        print(f"[+] OMNI-ENGINE v6.0: AXIOMATIC MODE")
        print(f"[+] Zařízení: {best_dev.name}")
        return best_dev

    def inject_atom(self):
        print("[*] Generuji fázový uzel přesně dle LUT_CORE (Ψ = 138.20)...")
        x, y, z = np.indices((self.N, self.N, self.N))
        dx, dy, dz = x - self.N//2, y - self.N//2, z - self.N//2
        r = np.sqrt(dx**2 + dy**2 + dz**2)

        # Vodíkové jádro tvořené geometrickou 10.305° anomálií
        # Začínáme přesně na pnutí 138.20
        p_env = 138.20 * np.exp(-(r**2) / 12.0)
        
        # Fázový Trefoil (Topologický zkrut fáze)
        phase = (dx*dy*dz) / (self.N**2) * np.pi 

        pr = p_env * np.cos(phase)
        pi = p_env * np.sin(phase)

        self.d_pr = cl_array.to_device(self.queue, pr.flatten().astype(np.float64))
        self.d_pi = cl_array.to_device(self.queue, pi.flatten().astype(np.float64))

    def update_visuals(self, t):
        mass_3d = self.d_hm.get().reshape((self.N, self.N, self.N))
        slice_data = mass_3d[self.N//2, :, :]
        slice_data = gaussian_filter(slice_data, sigma=0.5)

        self.im.set_data(slice_data)

        if self.contours:
            for c in self.contours.collections:
                c.remove()

        # Izolinie přesně sladěné s fázovým pnutím (Vodík = 138.20)
        levels = [1.0, 20.0, 50.0, 100.0, 130.0, 138.2]
        self.contours = self.ax.contour(slice_data, levels=levels, colors='cyan', alpha=0.3, linewidths=0.6)

        max_val = np.max(slice_data)
        self.ax.set_title(f"Vodík (H) | Axiomatický čas: {t} | Max Ψ: {max_val:.2f} / 138.20", color='white', fontsize=12)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self, ticks=20000):
        self.inject_atom()
        for t in range(ticks):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_hm.data, np.float64(self.dt), np.int32(self.N))
            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi
            if t % 50 == 0:
                self.queue.finish()
                self.update_visuals(t)

if __name__ == "__main__":
    sim = HydrogenSimulator(size=64)
    sim.run()

