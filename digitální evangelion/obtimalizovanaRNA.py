import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.pyplot as plt
import csv
import time
import math

# =============================================================================
# OMNI-ENGINE v16.1 - GEOMETRIC FAST-TRACK (RNA REPLICATION - FIX)
# Oprava Architekta: Vrácena homeostáza mateřskému vláknu a snížen práh
# pro zrcadlení. Spirála se nyní správně ukotví a vyvolá fázový otisk.
# =============================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

double rand(int seed, int x, int y, int z) {
    long n = seed + x * 374761393 + y * 668265263 + z * 1013904223;
    n = (n ^ (n >> 13)) * 1274126177;
    return (double)(n & 0x7FFFFFFF) / (double)0x7FFFFFFF;
}

__kernel void tkv_optimized_replication(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *h_mass,
    const double dt, const int N, const double noise_level, const int seed)
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

    const double PHI = 1.6180339887;
    const double LAMBDA = 0.0833333333;
    const double KAPPA = 275.42;
    const double DIFFUSION = LAMBDA * PHI; 
    
    double pr = psi_r[i]; double pi = psi_i[i];
    double current_m = sqrt(pr*pr + pi*pi);

    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    // =========================================================
    // OPRAVA: PŘÍSUN ENERGIE PRO MATEŘSKÉ VLÁKNO
    // =========================================================
    double intake = 0.0;
    if (x < N/2) {
        intake = 0.15; // Mateřské vlákno musí aktivně sát, aby nezmizelo
    }

    // =========================================================
    // FÁZOVÉ ZRCADLENÍ (OTISK DO PRAVÉ POLOVINY)
    // =========================================================
    double mirror_pull = 0.0;
    if (x >= N/2 && x < N/2 + 25) {
        int template_x = N - x; // Přesné geometrické zrcadlo podle středu N/2
        int template_idx = template_x * N * N + y * N + z;
        double template_m = sqrt(psi_r[template_idx]*psi_r[template_idx] + psi_i[template_idx]*psi_i[template_idx]);
        
        // Snížen práh! Mřížka kopíruje cokoliv stabilnějšího než šum
        if (template_m > 20.0) {
            mirror_pull = 0.35; 
        }
    }

    double total_pull = intake + mirror_pull;

    double jitter_r = (rand(seed, x, y, z) - 0.5) * noise_level;
    double jitter_i = (rand(seed+1, x, y, z) - 0.5) * noise_level;

    double saturation = 0.5 * (1.0 - tanh(PHI * 2.0 * (current_m / KAPPA - 1.0)));
    if (saturation < 0.0) saturation = 0.0;

    double nr = pr + (DIFFUSION * lap_r * dt) + (pr * total_pull * saturation * dt) + jitter_r * dt;
    double ni = pi + (DIFFUSION * lap_i * dt) + (pi * total_pull * saturation * dt) + jitter_i * dt;

    double nm = sqrt(nr*nr + ni*ni);
    
    double limit = KAPPA * PHI; 
    if (nm > limit) {
        nr *= (limit / nm);
        ni *= (limit / nm);
        nm = limit;
    }

    psi_rn[i] = nr; 
    psi_in[i] = ni;
    h_mass[i] = nm;
}
"""

class FastRNAEngineFix:
    def __init__(self, N=90):
        self.N = N
        self.dt = 0.02 
        
        platforms = cl.get_platforms()
        gpus = []
        for p in platforms:
            try: gpus.extend(p.get_devices(device_type=cl.device_type.GPU))
            except: pass
        dev = gpus[0] if gpus else platforms[0].get_devices()[0]
        
        self.ctx = cl.Context([dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_optimized_replication")
        
        self.KAPPA = 275.42
        self.GOLDEN_ANGLE = 137.508 
        
        self.strand_length = 12
        self.spacing_z = 5
        self.helix_radius = 8
        self.parent_x = N//2 - 12
        self.daughter_x = N//2 + 12

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 11), gridspec_kw={'height_ratios': [3, 1]})
        self.fig.patch.set_facecolor('#02040a')
        
        self.im = self.ax1.imshow(np.zeros((N, N)), cmap='viridis', origin='lower', vmax=350)
        self.ax1.set_title("TCD Engine v16.1: Zlatá RNA Šroubovice", color='#38bdf8', fontsize=14)
        self.ax1.axis('off')
        
        self.info_text = self.ax1.text(0.02, 0.95, '', transform=self.ax1.transAxes, color='white', fontsize=12)
        
        self.ax2.set_facecolor('#010205')
        self.ax2.set_title("Vakuový šum vs. Geometrická Integrita", color='#38bdf8')
        self.ax2.set_xlim(0, 10000)
        self.ax2.set_ylim(0, 105)
        self.line_integrity, = self.ax2.plot([], [], color='#e879f9', lw=2)
        self.x_data, self.y_data = [], []

    def inject_golden_helix(self):
        print("[*] Vykresluji 3D RNA šroubovici pomocí Zlatého úhlu (137.5°)...")
        x, y, z = np.indices((self.N, self.N, self.N))
        pr_total = np.zeros_like(x, dtype=np.float64)
        pi_total = np.zeros_like(x, dtype=np.float64)
        
        start_z = self.N//2 - (self.strand_length * self.spacing_z)//2
        angle_rad = np.deg2rad(self.GOLDEN_ANGLE)
        
        self.codons = [] 
        
        for i in range(self.strand_length):
            cz = start_z + i * self.spacing_z
            cy = self.N//2 + self.helix_radius * np.sin(i * angle_rad)
            cx = self.parent_x + self.helix_radius * np.cos(i * angle_rad)
            
            self.codons.append((int(cx), int(cy), int(cz)))
            
            r = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
            # OPRAVA: Zvýšena startovní amplituda, aby se uzel hned "chytil" mřížky
            env = 65.0 * np.exp(-(r**2)/8.0) 
            phase = r + (z/self.N)*np.pi
            
            pr_total += env * np.cos(phase)
            pi_total += env * np.sin(phase)
            
        self.d_pr = cl_array.to_device(self.queue, pr_total)
        self.d_pi = cl_array.to_device(self.queue, pi_total)
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        self.d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

    def analyze_helix_copy(self, mass_3d):
        errors = 0
        for cx, cy, cz in self.codons:
            # Zrcadlová pozice dcery (N - cx přesně odpovídá C-Kernel logice)
            d_cx = self.N - cx 
            
            daughter_mass = mass_3d[d_cx, cy, cz]
            
            # Je tam uzel, který vyrostl z vakua?
            if daughter_mass < (self.KAPPA * 0.70) or daughter_mass > (self.KAPPA * 1.30):
                errors += 1
                
        integrity = ((self.strand_length - errors) / self.strand_length) * 100
        return integrity

    def run_fast_track(self, noise_level=60.0, ticks=10000):
        self.inject_golden_helix()
        print(f"[*] START OPTIMALIZOVANÉHO ENGINU | Šum: {noise_level}")
        
        t_start = time.time()
        for t in range(ticks):
            seed = int(time.time() * 1000) % 1000000 + t
            
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_hm.data, np.float64(self.dt), np.int32(self.N), 
                     np.float64(noise_level), np.int32(seed))
            
            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi
            
            if t % 15 == 0:
                self.queue.finish()
                mass_3d = self.d_hm.get().reshape((self.N, self.N, self.N))
                
                # Depth map projekce podél osy Y -> Vykreslí osu X a Z (Zboku)
                depth_map = np.max(mass_3d, axis=1).T 
                self.im.set_data(depth_map)
                self.im.set_clim(vmin=0, vmax=350)
                
                integrity = self.analyze_helix_copy(mass_3d)
                
                self.x_data.append(t)
                self.y_data.append(integrity)
                self.line_integrity.set_data(self.x_data, self.y_data)
                
                status = "STABILNÍ REPLIKACE (DNA SE KOPÍRUJE)" if integrity > 80 else "ZRCADLENÍ... (BOJ SE ŠUMEM)"
                if t < 50: status = "KONSOLIDACE MATRICE"
                
                self.info_text.set_text(f"Tik: {t} | Šum: {noise_level}\nIntegrita DNA kopie: {integrity:.1f} %\nStav: {status}")
                
                plt.pause(0.01)

        elapsed = time.time() - t_start
        print(f"\n[OK] {ticks} tiků vypočítáno za {elapsed:.2f} s.")
        print(f"[OK] Finální integrita vlákna: {integrity}%")
        plt.ioff(); plt.show()

if __name__ == "__main__":
    engine = FastRNAEngineFix(N=90)
    engine.run_fast_track(noise_level=75.0, ticks=10000)
