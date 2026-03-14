import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import time
import csv

# =============================================================================
# OMNI-ENGINE v8.4 - PROJEKT EVA (PHASE SYNC OPTIMIZER)
# Fixní rezonance: 10.0 Hz. Fixní tlak: 2.5.
# Úkol: Najít přesný fázový klíč (0° - 360°), který otevře Coulombův zámek.
# =============================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void tkv_sync_step(
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

    const double DEUTERIUM_TARGET = 552.93; 
    const double RESONANCE_FREQ = 10.0;

    double pr = psi_r[i]; double pi = psi_i[i];
    double current_m = sqrt(pr*pr + pi*pi);

    // 1. Rezonanční rotace (10 Hz)
    double angle = RESONANCE_FREQ * dt;
    double pr_rot = pr * cos(angle) - pi * sin(angle);
    double pi_rot = pr * sin(angle) + pi * cos(angle);
    pr = pr_rot; pi = pi_rot;

    // 2. Kinetický tlak (Stabilní momentum 2.5)
    double drift_r = 0.0; double drift_i = 0.0;
    double mom = 2.5;
    if (t_step < 350) {
        if (x < N/2 - 3) { drift_r = mom * (psi_r[id_xm] - pr); drift_i = mom * (psi_i[id_xm] - pi); }
        else if (x > N/2 + 3) { drift_r = mom * (psi_r[id_xp] - pr); drift_i = mom * (psi_i[id_xp] - pi); }
    }

    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    // Topologická Směna
    double intake = 0.0;
    if (t_step > 200) {
        double saturation = 0.5 * (1.0 - tanh(4.5 * (current_m / DEUTERIUM_TARGET - 0.98)));
        intake = 0.28 * saturation; 
    }
    
    double nr = pr + (0.08 * lap_r * dt) + (drift_r * dt) + (pr * intake * dt);
    double ni = pi + (0.08 * lap_i * dt) + (drift_i * dt) + (pi * intake * dt);

    double nm = sqrt(nr*nr + ni*ni);
    if (nm > 800.0) { nr *= (800.0/nm); ni *= (800.0/nm); nm = 800.0; }

    psi_rn[i] = nr; psi_in[i] = ni;
    h_mass[i] = nm;
}
"""

class SyncOptimizer:
    def __init__(self, N=80):
        self.N = N
        self.dt = 0.015
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_sync_step")
        self.results = []

    def run_sync_test(self, phase_deg):
        phase_rad = np.deg2rad(phase_deg)
        x, y, z = np.indices((self.N, self.N, self.N))
        start_dist = 12.0 
        
        # PROTON
        cx1, cy1, cz1 = self.N//2 - start_dist/2, self.N//2, self.N//2
        r1 = np.sqrt((x-cx1)**2 + (y-cy1)**2 + (z-cz1)**2)
        env1 = 15.0 * np.exp(-(r1**2)/15.0)
        phase1 = r1 + (x/self.N)*np.pi
        
        # NEUTRON - SKENOVANÝ KLÍČ
        cx2, cy2, cz2 = self.N//2 + start_dist/2, self.N//2, self.N//2
        r2 = np.sqrt((x-cx2)**2 + (y-cy2)**2 + (z-cz2)**2)
        env2 = 17.5 * np.exp(-(r2**2)/15.0) 
        phase2 = r2 - (x/self.N)*np.pi + phase_rad # ZDE TESTUJEME ÚHEL

        pr_init = env1 * np.cos(phase1) + env2 * np.cos(phase2)
        pi_init = env1 * np.sin(phase1) + env2 * np.sin(phase2)

        d_pr = cl_array.to_device(self.queue, pr_init.astype(np.float64))
        d_pi = cl_array.to_device(self.queue, pi_init.astype(np.float64))
        d_pr_n = cl_array.empty_like(d_pr)
        d_pi_n = cl_array.empty_like(d_pi)
        d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

        for t in range(450):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     d_pr.data, d_pi.data, d_pr_n.data, d_pi_n.data, 
                     d_hm.data, np.float64(self.dt), np.int32(self.N), np.int32(t))
            d_pr, d_pr_n = d_pr_n, d_pr
            d_pi, d_pi_n = d_pi_n, d_pi

        mass_3d = d_hm.get().reshape((self.N, self.N, self.N))
        max_density = np.max(mass_3d)
        bridge_density = mass_3d[self.N//2, self.N//2, self.N//2] 

        status = "ODRAZ"
        if bridge_density > 80.0 and max_density < 750.0: status = "HANDOVER!"
        elif max_density >= 799.0: status = "KOLAPS"

        return phase_deg, max_density, bridge_density, status

    def start_scan(self):
        print("[*] TCD SYNC-OPTIMIZER v8.4 online.")
        print("[*] Skenuji fázové klíče 0° - 360° při rezonanci 10.0 Hz...")
        print("-" * 65)
        
        start_time = time.time()
        # Skenujeme po 2 stupních pro maximální šanci
        for deg in range(0, 361, 2):
            res = self.run_sync_test(deg)
            self.results.append(res)
            
            if res[3] == "HANDOVER!":
                print(f"[!] >>> ZÁMEK KLAPL! <<< Úhel: {res[0]}° | Most: {res[2]:.2f}")
            elif res[0] % 30 == 0:
                print(f"[*] Skener na {res[0]}°... (Most: {res[2]:.2f})")

        print("-" * 65)
        print(f"[OK] Skenování dokončeno za {round(time.time() - start_time, 1)} vteřin.")
        self.export_data()

    def export_data(self):
        filename = "eva_sync_optimizer_results.csv"
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["Uhel_Stupne", "Max_Pnuti", "Hustota_Mostu", "Status"])
            for r in self.results: writer.writerow(r)
        print(f"[OK] Data uložena do {filename}. Najdeme ten zlatý úhel!")

if __name__ == "__main__":
    scanner = SyncOptimizer(N=80)
    scanner.start_scan()
