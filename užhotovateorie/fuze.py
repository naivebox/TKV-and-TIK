import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import csv
import time

# =============================================================================
# OMNI-ENGINE v8.8 - THE GRAND IGNITION (PROJEKT EVA)
# FINÁLNÍ VALIDACE GEOMETRICKÉHO ZÁMKU.
# Fixní parametry: 10.0 Hz | 180° Phase | 0° Spin Offset
# Skenujeme pouze kinetický tlak (Momentum) v realistickém rozsahu.
# =============================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void tkv_ignition_step(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *h_mass,
    const double dt, const int N, const double current_momentum, const int t_step)
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

    // 1. REZONANČNÍ SYNCHRONIZACE
    double angle = RESONANCE_FREQ * dt;
    double pr_rot = pr * cos(angle) - pi * sin(angle);
    double pi_rot = pr * sin(angle) + pi * cos(angle);
    pr = pr_rot; pi = pi_rot;

    // 2. KINETICKÝ TLAK (Testované Momentum)
    double drift_r = 0.0; double drift_i = 0.0;
    if (t_step < 400) {
        if (x < N/2 - 2) { 
            drift_r = current_momentum * (psi_r[id_xm] - pr);
            drift_i = current_momentum * (psi_i[id_xm] - pi);
        } else if (x > N/2 + 2) { 
            drift_r = current_momentum * (psi_r[id_xp] - pr);
            drift_i = current_momentum * (psi_i[id_xp] - pi);
        }
    }

    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    // 3. TOPOLOGICKÝ HANDOVER (Změna z dvou uzlů na jeden)
    // Při dosažení geometrické rezonance se koeficient nasávání radikálně mění
    double intake = 0.0;
    if (t_step > 250) {
        // Kritická saturace pro vznik Deuteria
        double saturation = 0.5 * (1.0 - tanh(4.5 * (current_m / DEUTERIUM_TARGET - 0.98)));
        intake = 0.32 * saturation; 
    }
    
    double nr = pr + (0.08 * lap_r * dt) + (drift_r * dt) + (pr * intake * dt);
    double ni = pi + (0.08 * lap_i * dt) + (drift_i * dt) + (pi * intake * dt);

    double nm = sqrt(nr*nr + ni*ni);
    // Přísný limit mřížky (Vakuum nepustí víc než 900 jednotek pnutí)
    if (nm > 900.0) { nr *= (900.0/nm); ni *= (900.0/nm); nm = 900.0; }

    psi_rn[i] = nr; psi_in[i] = ni;
    h_mass[i] = nm;
}
"""

class GrandIgnition:
    def __init__(self, N=80):
        self.N = N
        self.dt = 0.015
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_ignition_step")
        self.results = []

    def run_test(self, test_momentum):
        x, y, z = np.indices((self.N, self.N, self.N))
        
        # NAŠE VÍTĚZNÁ ČÍSLA
        phase_key = np.pi # 180 stupňů
        spin_key = 0.0    # 0 stupňů
        
        # Startovní pozice (středně blízko)
        cx1, cy1, cz1 = self.N//2 - 6, self.N//2, self.N//2
        r1 = np.sqrt((x-cx1)**2 + (y-cy1)**2 + (z-cz1)**2)
        env1 = 15.0 * np.exp(-(r1**2)/15.0)
        phase1 = r1 + (x/self.N)*np.pi
        
        cx2, cy2, cz2 = self.N//2 + 6, self.N//2, self.N//2
        r2 = np.sqrt((x-cx2)**2 + (y-cy2)**2 + (z-cz2)**2)
        env2 = 17.5 * np.exp(-(r2**2)/15.0) 
        # Aplikace obou klíčů
        phase2 = r2 - (x/self.N)*np.pi + phase_key + spin_key

        pr_init = env1 * np.cos(phase1) + env2 * np.cos(phase2)
        pi_init = env1 * np.sin(phase1) + env2 * np.sin(phase2)

        d_pr = cl_array.to_device(self.queue, pr_init.astype(np.float64))
        d_pi = cl_array.to_device(self.queue, pi_init.astype(np.float64))
        d_pr_n = cl_array.empty_like(d_pr)
        d_pi_n = cl_array.empty_like(d_pi)
        d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

        for t in range(550):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     d_pr.data, d_pi.data, d_pr_n.data, d_pi_n.data, 
                     d_hm.data, np.float64(self.dt), np.int32(self.N), 
                     np.float64(test_momentum), np.int32(t))
            d_pr, d_pr_n = d_pr_n, d_pr
            d_pi, d_pi_n = d_pi_n, d_pi

        mass_3d = d_hm.get().reshape((self.N, self.N, self.N))
        bridge_density = mass_3d[self.N//2, self.N//2, self.N//2] 

        status = "REPULZE"
        # Hranice pro handover (Fúzní zámek)
        if bridge_density > 100.0: status = "!!! HANDOVER !!!"

        return test_momentum, np.max(mass_3d), bridge_density, status

    def start(self):
        print("[*] SPUŠTĚNÍ OPERACE GRAND IGNITION (v8.8)")
        print("[*] Rezonance: 10.0 Hz | Fáze: 180° | Spin: 0°")
        print("-" * 65)
        
        start_time = time.time()
        # Skenujeme momentum v realistickém rozsahu 2.0 až 15.0
        for m in np.arange(2.0, 15.1, 1.0):
            res = self.run_test(m)
            self.results.append(res)
            print(f"Tlak: {res[0]:4.1f} | Most: {res[2]:10.2f} | Status: {res[3]}")
            if res[3] == "!!! HANDOVER !!!":
                print(f"\n[!] CÍL DOSAŽEN! Handover klapl při tlaku {res[0]}!")
                break

        print("-" * 65)
        print(f"[OK] Validace dokončena za {round(time.time() - start_time, 1)} s.")
        self.export_data()

    def export_data(self):
        filename = "tcd_grand_ignition_results.csv"
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["Tlak", "Max_Pnuti", "Hustota_Mostu", "Status"])
            for r in self.results: writer.writerow(r)
        print(f"[OK] Finální report uložen do {filename}.")

if __name__ == "__main__":
    ignition = GrandIgnition(N=80)
    ignition.start()
