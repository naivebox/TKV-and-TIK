import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import time
import csv
import sys
from scipy.ndimage import maximum_filter, label

# =============================================================================
# OMNI-ENGINE v17.1 - HEADLESS CELL DIVISION (MITOSIS DATA MINER)
# Architekt: R. Bandor
# Účel: Čistá datová těžba bez vizualizace. Sledujeme, zda kontinuální přísun
# fázového pnutí donutí buňku k topologickému rozštěpení (vzniku 2 buněk).
# =============================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

double rand(int seed, int x, int y, int z) {
    long n = seed + x * 374761393 + y * 668265263 + z * 1013904223;
    n = (n ^ (n >> 13)) * 1274126177;
    return (double)(n & 0x7FFFFFFF) / (double)0x7FFFFFFF;
}

__kernel void tkv_mitosis_step(
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

    // Kdekoliv se nachází hmota, budeme to považovat za střed (buňka se může hýbat)
    double cx = N/2.0; double cy = N/2.0; double cz = N/2.0;
    double dx = (double)x - cx; double dy = (double)y - cy; double dz = (double)z - cz;
    double r_dist = sqrt(dx*dx + dy*dy + dz*dz);

    // 1. TERMODYNAMICKÝ ŠUM (Konstantní přítok z okolí)
    double noise_level = 75.0; 
    if (current_m > 50.0) noise_level = 15.0; // Uvnitř buňky je klidněji
    
    double jitter_r = (rand(seed, x, y, z) - 0.5) * noise_level;
    double jitter_i = (rand(seed+1, x, y, z) - 0.5) * noise_level;

    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    // 2. RŮST A DĚLENÍ (MITÓZA)
    double intake = 0.0;
    double siphon = 0.0;
    double cleavage = 0.0; // Rozštěpná rýha

    // Membrána (udržuje obal tam, kde fázové pnutí klesá)
    if (current_m > 150.0 && current_m < 250.0) {
        intake = 0.15; // Regenerace stěny
    }

    // Jádro (Saje energii, buňka pomalu Roste)
    if (current_m >= 250.0) {
        intake = 0.30 * (1.0 - tanh(current_m / 800.0)); // Silnější sání
        siphon = 0.15; 
        
        // TOPOLOGICKÉ ŠTĚPENÍ: Jakmile je jádro příliš těžké, mřížka ho roztrhne
        if (current_m > 650.0 && x >= N/2 - 2 && x <= N/2 + 2) {
            cleavage = -1.5; // Destruktivní interference uprostřed
        }
    }

    // Odtah entropie klesá se vzdáleností od hutných částí
    if (current_m > 50.0) {
        siphon += 0.05;
    }

    double nr = pr + (0.08 * lap_r * dt) + (pr * intake * dt) - (pr * siphon * dt) + (pr * cleavage * dt) + (jitter_r * dt);
    double ni = pi + (0.08 * lap_i * dt) + (pi * intake * dt) - (pi * siphon * dt) + (pi * cleavage * dt) + (jitter_i * dt);

    double nm = sqrt(nr*nr + ni*ni);
    if (nm > 1000.0) { nr *= (1000.0/nm); ni *= (1000.0/nm); nm = 1000.0; }

    psi_rn[i] = nr; psi_in[i] = ni;
    h_mass[i] = nm;
}
"""

class CellMitosisDataMiner:
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
        self.knl = cl.Kernel(self.prg, "tkv_mitosis_step")
        self.history = []

        print("="*65)
        print(" OMNI-ENGINE v17.1 - BUNĚČNÉ DĚLENÍ (ČISTÁ DATA)")
        print(f" Hardware: {self.dev.name}")
        print("="*65)
        self.init_cell()

    def init_cell(self):
        x, y, z = np.indices((self.N, self.N, self.N))
        cx, cy, cz = self.N//2, self.N//2, self.N//2
        r = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
        
        membrane = 200.0 * np.exp(-((r - 35.0)**2) / 8.0)
        core = 450.0 * np.exp(-(r**2) / 20.0) # Startujeme s těžším jádrem
        
        env = membrane + core
        phase = r + (x/self.N)*np.pi
        
        self.d_pr = cl_array.to_device(self.queue, (env * np.cos(phase)).astype(np.float64))
        self.d_pi = cl_array.to_device(self.queue, (env * np.sin(phase)).astype(np.float64))
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        self.d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

    def detect_cores(self, mass_3d):
        """Spočítá, kolik jader (buněk) aktuálně v mřížce existuje."""
        # Odfiltrujeme šum a membránu, hledáme jen hustá jádra (> 400 Psi)
        core_mask = mass_3d > 400.0
        labeled_array, num_features = label(core_mask)
        return num_features

    def run_mining(self, total_ticks=3000):
        print(f"[*] Spouštím těžbu dat na {total_ticks} tiků. Čekejte...\n")
        
        start_time = time.time()
        
        for t in range(1, total_ticks + 1):
            seed = int(time.time() * 1000) % 1000000 + t
            
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_hm.data, np.float64(self.dt), np.int32(self.N), 
                     np.int32(t), np.int32(seed))
            
            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi
            
            if t % 50 == 0:
                self.queue.finish()
                mass_3d = self.d_hm.get().reshape((self.N, self.N, self.N))
                
                max_core = np.max(mass_3d)
                total_mass = np.sum(mass_3d)
                num_cells = self.detect_cores(mass_3d)
                
                # Změříme integritu membrány (hustotu v prstenci okolo buněk)
                membrane_mask = (mass_3d > 150) & (mass_3d < 250)
                mem_integrity = np.mean(mass_3d[membrane_mask]) if np.any(membrane_mask) else 0.0
                
                if num_cells == 1:
                    status = "RŮST (G1 Fáze)" if max_core < 600 else "PŘETLAK (S Fáze)"
                elif num_cells >= 2:
                    status = "MITÓZA DOKONČENA"
                else:
                    status = "KOLAPS"

                self.history.append([t, round(max_core, 2), round(total_mass, 2), round(mem_integrity, 2), num_cells, status])
                
                sys.stdout.write(f"\r\tTik: {t:04d} | Max Jádro: {max_core:6.1f} | Buněk: {num_cells} | Stav: {status}      ")
                sys.stdout.flush()

        elapsed = time.time() - start_time
        print(f"\n\n[OK] Těžba dat dokončena za {elapsed:.2f} s. ({total_ticks/elapsed:.0f} tiků/s)")
        self.export_csv()

    def export_csv(self):
        filename = "tcd_cell_division_data.csv"
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(["Tik", "Max_Hustota_Jadra", "Celkova_Hmotnost", "Integrita_Membrany", "Pocet_Bunek", "Status"])
            writer.writerows(self.history)
        print(f"[*] Surová data o buněčném dělení uložena do: {filename}")

if __name__ == "__main__":
    miner = CellMitosisDataMiner(N=120)
    miner.run_mining(total_ticks=3000)
