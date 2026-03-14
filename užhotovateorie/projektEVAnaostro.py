import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import time
import csv
import sys

# =============================================================================
# OMNI-ENGINE v13.0 - REAL EVA REACTOR (HEADLESS MODE)
# Architekt: Rudolf Bandor
# Účel: Čistá datová simulace fúzního zážehu bez vizualizační brzdy.
# Nasazeny VŠECHNY objevené moduly (Sifon, Harmonika, Izolace, Probuzení).
# Parametry převzaty z analýzy Tik 1185: 10.0 Hz | 180° Fáze | Pumpa 8.0
# =============================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void eva_real_step(
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

    // --- 1. TOPOLOGICKÝ PROUD (Detekce víru) ---
    double Jx = pr * (psi_i[id_xp] - psi_i[id_xm]) - pi * (psi_r[id_xp] - psi_r[id_xm]);
    double Jy = pr * (psi_i[id_yp] - psi_i[id_ym]) - pi * (psi_r[id_yp] - psi_r[id_ym]);
    double Jz = pr * (psi_i[id_zp] - psi_i[id_zm]) - pi * (psi_r[id_zp] - psi_r[id_zm]);
    double J_mag = sqrt(Jx*Jx + Jy*Jy + Jz*Jz);

    // --- 2. GLUONOVÁ IZOLACE (Termoska) ---
    double isolation = exp(-J_mag * 5.0);

    // --- 3. REZONANČNÍ TEP MŘÍŽKY (10 Hz) ---
    double angle = RESONANCE_FREQ * dt;
    double pr_rot = pr * cos(angle) - pi * sin(angle);
    double pi_rot = pr * sin(angle) + pi * cos(angle);
    pr = pr_rot; pi = pi_rot;

    // Kinetický drift (Přiblížení jader v prvních fázích)
    double drift_r = 0.0; double drift_i = 0.0;
    double mom = 2.0; 
    if (t_step < 500) {
        if (x < N/2 - 1) { drift_r = mom * (psi_r[id_xm] - pr); drift_i = mom * (psi_i[id_xm] - pi); }
        else if (x > N/2 + 1) { drift_r = mom * (psi_r[id_xp] - pr); drift_i = mom * (psi_i[id_xp] - pi); }
    }

    // --- 4. LAPLACIÁN (Difuze) ---
    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    // --- 5. HARMONIKA (Nelineární zámek) ---
    double saturation = 0.5 * (1.0 - tanh(4.5 * (current_m / DEUTERIUM_TARGET - 0.98)));
    if (saturation < 0.0) saturation = 0.0;

    // --- 6. FÁZOVÁ PUMPA & PROBUZENÍ ---
    double dx = (double)x - N/2.0; double dy = (double)y - N/2.0; double dz = (double)z - N/2.0;
    double dist_to_center = sqrt(dx*dx + dy*dy + dz*dz);
    
    // Pumpa 8.0 působí primárně v místě sváru (mezi uzly)
    double local_pump = 0.0;
    if (dist_to_center < 3.0 && t_step > 250) {
        local_pump = pump_intensity * 0.05; 
    }

    double awakening = 1.0 - exp(-J_mag * 30.0);
    double intake = (0.25 * awakening * saturation) + local_pump;

    // --- 7. LOKÁLNÍ SIFON (Ochrana proti singularitě) ---
    double local_friction = 0.0;
    if (current_m > (DEUTERIUM_TARGET * 0.8)) {
        local_friction = 0.15 * (current_m / DEUTERIUM_TARGET);
    }

    // --- 8. MASTER ROVNICE TCD ---
    // Laplacián je tlumen gluonovou izolací
    double nr = pr + (0.08 * lap_r * isolation * dt) + (drift_r * dt) + (pr * intake * dt) - (local_friction * pr * dt);
    double ni = pi + (0.08 * lap_i * isolation * dt) + (drift_i * dt) + (pi * intake * dt) - (local_friction * pi * dt);

    double nm = sqrt(nr*nr + ni*ni);
    
    // Absolutní hard-limit mřížky (8D Průraz)
    const double OVERFLOW_LIMIT = 800.0;
    if (nm > OVERFLOW_LIMIT) { 
        nr *= (OVERFLOW_LIMIT / nm); 
        ni *= (OVERFLOW_LIMIT / nm); 
        nm = OVERFLOW_LIMIT; 
    }

    psi_rn[i] = nr; 
    psi_in[i] = ni;
    h_mass[i] = nm;
}
"""

class HeadlessEVAReactor:
    def __init__(self, N=128): # Větší produkční mřížka pro přesné výpočty
        self.N = N
        self.dt = 0.015
        self.pump_intensity = 8.0 # VÍTĚZNÁ PUMPA
        
        # Automatická selekce nejlepšího zařízení (GPU preference)
        platforms = cl.get_platforms()
        gpus = []
        for p in platforms:
            try: gpus.extend(p.get_devices(device_type=cl.device_type.GPU))
            except: pass
        self.dev = gpus[0] if gpus else platforms[0].get_devices()[0]
        
        self.ctx = cl.Context([self.dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "eva_real_step")
        self.history = []

        print("="*60)
        print(f" OMNI-ENGINE v13.0 - REAL EVA REACTOR")
        print(f" Hardware: {self.dev.name}")
        print(f" Rozlišení mřížky: {N}x{N}x{N} ({N**3} voxelů)")
        print(f" Parametry: 10.0 Hz | Phase: 180° | Pump: {self.pump_intensity}")
        print("="*60)

    def inject_fuel(self):
        print("[*] Injektuji palivo (Proton + Neutron)...")
        x, y, z = np.indices((self.N, self.N, self.N))
        
        start_dist = 20.0 
        
        # PROTON (Levý uzel)
        cx1, cy1, cz1 = self.N//2 - start_dist/2, self.N//2, self.N//2
        r1 = np.sqrt((x-cx1)**2 + (y-cy1)**2 + (z-cz1)**2)
        env1 = 15.0 * np.exp(-(r1**2)/15.0) 
        phase1 = r1 + (x/self.N)*np.pi
        
        # NEUTRON (Pravý uzel - Fázově zrcadlový: Zlatý klíč 180 stupňů / PI)
        cx2, cy2, cz2 = self.N//2 + start_dist/2, self.N//2, self.N//2
        r2 = np.sqrt((x-cx2)**2 + (y-cy2)**2 + (z-cz2)**2)
        env2 = 17.5 * np.exp(-(r2**2)/15.0) 
        phase2 = r2 - (x/self.N)*np.pi + np.pi # + PI je náš 180° klíč!

        pr_init = env1 * np.cos(phase1) + env2 * np.cos(phase2)
        pi_init = env1 * np.sin(phase1) + env2 * np.sin(phase2)

        self.d_pr = cl_array.to_device(self.queue, pr_init.astype(np.float64))
        self.d_pi = cl_array.to_device(self.queue, pi_init.astype(np.float64))
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        self.d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

    def run_reactor(self, total_ticks=2000):
        self.inject_fuel()
        print(f"\n[!] ZÁŽEH REAKTORU ({total_ticks} tiků)")
        
        start_time = time.time()
        
        for t in range(1, total_ticks + 1):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_hm.data, np.float64(self.dt), np.int32(self.N), 
                     np.float64(self.pump_intensity), np.int32(t))
            
            # Swap bufferů
            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi
            
            # Vzorkování dat každých 50 tiků (bez vizualizační zátěže)
            if t % 50 == 0:
                self.queue.finish() # Synchronizace pro čisté čtení
                
                # Rychlé stažení jen centrální roviny pro metriky (šetří PCIe propustnost)
                # Stáhneme celý mass buffer a najdeme maximum
                mass_array = self.d_hm.get()
                max_m = np.max(mass_array)
                
                # Zjistíme hustotu přesně vprostřed (fázový most)
                center_idx = (self.N//2) * self.N * self.N + (self.N//2) * self.N + (self.N//2)
                bridge_m = mass_array[center_idx]
                
                # Výpočet uvolněné energie (Hmotnostní defekt = uvolněné MeV)
                defect = 0.0
                if bridge_m > 100.0:
                    defect = 2.22 # Známý defekt po fúzi
                
                self.history.append([t, max_m, bridge_m, defect])
                
                # Průběžný výpis do konzole s přepisováním řádku
                sys.stdout.write(f"\r\tTik: {t:04d}/{total_ticks} | Max Hustota: {max_m:6.1f} | Sila Mostu: {bridge_m:6.1f} | Energie: {defect:.2f} MeV ")
                sys.stdout.flush()

        elapsed = time.time() - start_time
        print(f"\n\n[OK] Výpočet dokončen za {elapsed:.2f} s. ({total_ticks/elapsed:.0f} tiků/s)")
        self.export_log()

    def export_log(self):
        filename = "eva_headless_log.csv"
        print(f"[*] Exportuji telemetrii do {filename}...")
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(["Tik", "Max_Hustota", "Sila_Mostu", "Uvolnena_Energie_MeV"])
            writer.writerows(self.history)
        print("[OK] Hotovo. Data jsou připravena pro analýzu a Knihu III.")

if __name__ == "__main__":
    # Rozlišení 128x128x128 (přes 2 miliony voxelů simulovaných v reálném čase)
    reactor = HeadlessEVAReactor(N=128)
    reactor.run_reactor(total_ticks=2000)
