import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import time
import csv
import sys
import matplotlib.pyplot as plt

# =============================================================================
# OMNI-ENGINE v13.1 - STOCHASTIC ROBUSTNESS TEST (PROJEKT EVA)
# Účel: Test fúzního zážehu za chaotických (reálných) počátečních podmínek.
# Skript provede N nezávislých běhů s náhodným šumem, různou vzdáleností 
# a fázovými odchylkami. Následně zprůměruje data a vykreslí graf.
# =============================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void eva_real_step(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *h_mass,
    const double dt, const int N, const double pump_intensity, const int t_step)
{
    // ... [ZDE JE POUŽITO NAPROSTO IDENTICKÉ JÁDRO Z v13.0] ...
    // Fyzika se nemění, měníme pouze chaos na vstupu!
    
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

    double Jx = pr * (psi_i[id_xp] - psi_i[id_xm]) - pi * (psi_r[id_xp] - psi_r[id_xm]);
    double Jy = pr * (psi_i[id_yp] - psi_i[id_ym]) - pi * (psi_r[id_yp] - psi_r[id_ym]);
    double Jz = pr * (psi_i[id_zp] - psi_i[id_zm]) - pi * (psi_r[id_zp] - psi_r[id_zm]);
    double J_mag = sqrt(Jx*Jx + Jy*Jy + Jz*Jz);

    double isolation = exp(-J_mag * 5.0);

    double angle = RESONANCE_FREQ * dt;
    double pr_rot = pr * cos(angle) - pi * sin(angle);
    double pi_rot = pr * sin(angle) + pi * cos(angle);
    pr = pr_rot; pi = pi_rot;

    double drift_r = 0.0; double drift_i = 0.0;
    double mom = 2.0; 
    if (t_step < 500) {
        if (x < N/2 - 1) { drift_r = mom * (psi_r[id_xm] - pr); drift_i = mom * (psi_i[id_xm] - pi); }
        else if (x > N/2 + 1) { drift_r = mom * (psi_r[id_xp] - pr); drift_i = mom * (psi_i[id_xp] - pi); }
    }

    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    double saturation = 0.5 * (1.0 - tanh(4.5 * (current_m / DEUTERIUM_TARGET - 0.98)));
    if (saturation < 0.0) saturation = 0.0;

    double dx = (double)x - N/2.0; double dy = (double)y - N/2.0; double dz = (double)z - N/2.0;
    double dist_to_center = sqrt(dx*dx + dy*dy + dz*dz);
    
    double local_pump = 0.0;
    if (dist_to_center < 3.0 && t_step > 250) {
        local_pump = pump_intensity * 0.05; 
    }

    double awakening = 1.0 - exp(-J_mag * 30.0);
    double intake = (0.25 * awakening * saturation) + local_pump;

    double local_friction = 0.0;
    if (current_m > (DEUTERIUM_TARGET * 0.8)) {
        local_friction = 0.15 * (current_m / DEUTERIUM_TARGET);
    }

    double nr = pr + (0.08 * lap_r * isolation * dt) + (drift_r * dt) + (pr * intake * dt) - (local_friction * pr * dt);
    double ni = pi + (0.08 * lap_i * isolation * dt) + (drift_i * dt) + (pi * intake * dt) - (local_friction * pi * dt);

    double nm = sqrt(nr*nr + ni*ni);
    
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

class RobustnessTester:
    def __init__(self, N=96, runs=5):
        self.N = N
        self.dt = 0.015
        self.runs = runs
        self.pump_intensity = 8.0 
        
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
        
        # Sběrače dat pro všechny běhy
        self.all_max_mass = []
        self.all_bridge_mass = []
        self.time_ticks = []

        print("="*65)
        print(f" OMNI-ENGINE v13.1 - STOCHASTIC ROBUSTNESS TEST")
        print(f" Počet simulovaných kolizí: {self.runs}")
        print(f" Přidávám: Tepelný šum, odchylky vzdálenosti a fázový jitter.")
        print("="*65)

    def inject_chaotic_fuel(self, run_index):
        """Injektuje palivo s náhodnými odchylkami simulujícími reálnou fyziku."""
        np.random.seed(int(time.time() * 1000) % 2**32) # Různý seed pro každý běh
        
        # NÁHODNÉ PODMÍNKY:
        dist_noise = np.random.uniform(-4.0, 4.0)
        start_dist = 20.0 + dist_noise
        phase_jitter = np.random.uniform(-0.2, 0.2) # Drobná nepřesnost ve fázovém klíči (šum plazmatu)
        thermal_noise_level = np.random.uniform(0.05, 0.15) # Teplo vakua
        
        print(f"\n[Běh {run_index+1}/{self.runs}] Start dist: {start_dist:.1f} | Phase Jitter: {phase_jitter:+.3f} rad | Teplo vakua: {thermal_noise_level:.3f}")
        
        x, y, z = np.indices((self.N, self.N, self.N))
        
        cx1, cy1, cz1 = self.N//2 - start_dist/2, self.N//2, self.N//2
        r1 = np.sqrt((x-cx1)**2 + (y-cy1)**2 + (z-cz1)**2)
        env1 = 15.0 * np.exp(-(r1**2)/15.0) 
        phase1 = r1 + (x/self.N)*np.pi
        
        cx2, cy2, cz2 = self.N//2 + start_dist/2, self.N//2, self.N//2
        r2 = np.sqrt((x-cx2)**2 + (y-cy2)**2 + (z-cz2)**2)
        env2 = 17.5 * np.exp(-(r2**2)/15.0) 
        phase2 = r2 - (x/self.N)*np.pi + np.pi + phase_jitter # 180° klíč + chyba!

        # Vytvoření částic + TEPELNÝ ŠUM MŘÍŽKY
        noise_r = np.random.normal(0, thermal_noise_level, (self.N, self.N, self.N))
        noise_i = np.random.normal(0, thermal_noise_level, (self.N, self.N, self.N))

        pr_init = env1 * np.cos(phase1) + env2 * np.cos(phase2) + noise_r
        pi_init = env1 * np.sin(phase1) + env2 * np.sin(phase2) + noise_i

        self.d_pr = cl_array.to_device(self.queue, pr_init.astype(np.float64))
        self.d_pi = cl_array.to_device(self.queue, pi_init.astype(np.float64))
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        self.d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

    def run_tests(self, total_ticks=1500):
        for run in range(self.runs):
            self.inject_chaotic_fuel(run)
            
            run_max_mass = []
            run_bridge_mass = []
            
            for t in range(1, total_ticks + 1):
                self.knl(self.queue, (self.N, self.N, self.N), None, 
                         self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                         self.d_hm.data, np.float64(self.dt), np.int32(self.N), 
                         np.float64(self.pump_intensity), np.int32(t))
                
                self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
                self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi
                
                if t % 50 == 0:
                    self.queue.finish()
                    mass_array = self.d_hm.get()
                    
                    max_m = np.max(mass_array)
                    center_idx = (self.N//2) * self.N * self.N + (self.N//2) * self.N + (self.N//2)
                    bridge_m = mass_array[center_idx]
                    
                    run_max_mass.append(max_m)
                    run_bridge_mass.append(bridge_m)
                    
                    if run == 0:
                        self.time_ticks.append(t)
                        
                    sys.stdout.write(f"\r\tTik: {t:04d}/{total_ticks} | Max: {max_m:6.1f} | Most: {bridge_m:6.1f} ")
                    sys.stdout.flush()
                    
            self.all_max_mass.append(run_max_mass)
            self.all_bridge_mass.append(run_bridge_mass)
            print(" -> OK")

        self.generate_report()

    def generate_report(self):
        print("\n[*] Agreguji data a generuji výstupní graf...")
        
        # Převod na Numpy pole pro snadnou statistiku
        mat_max = np.array(self.all_max_mass)
        mat_bridge = np.array(self.all_bridge_mass)
        ticks = np.array(self.time_ticks)
        
        # Výpočet průměrů a směrodatné odchylky (rozptylu šumu)
        mean_max = np.mean(mat_max, axis=0)
        std_max = np.std(mat_max, axis=0)
        
        mean_bridge = np.mean(mat_bridge, axis=0)
        std_bridge = np.std(mat_bridge, axis=0)
        
        # --- ULOŽENÍ DO CSV ---
        with open("eva_stochastic_average.csv", mode='w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(["Tik", "Prumer_Max", "Odchylka_Max", "Prumer_Most", "Odchylka_Most"])
            for i in range(len(ticks)):
                writer.writerow([ticks[i], round(mean_max[i], 2), round(std_max[i], 2), round(mean_bridge[i], 2), round(std_bridge[i], 2)])

        # --- VYKRESLENÍ GRAFU ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Most (Fúzní bod)
        ax.plot(ticks, mean_bridge, color='#e879f9', linewidth=2.5, label='Hustota sváru (Průměr)')
        ax.fill_between(ticks, mean_bridge - std_bridge, mean_bridge + std_bridge, color='#e879f9', alpha=0.2, label='Tepelná odchylka sváru')
        
        # Max hustota (Jádra)
        ax.plot(ticks, mean_max, color='#38bdf8', linewidth=2.5, label='Max hustota jádra (Průměr)')
        ax.fill_between(ticks, mean_max - std_max, mean_max + std_max, color='#38bdf8', alpha=0.2)
        
        # Vyznačení hranice Homeostázy Deuteria (cca 552)
        ax.axhline(y=552.93, color='yellow', linestyle='--', alpha=0.6, label='TCD Teoretický Target Deuteria')
        
        ax.set_title(f"Robustnost Fúze: Agregace {self.runs} simulací s náhodným vakuovým šumem", fontsize=14, color='white', pad=15)
        ax.set_xlabel("Čas (Tiky mřížky Level D)", fontsize=12, color='gray')
        ax.set_ylabel("Topologické pnutí (Ψ)", fontsize=12, color='gray')
        ax.grid(color='white', alpha=0.1)
        ax.legend(loc='upper left', frameon=False)
        
        plt.tight_layout()
        plt.savefig("eva_robustness_graph.png", dpi=300)
        print("[OK] Graf 'eva_robustness_graph.png' byl vytvořen.")
        print("[OK] Agregovaná data uložena do 'eva_stochastic_average.csv'.")
        # Odkomentujte pro zobrazení okna grafu po dokončení
        # plt.show() 

if __name__ == "__main__":
    # Spustíme 5 nezávislých běhů s náhodnými počátečními podmínkami
    tester = RobustnessTester(N=96, runs=5)
    tester.run_tests(total_ticks=1500)
