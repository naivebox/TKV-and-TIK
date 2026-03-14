import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import csv
import time

# =============================================================================
# OMNI-ENGINE v7.3 - HEADLESS PHASE SCANNER (OPRAVENO)
# Skener nyní obsahuje logiku tlakem vynucené inverze (Beta rozpadu).
# Hledáme přesné úhly (spiny), pod kterými se Coulombova bariéra 
# prolomí a dojde k fázovému uzamčení (vznik Deuteria).
# =============================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void tkv_scan_step(
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

    const double PROTON_TARGET = 275.42;
    const double DEUTERIUM_TARGET = 552.93; 

    double pr = psi_r[i]; double pi = psi_i[i];
    
    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    double current_m = sqrt(pr*pr + pi*pi);
    
    double drift_r = 0.0; double drift_i = 0.0;
    double momentum = 3.5; 
    double intake = 0.0;
    
    // 1. PŘÍTLAČNÁ FÁZE
    if (t_step < 350) {
        if (x < N/2 - 2) { 
            drift_r = momentum * (psi_r[id_xm] - pr);
            drift_i = momentum * (psi_i[id_xm] - pi);
        } else if (x > N/2 + 2) { 
            drift_r = momentum * (psi_r[id_xp] - pr);
            drift_i = momentum * (psi_i[id_xp] - pi);
        }
    }

    // 2. TLAKOVÁ INVERZE (Beta rozpad)
    if (t_step > 120 && t_step < 300 && x > N/2 + 1) {
        if (current_m < 30.0) {
            intake = -0.015; // Zhroucení aury
        } else {
            intake = 0.035;  // Přehuštění jádra na neutron
        }
    }

    // 3. POKUS O TOPOLOGICKÝ ZÁMEK
    if (t_step > 250) {
        double saturation = 0.5 * (1.0 - tanh(3.5 * (current_m / DEUTERIUM_TARGET - 0.98)));
        intake += 0.18 * saturation;
    }
    
    double nr = pr + (0.08 * lap_r * dt) + (drift_r * dt) + (pr * intake * dt);
    double ni = pi + (0.08 * lap_i * dt) + (drift_i * dt) + (pi * intake * dt);

    double nm = sqrt(nr*nr + ni*ni);
    if (nm > 800.0) { // Ochrana proti singularitě
        nr *= (800.0 / nm);
        ni *= (800.0 / nm);
        nm = 800.0;
    }

    psi_rn[i] = nr; 
    psi_in[i] = ni;
    h_mass[i] = nm;
}
"""

class PhaseScannerFix:
    def __init__(self, N=80):
        self.N = N
        self.dt = 0.015
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_scan_step")
        self.results = []

    def run_simulation(self, phase_shift_deg):
        phase_shift_rad = np.deg2rad(phase_shift_deg)
        x, y, z = np.indices((self.N, self.N, self.N))
        
        start_dist = 20.0 
        
        # PROTON 1 (Fixní fáze)
        cx1, cy1, cz1 = self.N//2 - start_dist/2, self.N//2, self.N//2
        r1 = np.sqrt((x-cx1)**2 + (y-cy1)**2 + (z-cz1)**2)
        env1 = 15.0 * np.exp(-(r1**2)/12.0)
        phase1 = r1 + (x/self.N)*np.pi
        
        # PROTON 2 (Testovaný úhel/spin)
        cx2, cy2, cz2 = self.N//2 + start_dist/2, self.N//2, self.N//2
        r2 = np.sqrt((x-cx2)**2 + (y-cy2)**2 + (z-cz2)**2)
        env2 = 15.0 * np.exp(-(r2**2)/12.0)
        phase2 = r2 - (x/self.N)*np.pi + phase_shift_rad # Aplikace klíče

        pr_init = env1 * np.cos(phase1) + env2 * np.cos(phase2)
        pi_init = env1 * np.sin(phase1) + env2 * np.sin(phase2)

        d_pr = cl_array.to_device(self.queue, pr_init.astype(np.float64))
        d_pi = cl_array.to_device(self.queue, pi_init.astype(np.float64))
        d_pr_n = cl_array.empty_like(d_pr)
        d_pi_n = cl_array.empty_like(d_pi)
        d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

        # 450 tiků, aby mřížka měla dost času na stabilizaci sváru
        for t in range(450):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     d_pr.data, d_pi.data, d_pr_n.data, d_pi_n.data, 
                     d_hm.data, np.float64(self.dt), np.int32(self.N), np.int32(t))
            d_pr, d_pr_n = d_pr_n, d_pr
            d_pi, d_pi_n = d_pi_n, d_pi

        mass_3d = d_hm.get().reshape((self.N, self.N, self.N))
        max_density = np.max(mass_3d)
        bridge_density = mass_3d[self.N//2, self.N//2, self.N//2] 

        status = "NEZNÁMÝ"
        if max_density >= 799.0:
            status = "SINGULARITA (Destrukce)"
        elif bridge_density < 30.0:
            status = "REPULZE (Odraz)"
        elif bridge_density > 80.0 and max_density < 799.0:
            status = "ZÁMEK KLAPL (Stabilní Deuterium)"
        else:
            status = "METASTABILNÍ (Šum)"

        return {
            "uhel_stupne": phase_shift_deg,
            "uhel_radiany": round(phase_shift_rad, 4),
            "max_pnuti": round(max_density, 2),
            "stredovy_most": round(bridge_density, 2),
            "vysledek": status
        }

    def start_scan(self):
        print("[*] Inicializuji OMNI-ENGINE Skener (v7.3) s aktivní Inverzí...")
        print("[*] Prohledávám spinové úhly od 0° do 360° (krok 5°)")
        print("-" * 65)
        
        start_time = time.time()
        
        for deg in range(0, 361, 5):
            res = self.run_simulation(deg)
            self.results.append(res)
            
            if res["vysledek"] == "ZÁMEK KLAPL (Stabilní Deuterium)":
                print(f"[!] *** NALEZEN ZÁMEK! *** Úhel: {deg}° | Most: {res['stredovy_most']} | Max pnutí: {res['max_pnuti']}")
            elif res["vysledek"] == "SINGULARITA (Destrukce)":
                print(f"[X] SINGULARITA na úhlu {deg}° (Mřížka nevydržela)")
            elif deg % 30 == 0:
                print(f"[*] Skenuji úhel {deg}°... (Výsledek: {res['vysledek']})")

        print("-" * 65)
        print(f"[OK] Skenování dokončeno za {round(time.time() - start_time, 1)} vteřin.")
        self.export_data()

    def export_data(self):
        filename = "tcd_faze_klice_export_OPRAVENO.csv"
        print(f"[*] Zapisuji data do souboru: {filename}")
        
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["Uhel (Stupne)", "Uhel (Radiany)", "Max Pnuti", "Hustota Mostu", "Status Mrizky"])
            
            for r in self.results:
                writer.writerow([r["uhel_stupne"], r["uhel_radiany"], r["max_pnuti"], r["stredovy_most"], r["vysledek"]])
                
        print(f"[OK] Data úspěšně uložena. Zkontrolujte soubor {filename}!")

if __name__ == "__main__":
    scanner = PhaseScannerFix(N=80)
    scanner.start_scan()
