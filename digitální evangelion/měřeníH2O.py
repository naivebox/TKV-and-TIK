import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import time
import csv

# =============================================================================
# OMNI-ENGINE v9.1 - H2O ANGLE SWEEP (TVRDÁ DATA)
# Testujeme různé úhly vazby H-O-H. Měříme celkové topologické pnutí mřížky.
# Cíl: Dokázat, že 104.5 stupňů je absolutní energetické minimum Levelu D.
# =============================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void tkv_h2o_step(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *h_mass,
    const double dt, const int N)
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

    const double H2O_TARGET = 4900.0; 

    double pr = psi_r[i]; double pi = psi_i[i];
    double current_m = sqrt(pr*pr + pi*pi);

    double Jx = pr * (psi_i[id_xp] - psi_i[id_xm]) - pi * (psi_r[id_xp] - psi_r[id_xm]);
    double Jy = pr * (psi_i[id_yp] - psi_i[id_ym]) - pi * (psi_r[id_yp] - psi_r[id_ym]);
    double Jz = pr * (psi_i[id_zp] - psi_i[id_zm]) - pi * (psi_r[id_zp] - psi_r[id_zm]);
    double J_mag = sqrt(Jx*Jx + Jy*Jy + Jz*Jz);

    double isolation = exp(-J_mag * 3.5);

    double cx = N/2.0; double cy = N/2.0; double cz = N/2.0;
    double dx_c = (double)x - cx;
    double dy_c = (double)y - cy;
    double dz_c = (double)z - cz;
    double r_dist = sqrt(dx_c*dx_c + dy_c*dy_c + dz_c*dz_c) + 0.001;

    double pull = 0.0;
    if (r_dist > 2.0 && r_dist < 40.0) {
        pull = 15.0 * (1.0 / r_dist); 
    }
    
    double drift_r = pull * (dx_c/r_dist) * 0.15 * pr;
    double drift_i = pull * (dx_c/r_dist) * 0.15 * pi;

    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    double saturation = 0.5 * (1.0 - tanh(4.5 * (current_m / H2O_TARGET - 0.98)));
    
    double intake = 0.20 * saturation;
    double local_friction = (current_m > H2O_TARGET * 0.8) ? 0.15 : 0.0;

    double nr = pr + (0.08 * lap_r * isolation * dt) - (drift_r * dt) + (pr * intake * dt) - (local_friction * pr * dt);
    double ni = pi + (0.08 * lap_i * isolation * dt) - (drift_i * dt) + (pi * intake * dt) - (local_friction * pi * dt);

    double nm = sqrt(nr*nr + ni*ni);
    if (nm > H2O_TARGET * 1.5) {
        nr *= (H2O_TARGET * 1.5 / nm);
        ni *= (H2O_TARGET * 1.5 / nm);
        nm = H2O_TARGET * 1.5;
    }

    psi_rn[i] = nr; psi_in[i] = ni; h_mass[i] = nm;
}
"""

class H2OAngleScanner:
    def __init__(self, N=96): # Mírně zmenšená mřížka pro rychlost skenování
        self.N = N
        self.dt = 0.015
        
        platforms = cl.get_platforms()
        dev = platforms[0].get_devices()[0]
        self.ctx = cl.Context([dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_h2o_step")
        self.results = []

    def run_angle_test(self, test_angle_deg):
        angle_rad = np.deg2rad(test_angle_deg)
        x, y, z = np.indices((self.N, self.N, self.N))
        cx, cy, cz = self.N//2, self.N//2, self.N//2
        
        # 1. Kyslík
        r_O = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
        env_O = 45.0 * np.exp(-(r_O**2) / 14.0)
        phase_O = r_O + (x/self.N)*np.pi + np.pi/2
        
        # Startovní vzdálenost vodíků (orbital)
        bond_length = 18.0 
        
        # 2. Vodík 1
        h1_x = cx + bond_length * np.cos(angle_rad / 2)
        h1_y = cy + bond_length * np.sin(angle_rad / 2)
        r_H1 = np.sqrt((x-h1_x)**2 + (y-h1_y)**2 + (z-cz)**2)
        env_H1 = 15.0 * np.exp(-(r_H1**2) / 6.0)
        phase_H1 = r_H1 + (x/self.N)*np.pi
        
        # 3. Vodík 2
        h2_x = cx + bond_length * np.cos(-angle_rad / 2)
        h2_y = cy + bond_length * np.sin(-angle_rad / 2)
        r_H2 = np.sqrt((x-h2_x)**2 + (y-h2_y)**2 + (z-cz)**2)
        env_H2 = 15.0 * np.exp(-(r_H2**2) / 6.0)
        phase_H2 = r_H2 + (x/self.N)*np.pi + np.pi # Inverzní spin pro zámek

        # Superpozice
        pr_init = env_O * np.cos(phase_O) + env_H1 * np.cos(phase_H1) + env_H2 * np.cos(phase_H2)
        pi_init = env_O * np.sin(phase_O) + env_H1 * np.sin(phase_H1) + env_H2 * np.sin(phase_H2)

        d_pr = cl_array.to_device(self.queue, pr_init.astype(np.float64))
        d_pi = cl_array.to_device(self.queue, pi_init.astype(np.float64))
        d_pr_n = cl_array.empty_like(d_pr)
        d_pi_n = cl_array.empty_like(d_pi)
        d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

        # Spustíme stabilizaci na 600 tiků
        for t in range(600):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     d_pr.data, d_pi.data, d_pr_n.data, d_pi_n.data, 
                     d_hm.data, np.float64(self.dt), np.int32(self.N))
            d_pr, d_pr_n = d_pr_n, d_pr
            d_pi, d_pi_n = d_pi_n, d_pi

        # ZÍSKÁNÍ TVRDÝCH DAT
        mass_3d = d_hm.get()
        # Změříme CELKOVÉ pnutí v mřížce (Topologický stres systému)
        total_tension = np.sum(mass_3d)
        max_core = np.max(mass_3d)
        
        return {
            "uhel": test_angle_deg,
            "celkove_pnuti": total_tension,
            "max_jadro": max_core
        }

    def start_sweep(self):
        print("=" * 70)
        print(" OMNI-ENGINE v9.1: H2O BOND ANGLE SCANNER (TVRDÁ DATA)")
        print(" Hledáme topologické minimum energie v Levelu D.")
        print("=" * 70)
        
        # Testovací úhly (včetně očekávaného kontinua kolem 104.5)
        test_angles = [90.0, 95.0, 100.0, 102.5, 104.5, 106.0, 110.0, 115.0, 120.0, 180.0]
        
        start_time = time.time()
        
        for angle in test_angles:
            print(f"[*] Testuji konfiguraci: {angle:5.1f}° ... ", end="")
            res = self.run_angle_test(angle)
            self.results.append(res)
            print(f"Celkové pnutí (Stres): {res['celkove_pnuti']:12.1f} | Max jádro: {res['max_jadro']:.1f}")

        print("-" * 70)
        print(f"[OK] Skenování dokončeno za {time.time() - start_time:.1f} s.")
        
        # Analýza a export
        best_result = min(self.results, key=lambda x: x['celkove_pnuti'])
        print(f"\n[!!!] MATEMATICKÝ VERDIKT [!!!]")
        print(f"Nejnižší energetický stav (Nejstabilnější molekula) nastal při úhlu: {best_result['uhel']}°")
        
        filename = "tcd_h2o_angle_sweep_data.csv"
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["Uhel_Stupne", "Celkove_Pnuti", "Max_Hustota_Jadra"])
            for r in self.results:
                writer.writerow([r["uhel"], round(r["celkove_pnuti"], 2), round(r["max_jadro"], 2)])
        print(f"Data zapsána do: {filename}")

if __name__ == "__main__":
    scanner = H2OAngleScanner(N=96)
    scanner.start_sweep()
