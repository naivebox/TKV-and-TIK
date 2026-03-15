import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import time
import csv
from scipy.spatial.transform import Rotation

# =============================================================================
# OMNI-ENGINE v9.2 - CARBON VALENCE SCANNER (TOPOLOGICKÁ KOMBINATORIKA)
# Architekt: R. Bandor
# Účel: Automatické hledání 3D stabilních úhlů pro Uhlík (C-12) a Vodíky (H).
# Generuje náhodné 3D úhly, měří fázový stres mřížky a ukládá ty nejlepší.
# =============================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void tkv_valence_step(
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

    // Cílová hmota makromolekuly (Zhruba C + 4H)
    const double TARGET_MASS = 4300.0; 

    double pr = psi_r[i]; double pi = psi_i[i];
    double current_m = sqrt(pr*pr + pi*pi);

    // Detekce topologického proudu pro izolaci (Kovalentní plášť)
    double Jx = pr * (psi_i[id_xp] - psi_i[id_xm]) - pi * (psi_r[id_xp] - psi_r[id_xm]);
    double Jy = pr * (psi_i[id_yp] - psi_i[id_ym]) - pi * (psi_r[id_yp] - psi_r[id_ym]);
    double Jz = pr * (psi_i[id_zp] - psi_i[id_zm]) - pi * (psi_r[id_zp] - psi_r[id_zm]);
    double J_mag = sqrt(Jx*Jx + Jy*Jy + Jz*Jz);

    double isolation = exp(-J_mag * 3.5);

    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    double saturation = 0.5 * (1.0 - tanh(4.5 * (current_m / TARGET_MASS - 0.98)));
    if (saturation < 0.0) saturation = 0.0;

    // Elektronegativita (Uhlík si stahuje fázi vodíků)
    double cx = N/2.0; double cy = N/2.0; double cz = N/2.0;
    double dx_c = (double)x - cx; double dy_c = (double)y - cy; double dz_c = (double)z - cz;
    double r_dist = sqrt(dx_c*dx_c + dy_c*dy_c + dz_c*dz_c) + 0.001;

    double pull = 0.0;
    if (r_dist > 2.0 && r_dist < 40.0) {
        pull = 15.0 * (1.0 / r_dist); 
    }
    
    double drift_r = pull * (dx_c/r_dist) * 0.15 * pr;
    double drift_i = pull * (dx_c/r_dist) * 0.15 * pi;

    double intake = 0.20 * saturation;
    double local_friction = (current_m > TARGET_MASS * 0.8) ? 0.15 : 0.0;

    double nr = pr + (0.08 * lap_r * isolation * dt) - (drift_r * dt) + (pr * intake * dt) - (local_friction * pr * dt);
    double ni = pi + (0.08 * lap_i * isolation * dt) - (drift_i * dt) + (pi * intake * dt) - (local_friction * pi * dt);

    double nm = sqrt(nr*nr + ni*ni);
    if (nm > TARGET_MASS * 1.5) {
        nr *= (TARGET_MASS * 1.5 / nm); ni *= (TARGET_MASS * 1.5 / nm); nm = TARGET_MASS * 1.5;
    }

    psi_rn[i] = nr; psi_in[i] = ni; h_mass[i] = nm;
}
"""

class CarbonScanner:
    def __init__(self, N=96):
        self.N = N
        self.dt = 0.015
        platforms = cl.get_platforms()
        dev = platforms[0].get_devices()[0]
        self.ctx = cl.Context([dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_valence_step")
        self.results = []
        
        # Pnutí Uhlíku-12 z Hvězdné kovadliny
        self.CARBON_MASS = 3290.55 
        self.HYDROGEN_MASS = 275.42

    def generate_random_3d_points(self, num_points, radius):
        """Vygeneruje N bodů náhodně rozmístěných na sféře (Simulace vazeb)."""
        points = []
        for _ in range(num_points):
            # Sférické souřadnice pro náhodnou distribuci
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.arccos(np.random.uniform(-1, 1))
            
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            points.append(np.array([x, y, z]))
        return points

    def calculate_angles(self, points):
        """Vypočítá úhly mezi všemi vygenerovanými vazbami."""
        angles = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                v1 = points[i] / np.linalg.norm(points[i])
                v2 = points[j] / np.linalg.norm(points[j])
                dot_prod = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle_deg = np.rad2deg(np.arccos(dot_prod))
                angles.append(round(angle_deg, 1))
        return angles

    def run_sweep(self, num_hydrogens, test_id):
        bond_radius = 18.0
        h_points = self.generate_random_3d_points(num_hydrogens, bond_radius)
        angles = self.calculate_angles(h_points)
        
        x, y, z = np.indices((self.N, self.N, self.N))
        cx, cy, cz = self.N//2, self.N//2, self.N//2
        
        # 1. UHLÍK V CENTRU
        r_C = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
        env_C = (self.CARBON_MASS * 0.01) * np.exp(-(r_C**2) / 15.0)
        phase_C = r_C + (x/self.N)*np.pi
        
        pr_init = env_C * np.cos(phase_C)
        pi_init = env_C * np.sin(phase_C)
        
        # 2. VODÍKY OKOLO
        for idx, pt in enumerate(h_points):
            h_x, h_y, h_z = cx + pt[0], cy + pt[1], cz + pt[2]
            r_H = np.sqrt((x-h_x)**2 + (y-h_y)**2 + (z-h_z)**2)
            env_H = (self.HYDROGEN_MASS * 0.05) * np.exp(-(r_H**2) / 8.0)
            
            # Spinový klíč (střídáme 0 a PI pro minimalizaci repulze)
            spin_offset = 0.0 if idx % 2 == 0 else np.pi
            phase_H = r_H + (x/self.N)*np.pi + spin_offset
            
            pr_init += env_H * np.cos(phase_H)
            pi_init += env_H * np.sin(phase_H)

        d_pr = cl_array.to_device(self.queue, pr_init.astype(np.float64))
        d_pi = cl_array.to_device(self.queue, pi_init.astype(np.float64))
        d_pr_n = cl_array.empty_like(d_pr)
        d_pi_n = cl_array.empty_like(d_pi)
        d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

        # Rychlá stabilizace mřížky (Ověření, zda se to neroztrhne)
        for t in range(400):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     d_pr.data, d_pi.data, d_pr_n.data, d_pi_n.data, 
                     d_hm.data, np.float64(self.dt), np.int32(self.N))
            d_pr, d_pr_n = d_pr_n, d_pr
            d_pi, d_pi_n = d_pi_n, d_pi

        # HARD DATA: Změříme CELKOVÝ stres mřížky (Nejnižší stres = Přirozený tvar)
        mass_3d = d_hm.get()
        total_tension = np.sum(mass_3d)
        
        return {
            "ID": test_id,
            "Pocet_Vazeb": num_hydrogens,
            "Uhly_Mezi_Vazbami": str(angles),
            "Celkove_Pnuti": total_tension
        }

    def start_monte_carlo(self):
        print("=" * 70)
        print(" OMNI-ENGINE v9.2: TOPOLOGICKÁ KOMBINATORIKA (UHLÍK)")
        print(" Testujeme náhodné 3D konfigurace vazeb (C + 2H, 3H, 4H)")
        print("=" * 70)
        
        start_time = time.time()
        
        # Testujeme různé počty vazeb (např. CH2, CH3, CH4)
        for num_h in [2, 3, 4]:
            print(f"\n[*] Skenuji konfigurace pro C + {num_h}H ...")
            # Pro každý počet vazeb zkusíme 30 náhodných uspořádání
            for test_idx in range(30):
                res = self.run_sweep(num_h, f"{num_h}H_test_{test_idx}")
                self.results.append(res)
                if test_idx % 10 == 0:
                    print(f"    Sken {test_idx}/30 ... Stres mřížky: {res['Celkove_Pnuti']:.1f}")

        print("-" * 70)
        print(f"[OK] Skenování dokončeno za {time.time() - start_time:.1f} s.")
        
        # Najdeme absolutní vítěze (Nejnižší celkové pnutí = Nejdokonalejší geometrie)
        print("\n[!!!] MATEMATICKÝ VERDIKT: NEJSTABILNĚJŠÍ TVARY [!!!]")
        
        df = self.results
        for num_h in [2, 3, 4]:
            h_results = [r for r in df if r["Pocet_Vazeb"] == num_h]
            best = min(h_results, key=lambda x: x["Celkove_Pnuti"])
            print(f"\nNejlepší konfigurace pro {num_h} vazby:")
            print(f"-> Úhly vazeb: {best['Uhly_Mezi_Vazbami']}")
            print(f"-> Výsledný stres: {best['Celkove_Pnuti']:.1f}")

        # Export
        filename = "tcd_carbon_valences_data.csv"
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["ID", "Pocet_Vazeb", "Uhly_Mezi_Vazbami", "Celkove_Pnuti"])
            for r in self.results:
                writer.writerow([r["ID"], r["Pocet_Vazeb"], r["Uhly_Mezi_Vazbami"], round(r["Celkove_Pnuti"], 2)])
        print(f"\n[OK] Kompletní kombinatorická data zapsána do: {filename}")

if __name__ == "__main__":
    # Menší mřížka (96) pro rychlé prohledání desítek kombinací
    scanner = CarbonScanner(N=96)
    scanner.start_monte_carlo()
