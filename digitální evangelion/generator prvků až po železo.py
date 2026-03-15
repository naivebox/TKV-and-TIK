import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import csv
import sys
import time

# =============================================================================
# OMNI-ENGINE v14.1 - STELLAR FORGE (OPRAVA BERYLIOVÉ BARIÉRY)
# Architekt: R. Bandor
# Oprava: Do jádra byl vrácen dynamický Sifon, který brání přehuštění (runaway).
# Nyní můžeme bezpečně vyšplhat po "Alfa žebříku" až k Železu.
# =============================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void stellar_fusion_step(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *h_mass,
    const double dt, const int N, const double target_mass, 
    const double pump, const int t_step)
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

    // Rezonance 10.0 Hz
    double angle = 10.0 * dt;
    double pr_rot = pr * cos(angle) - pi * sin(angle);
    double pi_rot = pr * sin(angle) + pi * cos(angle);

    // Difuze a Laplacián
    double lap_r = -6.0*pr_rot + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi_rot + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    // Harmonika: Saturace sání
    double saturation = 0.5 * (1.0 - tanh(5.0 * (current_m / target_mass - 0.95)));
    if (saturation < 0.0) saturation = 0.0;

    double intake = (0.25 * saturation) + (pump * 0.01); // Zjemněná pumpa

    // DYNAMICKÝ SIFON (OPRAVA BERYLIOVÉ BARIÉRY)
    // Jakmile se blížíme k cílové mase, Sifon začne odsávat přebytečnou energii z pumpy.
    double local_friction = 0.0;
    if (current_m > target_mass * 0.8) {
        local_friction = 0.15 * (current_m / target_mass);
    }

    double nr = pr_rot + (0.08 * lap_r * dt) + (pr_rot * intake * dt) - (local_friction * pr_rot * dt);
    double ni = pi_rot + (0.08 * lap_i * dt) + (pi_rot * intake * dt) - (local_friction * pi_rot * dt);

    double nm = sqrt(nr*nr + ni*ni);
    
    // ABSOLUTNÍ SIFON MŘÍŽKY
    const double SUPERNOVA_LIMIT = 18000.0; 
    if (nm > SUPERNOVA_LIMIT) { 
        nr *= (SUPERNOVA_LIMIT / nm); 
        ni *= (SUPERNOVA_LIMIT / nm); 
        nm = SUPERNOVA_LIMIT; 
    }

    psi_rn[i] = nr; 
    psi_in[i] = ni;
    h_mass[i] = nm;
}
"""

class StellarForge:
    def __init__(self, N=64):
        self.N = N
        self.dt = 0.015
        
        platforms = cl.get_platforms()
        gpus = []
        for p in platforms:
            try: gpus.extend(p.get_devices(device_type=cl.device_type.GPU))
            except: pass
        self.dev = gpus[0] if gpus else platforms[0].get_devices()[0]
        self.ctx = cl.Context([self.dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "stellar_fusion_step")
        
        self.history = []

        # TCD "Alfa žebřík" nukleosyntézy
        self.elements_ladder = [
            ("Deuterium", "D", 1, 552.93),
            ("Hélium-4", "He", 2, 1100.45),
            ("Berylium-8", "Be", 4, 2198.10),
            ("Uhlík-12", "C", 6, 3290.55),
            ("Kyslík-16", "O", 8, 4385.20),
            ("Neon-20", "Ne", 10, 5480.00),
            ("Hořčík-24", "Mg", 12, 6570.80),
            ("Křemík-28", "Si", 14, 7665.40),
            ("Síra-32", "S", 16, 8750.10),
            ("Argon-36", "Ar", 18, 9845.00),
            ("Vápník-40", "Ca", 20, 10930.50),
            ("Titan-44", "Ti", 22, 12025.00),
            ("Chrom-48", "Cr", 24, 13110.00),
            ("Železo-56", "Fe", 26, 15400.00),
            ("Nikl-62", "Ni", 28, 17000.00) # Zde očekáváme průraz Sifonu!
        ]

        print("="*70)
        print(" OMNI-ENGINE v14.1 - STELLAR FORGE (Opravený Sifon)")
        print(f" Zámek mřížky (Kritický limit): 18 000 Psi")
        print("="*70)

    def fuse_element(self, element_name, symbol, atomic_num, target_psi):
        sys.stdout.write(f"[*] Kování: {element_name:<12} ({symbol:<2}) | Cíl: {target_psi:8.1f} Psi")
        sys.stdout.flush()
        
        x, y, z = np.indices((self.N, self.N, self.N))
        r = np.sqrt((x-self.N//2)**2 + (y-self.N//2)**2 + (z-self.N//2)**2)
        
        env = (target_psi * 0.8) * np.exp(-(r**2)/20.0) 
        phase = r + (x/self.N)*np.pi
        
        d_pr = cl_array.to_device(self.queue, (env*np.cos(phase)).astype(np.float64))
        d_pi = cl_array.to_device(self.queue, (env*np.sin(phase)).astype(np.float64))
        d_pr_n = cl_array.empty_like(d_pr)
        d_pi_n = cl_array.empty_like(d_pi)
        d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

        pump_intensity = 8.0
        final_mass = 0.0
        
        for t in range(1, 401): 
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     d_pr.data, d_pi.data, d_pr_n.data, d_pi_n.data, 
                     d_hm.data, np.float64(self.dt), np.int32(self.N), 
                     np.float64(target_psi), np.float64(pump_intensity), np.int32(t))
            
            d_pr, d_pr_n = d_pr_n, d_pr
            d_pi, d_pi_n = d_pi_n, d_pi
            
            if t == 400:
                self.queue.finish()
                final_mass = np.max(d_hm.get())

        status = "STABILNÍ"
        defect_mev = (target_psi - final_mass) * 0.1 
        
        if final_mass >= 17999.0: 
            status = "SUPERNOVA"
            defect_mev = 0.0
            
        print(f" -> Výsledek: {final_mass:8.1f} Psi | {status}")
        
        self.history.append([element_name, symbol, atomic_num, target_psi, final_mass, round(defect_mev, 2), status])
        
        return status != "SUPERNOVA"

    def run_ladder(self):
        print("")
        for el in self.elements_ladder:
            success = self.fuse_element(el[0], el[1], el[2], el[3])
            time.sleep(0.2) 
            
            if not success:
                print(f"\n[!!!] TOPOLOGICKÝ SIFON PŘETEKL NA PRVKU: {el[0]}")
                print("[!!!] Jádro je pro 3D stín příliš těžké. Konec nukleosyntézy.")
                break
                
        self.export_data()

    def export_data(self):
        filename = "tcd_nucleosynthesis_ladder_v2.csv"
        print(f"\n[*] Zapisuji opravenou tabulku do: {filename}")
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(["Prvek", "Symbol", "Atomove_Cislo", "Teoreticke_Pnuti_Psi", "Dosazene_Pnuti_Psi", "Hmotnostni_Defekt_MeV", "Status_Mrizky"])
            writer.writerows(self.history)
        print("[OK] Zápis kompletní.")

if __name__ == "__main__":
    forge = StellarForge(N=64)
    forge.run_ladder()
