import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import time
import csv

# =============================================================================
# OMNI-ENGINE v8.2 - PROJEKT EVA (DIGITAL TWIN REACTOR)
# Simulace hledání rezonanční frekvence pro nízkoenergetickou fúzi (LENR).
# Frekvence výboje natáčí vnitřní spin Neutronu. Hledáme "Zlatý úhel".
# =============================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void tkv_eva_fusion_step(
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
    
    double pr = psi_r[i]; double pi = psi_i[i];
    double current_m = sqrt(pr*pr + pi*pi);

    // Kinetický tlak reaktoru (Plazma je stlačené, uzly se tlačí k sobě)
    double drift_r = 0.0; double drift_i = 0.0;
    double momentum = 1.5; // Zvýšeno z 0.4 na 1.5 (Dostatečné pro překonání vzdálenosti)
    
    if (t_step < 250) {
        if (x < N/2 - 2) { 
            drift_r = momentum * (psi_r[id_xm] - pr);
            drift_i = momentum * (psi_i[id_xm] - pi);
        } else if (x > N/2 + 2) { 
            drift_r = momentum * (psi_r[id_xp] - pr);
            drift_i = momentum * (psi_i[id_xp] - pi);
        }
    }

    // Laplacián
    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    // Fúze (Topologická Směna) - Aktivuje se při správném překryvu
    double intake = 0.0;
    if (t_step > 150) {
        double saturation = 0.5 * (1.0 - tanh(4.0 * (current_m / DEUTERIUM_TARGET - 0.98)));
        intake = 0.25 * saturation; 
    }
    
    double nr = pr + (0.08 * lap_r * dt) + (drift_r * dt) + (pr * intake * dt);
    double ni = pi + (0.08 * lap_i * dt) + (drift_i * dt) + (pi * intake * dt);

    double nm = sqrt(nr*nr + ni*ni);
    if (nm > 800.0) { 
        nr *= (800.0 / nm);
        ni *= (800.0 / nm);
        nm = 800.0;
    }

    psi_rn[i] = nr; 
    psi_in[i] = ni;
    h_mass[i] = nm;
}
"""

class EVAReactorDigitalTwin:
    def __init__(self, N=80):
        self.N = N
        self.dt = 0.015
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_eva_fusion_step")
        self.results = []

    def run_sweep_test(self, test_frequency):
        x, y, z = np.indices((self.N, self.N, self.N))
        
        # Startovní vzdálenost snížena na 12 (Na dotek Coulombovy bariéry)
        start_dist = 12.0 
        
        # PROTON (Levý uzel)
        cx1, cy1, cz1 = self.N//2 - start_dist/2, self.N//2, self.N//2
        r1 = np.sqrt((x-cx1)**2 + (y-cy1)**2 + (z-cz1)**2)
        env1 = 15.0 * np.exp(-(r1**2)/15.0)
        phase1 = r1 + (x/self.N)*np.pi
        
        # NEUTRON (Pravý uzel)
        cx2, cy2, cz2 = self.N//2 + start_dist/2, self.N//2, self.N//2
        r2 = np.sqrt((x-cx2)**2 + (y-cy2)**2 + (z-cz2)**2)
        env2 = 17.5 * np.exp(-(r2**2)/15.0) 
        
        # MAGIE PROJEKTU EVA: Elektromagnetický výboj mění frekvenci.
        # Tato frekvence se projevuje jako natáčení fázového spinu Neutronu vůči Protonu.
        # Mapujeme frekvenci 0-20 Hz na 0-360 stupňů rotace.
        phase_shift = (test_frequency / 20.0) * 2 * np.pi 
        phase2 = r2 - (x/self.N)*np.pi + phase_shift 

        pr_init = env1 * np.cos(phase1) + env2 * np.cos(phase2)
        pi_init = env1 * np.sin(phase1) + env2 * np.sin(phase2)

        d_pr = cl_array.to_device(self.queue, pr_init.astype(np.float64))
        d_pi = cl_array.to_device(self.queue, pi_init.astype(np.float64))
        d_pr_n = cl_array.empty_like(d_pr)
        d_pi_n = cl_array.empty_like(d_pi)
        d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

        for t in range(400):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     d_pr.data, d_pi.data, d_pr_n.data, d_pi_n.data, 
                     d_hm.data, np.float64(self.dt), np.int32(self.N), np.int32(t))
            d_pr, d_pr_n = d_pr_n, d_pr
            d_pi, d_pi_n = d_pi_n, d_pi

        mass_3d = d_hm.get().reshape((self.N, self.N, self.N))
        max_density = np.max(mass_3d)
        bridge_density = mass_3d[self.N//2, self.N//2, self.N//2] 

        # Detekce výsledku fúze
        status = "NEZNÁMÝ"
        if max_density >= 799.0:
            status = "KOLAPS (Příliš velký tlak)"
        elif bridge_density < 40.0:
            status = "REPULZE (Frekvence netrefila zámek)"
        elif bridge_density > 80.0 and max_density < 700.0:
            status = "ÚSPĚŠNÁ FÚZE (Deuterium Handover!)"
        else:
            status = "ŠUM (Nestabilní)"

        return {
            "frekvence": round(test_frequency, 2),
            "max_pnuti": round(max_density, 2),
            "stredovy_most": round(bridge_density, 2),
            "vysledek": status
        }

    def start_sweep(self):
        print("[*] Inicializuji Digitální dvojče reaktoru EVA (v8.2)...")
        print("[*] Natlakováno plazmou. Začínám frekvenční Sweep (0.0 Hz - 20.0 Hz)")
        print("-" * 65)
        
        start_time = time.time()
        
        # Testujeme detailněji
        freq_range = np.arange(0.0, 20.5, 0.5)
        
        for freq in freq_range:
            res = self.run_sweep_test(freq)
            self.results.append(res)
            
            if "FÚZE" in res["vysledek"]:
                print(f"[!] >>> ZÁMEK KLAPL! <<< Freq: {res['frekvence']} Hz | Most: {res['stredovy_most']} | Max: {res['max_pnuti']}")
            elif res["frekvence"] % 5.0 == 0:
                print(f"[*] Skener na {res['frekvence']} Hz... (Stav: {res['vysledek']} | Most: {res['stredovy_most']})")

        print("-" * 65)
        print(f"[OK] Sweep reaktoru dokončen za {round(time.time() - start_time, 1)} vteřin.")
        self.export_data()

    def export_data(self):
        filename = "eva_reactor_sweep_results.csv"
        print(f"[*] Zapisuji data do souboru: {filename}")
        
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["Frekvence", "Max Pnuti", "Hustota Mostu", "Status Mrizky"])
            
            for r in self.results:
                writer.writerow([r["frekvence"], r["max_pnuti"], r["stredovy_most"], r["vysledek"]])
                
        print(f"[OK] Data úspěšně uložena!")

if __name__ == "__main__":
    scanner = EVAReactorDigitalTwin(N=80)
    scanner.start_sweep()
