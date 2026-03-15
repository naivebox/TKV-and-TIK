import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.pyplot as plt
import csv
import time

# =============================================================================
# OMNI-ENGINE v15.5 - RNA REPLICATION & MUTATION SCANNER
# Cíl: Kvantifikace chybovosti (mutací) při topologickém kopírování vlákna
# v závislosti na teplotě (fázovém šumu) vakua Levelu D.
# =============================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Generátor pseudo-náhodných čísel pro fázový šum přímo v GPU
double rand(int seed, int x, int y, int z) {
    long n = seed + x * 374761393 + y * 668265263 + z * 1013904223;
    n = (n ^ (n >> 13)) * 1274126177;
    return (double)(n & 0x7FFFFFFF) / (double)0x7FFFFFFF;
}

__kernel void tkv_replication_step(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *h_mass,
    const double dt, const int N, const double noise_level, const int seed)
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

    const double TARGET_MASS = 275.42; // Cílová hustota "kodonu"

    double pr = psi_r[i]; double pi = psi_i[i];
    double current_m = sqrt(pr*pr + pi*pi);

    // Laplacián
    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    // FÁZOVÉ ZRCADLENÍ (TEMPLATE PRINTING)
    // Původní vlákno sídlí v levé polovině (x < N/2). Tlačí svůj tvar do pravé poloviny.
    double mirror_pull = 0.0;
    if (x > N/2 && x < N/2 + 15) {
        // Nasávání zrcadlové informace od mateřského vlákna
        int template_x = N/2 - (x - N/2);
        int template_idx = template_x * N * N + y * N + z;
        double template_m = sqrt(psi_r[template_idx]*psi_r[template_idx] + psi_i[template_idx]*psi_i[template_idx]);
        
        // Pokud je v mateřském vlákně uzel, pravá strana se ho snaží "obkreslit"
        if (template_m > TARGET_MASS * 0.5) {
            mirror_pull = 0.25; 
        }
    }

    // TERMODYNAMICKÝ ŠUM (Zdroj mutací)
    double jitter_r = (rand(seed, x, y, z) - 0.5) * noise_level;
    double jitter_i = (rand(seed+1, x, y, z) - 0.5) * noise_level;

    // Harmonický zámek (snaží se udržet správný tvar navzdory šumu)
    double saturation = 0.5 * (1.0 - tanh(4.0 * (current_m / TARGET_MASS - 1.0)));

    double nr = pr + (0.08 * lap_r * dt) + (pr * mirror_pull * saturation * dt) + jitter_r * dt;
    double ni = pi + (0.08 * lap_i * dt) + (pi * mirror_pull * saturation * dt) + jitter_i * dt;

    double nm = sqrt(nr*nr + ni*ni);
    if (nm > TARGET_MASS * 1.5) {
        nr *= (TARGET_MASS * 1.5 / nm);
        ni *= (TARGET_MASS * 1.5 / nm);
        nm = TARGET_MASS * 1.5;
    }

    psi_rn[i] = nr; 
    psi_in[i] = ni;
    h_mass[i] = nm;
}
"""

class ReplicationDataMiner:
    def __init__(self, N=80):
        self.N = N
        self.dt = 0.015
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_replication_step")
        self.results = []
        
        self.strand_length = 10
        self.spacing = 6
        self.base_y = N//2 - (self.strand_length * self.spacing)//2
        self.parent_x = N//2 - 8
        self.daughter_x = N//2 + 8

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})
        self.fig.patch.set_facecolor('#051005')
        
        self.im = self.ax1.imshow(np.zeros((N, N)), cmap='viridis', origin='lower', vmax=300)
        self.ax1.set_title("TCD Replikátor: Informační Zrcadlení", color='#bef264', fontsize=14)
        self.ax1.axis('off')
        
        self.info_text = self.ax1.text(0.02, 0.95, '', transform=self.ax1.transAxes, color='white', fontsize=12)
        
        # Graf mutací
        self.ax2.set_facecolor('#020502')
        self.ax2.set_title("Míra Mutací (Topologických chyb)", color='#bef264')
        self.ax2.set_xlim(0, 1000)
        self.ax2.set_ylim(0, self.strand_length + 1)
        self.line_mutations, = self.ax2.plot([], [], color='red', lw=2)
        self.x_data, self.y_data = [], []

    def inject_parent_strand(self):
        """Vloží do mřížky původní RNA vlákno (Matrici)."""
        x, y, z = np.indices((self.N, self.N, self.N))
        pr_total = np.zeros_like(x, dtype=np.float64)
        pi_total = np.zeros_like(x, dtype=np.float64)
        
        for i in range(self.strand_length):
            cy = self.base_y + i * self.spacing
            r = np.sqrt((x-self.parent_x)**2 + (y-cy)**2 + (z-self.N//2)**2)
            env = 20.0 * np.exp(-(r**2)/5.0)
            phase = r + (x/self.N)*np.pi
            pr_total += env * np.cos(phase)
            pi_total += env * np.sin(phase)
            
        self.d_pr = cl_array.to_device(self.queue, pr_total)
        self.d_pi = cl_array.to_device(self.queue, pi_total)
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        self.d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

    def analyze_copy(self, mass_3d):
        """Změří, jak přesně mřížka obkreslila dceřiné vlákno."""
        mutations = 0
        total_daughter_mass = 0
        
        for i in range(self.strand_length):
            cy = self.base_y + i * self.spacing
            parent_mass = mass_3d[self.parent_x, cy, self.N//2]
            daughter_mass = mass_3d[self.daughter_x, cy, self.N//2]
            
            total_daughter_mass += daughter_mass
            
            # Pokud se dceřiný uzel liší od originálu o více než 15%, je to Mutace!
            if daughter_mass < (parent_mass * 0.85) or daughter_mass > (parent_mass * 1.15):
                mutations += 1
                
        integrity = ((self.strand_length - mutations) / self.strand_length) * 100
        return mutations, integrity, total_daughter_mass

    def run_experiment(self, noise_level, max_steps=1000):
        print(f"[*] SPUŠTĚNÍ EXPERIMENTU | Teplota (Šum vakua): {noise_level}")
        self.inject_parent_strand()
        self.x_data, self.y_data = [], []
        
        for t in range(max_steps):
            seed = int(time.time() * 1000) % 1000000 + t
            
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_hm.data, np.float64(self.dt), np.int32(self.N), 
                     np.float64(noise_level), np.int32(seed))
            
            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi
            
            if t % 20 == 0:
                mass_3d = self.d_hm.get().reshape((self.N, self.N, self.N))
                mutations, integrity, copy_mass = self.analyze_copy(mass_3d)
                
                # Záznam tvrdých dat
                self.results.append([t, round(noise_level, 2), round(np.sum(mass_3d), 2), mutations, round(integrity, 2)])
                
                # Vizualizace
                slice_2d = mass_3d[:, :, self.N//2].T # Pohled shora
                self.im.set_data(slice_2d)
                
                status = "STABILNÍ KOPÍROVÁNÍ" if mutations == 0 else f"MUTACE DETEKOVÁNY ({mutations})"
                color = 'white' if mutations == 0 else 'red'
                
                self.info_text.set_text(f"Tik: {t} | Šum: {noise_level:.1f}\nIntegrita kopie: {integrity:.1f} %\nStav: {status}")
                self.info_text.set_color(color)
                
                self.x_data.append(t)
                self.y_data.append(mutations)
                self.line_mutations.set_data(self.x_data, self.y_data)
                
                plt.pause(0.01)
                
        print(f"[OK] Experiment dokončen. Finální integrita: {integrity}%")

    def save_data(self):
        filename = "tcd_replication_data.csv"
        print(f"\n[*] Ukládám tvrdá data do: {filename}")
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["Tik", "Teplota_Vakua_Psi", "Celkove_Pnuti", "Pocet_Mutaci", "Integrita_Kopie_Procenta"])
            writer.writerows(self.results)
        print("[OK] Data uložena.")

if __name__ == "__main__":
    miner = ReplicationDataMiner(N=80)
    # Zvýšíme teplotu (šum) vakua, abychom vyvolali řízené chyby (mutace)
    miner.run_experiment(noise_level=85.0, max_steps=1000)
    miner.save_data()
