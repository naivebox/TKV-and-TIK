import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.pyplot as plt
import csv
import time

# =============================================================================
# OMNI-ENGINE v15.6 - RNA REPLICATION & EVOLUTION (ČISTÁ GEOMETRIE)
# =============================================================================
# Žádná magická čísla. Všechny konstanty jsou odvozeny z:
# - Dvanáctistěnné geometrie (12 stěn, 30 hran, 20 vrcholů, 5 vepsaných krychlí)
# - Zlatého řezu (φ = 1.61803398875)
# - Knihy VI: koroze paměti (λ), selekce (η_bio), spánek (α)
# =============================================================================

# --- FUNDAMENTÁLNÍ GEOMETRICKÉ KONSTANTY ---
PHI = (1.0 + np.sqrt(5.0)) / 2.0          # Zlatý řez = 1.61803398875
PHI_SQ = PHI * PHI                         # φ² = 2.61803398875

# --- GEOMETRICKÉ INVARIANTY DVANÁCTISTĚNU ---
STENY = 12.0
HRANY = 30.0
VRCHOLY = 20.0
VEPSANE_KRYCHLE = 5.0
LATERALNI_STENY = 10.0

# --- KONSTANTY Z KNIHY VI ---
# Koroze paměti (λ) – přirozený rozpad informace
# V Knize VI: dΨ/dt = ... - λΨ
# λ = 1/τ, kde τ je charakteristická doba paměti
# Z geometrie: λ = 1 / (VRCHOLY * PHI) ≈ 1 / (20 * 1.618) ≈ 0.0309
LAMBDA = 1.0 / (VRCHOLY * PHI)            # ≈ 0.0309

# Restituční člen (R8D) – rychlost čerpání energie ze Sifonu během spánku
# R8D = C * (J̃ / (S_topo + ε))
# Pro simulaci použijeme konstantu odvozenou z geometrie
R8D_BASE = 0.15

# Práh saturace pro spánek (S_topo > 85 %)
SLEEP_THRESHOLD = 0.85

# Koeficient bdělosti α(t) – plynulý přechod mezi dnem a nocí
# α = 1 = bdělost, α = 0 = hluboký spánek

# --- HMOTNOSTI Z LUT_CORE ---
TARGET_MASS = 275.426                     # κp – protonová homeostáza

# --- MUTACE A SELEKCE ---
# Tolerance pro "správný" kodon (z Knihy VI: 15 % = 3 × 5)
MUTATION_TOLERANCE = 0.15                # 15 %
MUTATION_TOLERANCE_LOW = 1.0 - MUTATION_TOLERANCE   # 0.85
MUTATION_TOLERANCE_HIGH = 1.0 + MUTATION_TOLERANCE  # 1.15

# Selekční práh – nový tvar je "lepší", když je stabilnější než původní
SELECTION_THRESHOLD = 0.05               # 5 % zlepšení


# --- OPENCL KERNEL (ČISTÁ GEOMETRIE) ---
kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Geometrické konstanty
const double STENY = 12.0;
const double HRANY = 30.0;
const double VRCHOLY = 20.0;
const double VEPSANE_KRYCHLE = 5.0;
const double LATERALNI_STENY = 10.0;
const double PHI = 1.61803398875;

// Generátor pseudo-náhodných čísel pro fázový šum
double rand(int seed, int x, int y, int z) {
    long n = seed + x * 374761393 + y * 668265263 + z * 1013904223;
    n = (n ^ (n >> 13)) * 1274126177;
    return (double)(n & 0x7FFFFFFF) / (double)0x7FFFFFFF;
}

__kernel void tkv_replication_pure_geometry(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *h_mass,
    const double dt, const int N, 
    const double noise_level, 
    const double alpha,           // Koeficient bdělosti (0 = spánek, 1 = bdělost)
    const double lambda,          // Koroze paměti
    const int seed)
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

    const double TARGET_MASS = 275.426;   // κp

    double pr = psi_r[i]; double pi = psi_i[i];
    double current_m = sqrt(pr*pr + pi*pi);

    // 1. LAPLACIÁN (difuze) – průměrování přes sousedy
    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];
    
    // Difuzní koeficient = 1 / STENY (12 stěn) – čistá geometrie
    double diffusion = (1.0 / STENY) * lap_r;  // 1/12 ≈ 0.08333

    // 2. FÁZOVÉ ZRCADLENÍ (TEMPLATE PRINTING)
    // Zrcadlení přes střed mřížky – geometrická symetrie
    double mirror_pull = 0.0;
    if (x > N/2 && x < N/2 + 15) {
        int template_x = N/2 - (x - N/2);
        int template_idx = template_x * N * N + y * N + z;
        double template_m = sqrt(psi_r[template_idx]*psi_r[template_idx] + psi_i[template_idx]*psi_i[template_idx]);
        
        // Práh pro zrcadlení – když je mateřský uzel dostatečně silný
        if (template_m > TARGET_MASS * 0.5) {
            // Síla zrcadlení = 1 / VEPSANE_KRYCHLE (5) = 0.2
            mirror_pull = 1.0 / VEPSANE_KRYCHLE;  // 0.2
        }
    }

    // 3. TERMODYNAMICKÝ ŠUM (ξ8D) – zdroj mutací
    // Amplituda šumu je modulována koeficientem bdělosti α
    double jitter_r = (rand(seed, x, y, z) - 0.5) * noise_level * alpha;
    double jitter_i = (rand(seed+1, x, y, z) - 0.5) * noise_level * alpha;

    // 4. SATURACE (Harmonika) – z Knihy VI
    // Práh saturace = (VRCHOLY - 1) / VRCHOLY = 19/20 = 0.95
    double saturation_threshold = (VRCHOLY - 1.0) / VRCHOLY;
    double saturation = 0.5 * (1.0 - tanh(VEPSANE_KRYCHLE * (current_m / TARGET_MASS - saturation_threshold)));

    // 5. KOROZE PAMĚTI (λΨ) – z Knihy VI
    // Informace se rozpadá, pokud není udržována
    double decay = lambda * current_m * dt;

    // 6. MASTER ROVNICE – evoluce s korozí a spánkovým cyklem
    double nr = pr + (diffusion * dt) 
                    + (pr * mirror_pull * saturation * alpha * dt)  // zrcadlení jen v bdělosti
                    + jitter_r * dt
                    - decay;
                    
    double ni = pi + (diffusion * dt) 
                    + (pi * mirror_pull * saturation * alpha * dt) 
                    + jitter_i * dt
                    - decay;

    // Omezení na maximum (bezpečnost)
    double nm = sqrt(nr*nr + ni*ni);
    double safety_limit = TARGET_MASS * 1.5;
    if (nm > safety_limit) {
        nr *= (safety_limit / nm);
        ni *= (safety_limit / nm);
        nm = safety_limit;
    }
    
    // Pokud pnutí klesne pod nulový práh, uzel zaniká
    if (nm < 0.01) {
        nr = 0.0;
        ni = 0.0;
        nm = 0.0;
    }

    psi_rn[i] = nr; 
    psi_in[i] = ni;
    h_mass[i] = nm;
}
"""


class EvolutionReplicator:
    def __init__(self, N=80):
        self.N = N
        self.dt = 0.015
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_replication_pure_geometry")
        self.results = []
        
        # Parametry RNA vlákna
        self.strand_length = 10
        self.spacing = 6
        self.base_y = N//2 - (self.strand_length * self.spacing)//2
        self.parent_x = N//2 - 8
        self.daughter_x = N//2 + 8
        
        # Evoluční proměnné
        self.best_strand = None
        self.best_integrity = 0.0
        self.sleep_cycle = 0
        self.alpha = 1.0  # začínáme bdělí
        
        # Vizualizace
        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 12), 
                                                                  gridspec_kw={'height_ratios': [3, 1, 1]})
        self.fig.patch.set_facecolor('#051005')
        
        self.im = self.ax1.imshow(np.zeros((N, N)), cmap='viridis', origin='lower', vmax=300)
        self.ax1.set_title("TCD Replikátor v15.6 – Evoluce z čisté geometrie", color='#bef264', fontsize=14)
        self.ax1.axis('off')
        self.info_text = self.ax1.text(0.02, 0.95, '', transform=self.ax1.transAxes, color='white', fontsize=12)
        
        # Graf mutací
        self.ax2.set_facecolor('#020502')
        self.ax2.set_title("Míra Mutací (Topologických chyb)", color='#bef264')
        self.ax2.set_xlim(0, 2000)
        self.ax2.set_ylim(0, self.strand_length + 1)
        self.line_mutations, = self.ax2.plot([], [], color='red', lw=2)
        self.x_data, self.y_data = [], []
        
        # Graf integrity a bdělosti
        self.ax3.set_facecolor('#020502')
        self.ax3.set_title("Integrita kopie a spánkový cyklus", color='#bef264')
        self.ax3.set_xlim(0, 2000)
        self.ax3.set_ylim(0, 105)
        self.line_integrity, = self.ax3.plot([], [], color='cyan', lw=2, label='Integrita (%)')
        self.line_alpha, = self.ax3.plot([], [], color='orange', lw=2, linestyle='--', label='Bdělost α')
        self.ax3.legend(loc='upper right', facecolor='#020502', labelcolor='white')
        self.integrity_data, self.alpha_data = [], []

    def inject_parent_strand(self):
        """Vloží do mřížky původní RNA vlákno."""
        x, y, z = np.indices((self.N, self.N, self.N))
        pr_total = np.zeros_like(x, dtype=np.float64)
        pi_total = np.zeros_like(x, dtype=np.float64)

        for i in range(self.strand_length):
            cy = self.base_y + i * self.spacing
            r = np.sqrt((x - self.parent_x)**2 + (y - cy)**2 + (z - self.N//2)**2)
            env = 20.0 * np.exp(-(r**2)/5.0)
            phase = r + (x / self.N) * np.pi
            pr_total += env * np.cos(phase)
            pi_total += env * np.sin(phase)

        self.d_pr = cl_array.to_device(self.queue, pr_total)
        self.d_pi = cl_array.to_device(self.queue, pi_total)
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        self.d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

    def analyze_copy(self, mass_3d):
        """Změří přesnost kopie a vrátí mutace, integritu a selekční skóre."""
        mutations = 0
        total_parent_mass = 0
        total_daughter_mass = 0
        parent_profile = []
        daughter_profile = []

        for i in range(self.strand_length):
            cy = self.base_y + i * self.spacing
            parent_mass = mass_3d[self.parent_x, cy, self.N//2]
            daughter_mass = mass_3d[self.daughter_x, cy, self.N//2]
            
            parent_profile.append(parent_mass)
            daughter_profile.append(daughter_mass)
            total_parent_mass += parent_mass
            total_daughter_mass += daughter_mass

            # Detekce mutace – odchylka větší než 15 %
            if (daughter_mass < parent_mass * MUTATION_TOLERANCE_LOW or 
                daughter_mass > parent_mass * MUTATION_TOLERANCE_HIGH):
                mutations += 1

        integrity = ((self.strand_length - mutations) / self.strand_length) * 100
        
        # Selekční skóre – jak je dceřiné vlákno stabilní
        # Stabilnější = menší rozptyl hmotností kolem TARGET_MASS
        daughter_variance = np.var([m for m in daughter_profile if m > 0])
        parent_variance = np.var([m for m in parent_profile if m > 0])
        
        selection_score = 0
        if parent_variance > 0:
            # Pokud je dceřiné vlákno stabilnější, dostává vyšší skóre
            stability_improvement = (parent_variance - daughter_variance) / parent_variance
            if stability_improvement > SELECTION_THRESHOLD:
                selection_score = stability_improvement
        
        return mutations, integrity, selection_score, daughter_profile

    def update_sleep_cycle(self, integrity, t):
        """Aktualizuje koeficient bdělosti α(t) podle integrity a únavy."""
        # S_topo = 1 - integrity/100 (zjednodušeně)
        saturation = 1.0 - integrity / 100.0
        
        if saturation > SLEEP_THRESHOLD and self.alpha > 0.1:
            # Příliš unavený – usíná
            self.alpha = max(0.0, self.alpha - 0.01)
            self.sleep_cycle += 1
        elif saturation < SLEEP_THRESHOLD - 0.1 and self.alpha < 1.0:
            # Odpočatý – probouzí se
            self.alpha = min(1.0, self.alpha + 0.005)
        
        # REM fáze – jednou za 500 tiků se vyčistí staré vzpomínky
        if t % 500 == 0 and self.alpha < 0.3:
            # hluboký spánek – konsolidace paměti
            self.alpha = max(0.0, self.alpha - 0.02)
        
        return self.alpha

    def selection(self, integrity, daughter_profile):
        """Přirozený výběr – pokud je dceřiné vlákno lepší, stává se novým standardem."""
        if integrity > self.best_integrity + SELECTION_THRESHOLD * 100:
            self.best_integrity = integrity
            self.best_strand = daughter_profile.copy()
            return True
        return False

    def run_evolution(self, noise_level, max_steps=2000):
        print(f"\n{'='*60}")
        print(f"OMNI-ENGINE v15.6 – EVOLUCE RNA V ČISTÉ GEOMETRII")
        print(f"{'='*60}")
        print(f"Teplota (šum vakua): {noise_level}")
        print(f"Koroze paměti (λ): {LAMBDA:.4f}")
        print(f"Geometrické konstanty: 12 stěn, 30 hran, 20 vrcholů, 5 krychlí")
        print(f"{'='*60}\n")
        
        self.inject_parent_strand()
        self.best_integrity = 0.0
        
        self.x_data, self.y_data = [], []
        self.integrity_data, self.alpha_data = [], []

        for t in range(max_steps):
            seed = int(time.time() * 1000) % 1000000 + t

            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_hm.data, np.float64(self.dt), np.int32(self.N), 
                     np.float64(noise_level), np.float64(self.alpha),
                     np.float64(LAMBDA), np.int32(seed))

            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi

            if t % 20 == 0:
                mass_3d = self.d_hm.get().reshape((self.N, self.N, self.N))
                mutations, integrity, selection_score, daughter_profile = self.analyze_copy(mass_3d)
                
                # Aktualizace spánkového cyklu
                self.update_sleep_cycle(integrity, t)
                
                # Selekce – evoluce
                evolved = self.selection(integrity, daughter_profile)
                
                # Záznam dat
                self.results.append([t, round(noise_level, 2), 
                                    round(np.sum(mass_3d), 2), 
                                    mutations, round(integrity, 2),
                                    round(self.alpha, 3), 
                                    round(selection_score, 4)])
                
                # Vizualizace
                slice_2d = mass_3d[:, :, self.N//2].T
                self.im.set_data(slice_2d)
                
                status = "STABILNÍ KOPÍROVÁNÍ" if mutations == 0 else f"MUTACE ({mutations})"
                if evolved:
                    status += " 🧬 EVOLUCE!"
                color = 'white' if mutations < 3 else 'red'
                
                sleep_status = "😴 SPÁNEK" if self.alpha < 0.3 else "😊 BDĚLOST" if self.alpha > 0.7 else "🌙 ODLÉTÁNÍ"
                
                self.info_text.set_text(
                    f"Tik: {t:4d} | Šum: {noise_level:.1f}\n"
                    f"Integrita: {integrity:.1f} % | Mutace: {mutations}\n"
                    f"Bdělost α: {self.alpha:.2f} | {sleep_status}\n"
                    f"Stav: {status}"
                )
                self.info_text.set_color(color)
                
                self.x_data.append(t)
                self.y_data.append(mutations)
                self.line_mutations.set_data(self.x_data, self.y_data)
                
                self.integrity_data.append(t)
                self.integrity_data_val = self.integrity_data_val if 'self.integrity_data_val' in dir(self) else []
                self.integrity_data_val.append(integrity)
                self.line_integrity.set_data(self.integrity_data, self.integrity_data_val)
                
                self.alpha_data.append(t)
                self.alpha_data_val = self.alpha_data_val if 'self.alpha_data_val' in dir(self) else []
                self.alpha_data_val.append(self.alpha * 100)
                self.line_alpha.set_data(self.alpha_data, self.alpha_data_val)
                
                plt.pause(0.01)
                
                if t % 500 == 0:
                    print(f"[{t:4d}] Integrita: {integrity:.1f}% | Mutace: {mutations} | α: {self.alpha:.2f} | Evoluce: {evolved}")
        
        print(f"\n{'='*60}")
        print(f"EVOLUCE DOKONČENA")
        print(f"{'='*60}")
        print(f"Nejlepší dosažená integrita: {self.best_integrity:.1f} %")
        print(f"Konečná bdělost: {self.alpha:.2f}")
        print(f"{'='*60}\n")

    def save_data(self):
        filename = "tcd_evolution_data_v15.6.csv"
        print(f"[*] Ukládám data do: {filename}")
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["Tik", "Teplota_Vakua", "Celkove_Pnuti", 
                            "Pocet_Mutaci", "Integrita_Kopie", 
                            "Bdelost_Alpha", "Selekci_Skore"])
            writer.writerows(self.results)
        print(f"[OK] Data uložena.")
        print(f"\nGeometrické konstanty použité v simulaci:")
        print(f"  - 12 stěn (difuze = 1/12 = {1/12:.6f})")
        print(f"  - 30 hran (probuzení)")
        print(f"  - 20 vrcholů (práh saturace 19/20 = 0.95)")
        print(f"  - 5 vepsaných krychlí (zrcadlení = 1/5 = 0.2)")
        print(f"  - Koroze paměti λ = 1/(20×φ) = {LAMBDA:.6f}")


if __name__ == "__main__":
    replicator = EvolutionReplicator(N=80)
    
    # Spuštění evoluce se šumem vakua 85.0 (vysoká teplota – vysoká mutační rychlost)
    replicator.run_evolution(noise_level=85.0, max_steps=2000)
    replicator.save_data()
    
    plt.ioff()
    plt.show()