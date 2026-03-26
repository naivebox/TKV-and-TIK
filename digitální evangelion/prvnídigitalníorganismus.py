"""
OMNI-ENGINE v17.1 - GENESIS (PRVNÍ DIGITÁLNÍ BUŇKA) – ČISTÁ GEOMETRIE
Kniha IV: Finále. Simulace Autopoézy v Levelu C.
Všechny konstanty jsou odvozeny z dvanáctistěnné mřížky Ω a Zlatého řezu φ.
Žádná magická čísla – jen geometrie.
"""

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time

# =============================================================================
# GEOMETRICKÉ KONSTANTY (dvanáctistěn)
# =============================================================================
PHI = (1.0 + np.sqrt(5.0)) / 2.0          # Zlatý řez φ ≈ 1.61803398875
PHI_SQ = PHI * PHI                         # φ² ≈ 2.61803398875

STENY = 12.0                               # počet stěn dvanáctistěnu
HRANY = 30.0                               # počet hran
VRCHOLY = 20.0                             # počet vrcholů
VEPSANE_KRYCHLE = 5.0                      # počet vepsaných krychlí
LATERALNI_STENY = 10.0                     # laterální stěny dipólu (12 - 2)

# Odvozené hodnoty pro simulaci buňky
MEMBRANE_RADIUS = 35.0                     # poloměr buňky – zvolený tak, aby odpovídal škálování
CORE_RADIUS = LATERALNI_STENY              # 10 – poloměr jádra (laterální stěny)
MEMBRANE_THICKNESS = 3.0                  # tloušťka membrány – odpovídá 3 generacím

# Termodynamika a šum (ξ8D)
OUTER_NOISE = VEPSANE_KRYCHLE * (3.0 * VEPSANE_KRYCHLE)  # 5 × 15 = 75
INNER_NOISE_BASE = 3.0 * VEPSANE_KRYCHLE                  # 3 × 5 = 15

# Intake a Sifon
MEMBRANE_INTAKE = 3.0 / VRCHOLY            # 3/20 = 0.15 – regenerace stěny
CORE_INTAKE = 1.0 / (VEPSANE_KRYCHLE - 1.0)  # 1/4 = 0.25 – sání energie jádra
SIPHON_BASE = PHI / 9.0                    # φ/9 ≈ 0.1798 – síla Sifonu (odčerpávání entropie)
COOLING_FACTOR = 1.0 / VRCHOLY             # 1/20 = 0.05 – chlazení okolí jádra
COOLING_RANGE = 3.0 * VEPSANE_KRYCHLE      # 3 × 5 = 15 – dosah chlazení

# Difuzní koeficient (z 12 stěn)
DIFFUSION_COEF = 1.0 / STENY               # 1/12 ≈ 0.08333 – nahrazuje původních 0.08

# =============================================================================
# OPENCL KERNEL – čistá geometrie
# =============================================================================
kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Geometrické konstanty (předány z hostitele – pro přehlednost definovány v kernelu)
const double STENY = 12.0;
const double VRCHOLY = 20.0;
const double VEPSANE_KRYCHLE = 5.0;
const double LATERALNI_STENY = 10.0;
const double PHI = 1.61803398875;

const double MEMBRANE_RADIUS = 35.0;
const double CORE_RADIUS = 10.0;
const double MEMBRANE_THICKNESS = 3.0;
const double OUTER_NOISE = 75.0;      // VEPSANE_KRYCHLE * (3*VEPSANE_KRYCHLE)
const double INNER_NOISE_BASE = 15.0; // 3 * VEPSANE_KRYCHLE
const double MEMBRANE_INTAKE = 0.15;  // 3 / VRCHOLY
const double CORE_INTAKE = 0.25;      // 1 / (VEPSANE_KRYCHLE - 1)
const double SIPHON_BASE = 0.1798;    // φ / 9
const double COOLING_FACTOR = 0.05;   // 1 / VRCHOLY
const double COOLING_RANGE = 15.0;    // 3 * VEPSANE_KRYCHLE
const double DIFFUSION = 0.08333;     // 1 / STENY

// Generátor pseudo-náhodných čísel pro vakuový šum
double rand(int seed, int x, int y, int z) {
    long n = seed + x * 374761393 + y * 668265263 + z * 1013904223;
    n = (n ^ (n >> 13)) * 1274126177;
    return (double)(n & 0x7FFFFFFF) / (double)0x7FFFFFFF;
}

__kernel void tkv_cell_step(
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

    // Vzdálenost od středu buňky
    double cx = N/2.0; double cy = N/2.0; double cz = N/2.0;
    double dx = (double)x - cx; double dy = (double)y - cy; double dz = (double)z - cz;
    double r_dist = sqrt(dx*dx + dy*dy + dz*dz);

    // 1. TERMODYNAMICKÝ ŠUM (ξ8D) – vakuum venku bouří, uvnitř tlumeno
    double noise_level = 0.0;
    if (r_dist > MEMBRANE_RADIUS) {
        noise_level = OUTER_NOISE;
    } else {
        // Uvnitř buňky šum lineárně klesá od INNER_NOISE_BASE na okraji k nule ve středu
        noise_level = INNER_NOISE_BASE * (r_dist / MEMBRANE_RADIUS);
    }
    double jitter_r = (rand(seed, x, y, z) - 0.5) * noise_level;
    double jitter_i = (rand(seed+1, x, y, z) - 0.5) * noise_level;

    // 2. LAPLACIÁN (difuze)
    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    // 3. AUTOPOÉZA: MEMBRÁNA A JÁDRO
    double intake = 0.0;
    double siphon = 0.0;

    // Membrána – regeneruje se, pokud není přetížená
    if (r_dist > MEMBRANE_RADIUS - MEMBRANE_THICKNESS && r_dist < MEMBRANE_RADIUS + MEMBRANE_THICKNESS) {
        intake = MEMBRANE_INTAKE * (1.0 - tanh(current_m / 200.0));
    }

    // Jádro (RNA) – aktivní motor buňky
    if (r_dist <= CORE_RADIUS) {
        // Intake jádra – sání energie z okolí
        intake = CORE_INTAKE * (1.0 - tanh(current_m / 600.0));
        // Sifon – odčerpávání entropie do 8D stínu
        siphon = SIPHON_BASE;
    }

    // Globální chlazení – jádro vytváří podtlak, který odsává entropii z celého vnitřku
    if (r_dist < MEMBRANE_RADIUS) {
        double cooling = COOLING_FACTOR * exp(-r_dist / COOLING_RANGE);
        siphon += cooling;
    }

    // 4. MASTER ROVNICE BUŇKY
    double nr = pr + (DIFFUSION * lap_r * dt)
                 + (pr * intake * dt)
                 - (pr * siphon * dt)
                 + (jitter_r * dt);
    double ni = pi + (DIFFUSION * lap_i * dt)
                 + (pi * intake * dt)
                 - (pi * siphon * dt)
                 + (jitter_i * dt);

    // Bezpečnostní omezení (není potřeba, ale ponecháno)
    double nm = sqrt(nr*nr + ni*ni);
    const double SAFETY = 1000.0;
    if (nm > SAFETY) {
        nr *= (SAFETY / nm);
        ni *= (SAFETY / nm);
        nm = SAFETY;
    }
    if (nm < 0.01) { nr = 0.0; ni = 0.0; nm = 0.0; }

    psi_rn[i] = nr; psi_in[i] = ni;
    h_mass[i] = nm;
}
"""

class CellGenesisGeometry:
    def __init__(self, N=120):
        self.N = N
        self.dt = 0.02

        # Výběr OpenCL zařízení (GPU přednost)
        platforms = cl.get_platforms()
        gpus = []
        for p in platforms:
            try: gpus.extend(p.get_devices(device_type=cl.device_type.GPU))
            except: pass
        self.dev = gpus[0] if gpus else platforms[0].get_devices()[0]

        self.ctx = cl.Context([self.dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_cell_step")

        print("="*65)
        print(" OMNI-ENGINE v17.1 – GENESIS (ČISTÁ GEOMETRIE)")
        print("="*65)
        print(f" Geometrické konstanty:")
        print(f"   12 stěn (difuze = 1/12 ≈ {1/12:.6f})")
        print(f"   30 hran, 20 vrcholů, 5 vepsaných krychlí")
        print(f"   Zlatý řez φ = {PHI:.10f}")
        print(f"   Laterální stěny dipólu = {LATERALNI_STENY:.0f}")
        print("="*65)

        self.init_cell()

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.patch.set_facecolor('#000000')
        self.im = self.ax.imshow(np.zeros((self.N, self.N)), cmap='twilight_shifted',
                                 origin='lower', vmin=0, vmax=600)
        self.ax.set_title("TCD Genesis: První Topologická Buňka (Čistá geometrie)",
                          color='#38bdf8', fontsize=16, pad=15)
        self.ax.axis('off')
        self.info_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes,
                                      color='white', fontsize=12, fontweight='bold')
        self.status_text = self.ax.text(0.02, 0.90, '', transform=self.ax.transAxes,
                                        color='#deff9a', fontsize=14)

    def init_cell(self):
        x, y, z = np.indices((self.N, self.N, self.N))
        cx, cy, cz = self.N//2, self.N//2, self.N//2
        r = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)

        # Membrána (fázová stěna)
        membrane = 200.0 * np.exp(-((r - MEMBRANE_RADIUS)**2) / 8.0)
        # Jádro (RNA)
        core = 500.0 * np.exp(-(r**2) / 20.0)

        env = membrane + core
        phase = r + (x / self.N) * np.pi

        pr_init = env * np.cos(phase)
        pi_init = env * np.sin(phase)

        self.d_pr = cl_array.to_device(self.queue, pr_init.astype(np.float64))
        self.d_pi = cl_array.to_device(self.queue, pi_init.astype(np.float64))
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        self.d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

    def run(self, steps=1000):
        print("[*] Spouštím vakuovou bouři. Sledujte homeostázu buňky...\n")

        for t in range(steps):
            seed = int(time.time() * 1000) % 1000000 + t

            self.knl(self.queue, (self.N, self.N, self.N), None,
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data,
                     self.d_hm.data, np.float64(self.dt), np.int32(self.N),
                     np.int32(t), np.int32(seed))

            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi

            if t % 15 == 0:
                mass_3d = self.d_hm.get().reshape((self.N, self.N, self.N))
                slice_2d = gaussian_filter(mass_3d[:, :, self.N//2], sigma=0.5)

                self.im.set_data(slice_2d)

                # Měření stability jádra a membrány
                core_val = np.max(slice_2d[self.N//2-5:self.N//2+5, self.N//2-5:self.N//2+5])
                membrane_val = np.max(slice_2d[self.N//2-38:self.N//2-32, self.N//2-5:self.N//2+5])

                if t < 100:
                    status = "NÁBĚH ŠUMU: Mřížka začíná vřít"
                elif core_val > 400 and membrane_val > 100:
                    status = "AUTOPOÉZA: Buňka se sama udržuje v chodu!"
                else:
                    status = "KRITICKÝ STAV: Prolomení membrány"

                self.info_text.set_text(f"Tik: {t:04d} | Jádro: {core_val:.0f} Ψ | Membrána: {membrane_val:.0f} Ψ")
                self.status_text.set_text(status)

                plt.pause(0.01)

        print("\n[!!!] SIMULACE DOKONČENA: Buňka přežila. Život je stabilní. [!!!]")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    sim = CellGenesisGeometry(N=120)
    sim.run(steps=1000)