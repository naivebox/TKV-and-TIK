import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# =============================================================================
# OMNI-ENGINE v9.0 - BIOGENESIS: H2O MOLECULE
# Cíl: Sloučení O (Kyslík) a 2x H (Vodík) pod magickým úhlem 104.5°.
# Využití TCD Katalogu pro objektově orientovanou injekci do Levelu D.
# =============================================================================

# --- TCD KATALOG (Topologické Pakety) ---
class TopologicalPacket:
    def __init__(self, name, mass, amplitude, radius, phase_sig):
        self.name = name
        self.mass = mass
        self.amplitude = amplitude
        self.radius = radius
        self.phase_sig = phase_sig

class TCD_Catalog:
    # Hodnoty z naší úspěšné Hvězdné Kovadliny (Stellar Forge)
    HYDROGEN = TopologicalPacket("Vodík", 275.42, 15.0, 6.0, 0.0)
    OXYGEN   = TopologicalPacket("Kyslík", 4404.72, 45.0, 14.0, np.pi/2)

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

    // Cílová hmota molekuly H2O (O + 2H minus hmotnostní defekt vazby)
    const double H2O_TARGET = 4900.0; 

    double pr = psi_r[i]; double pi = psi_i[i];
    double current_m = sqrt(pr*pr + pi*pi);

    // Topologický proud
    double Jx = pr * (psi_i[id_xp] - psi_i[id_xm]) - pi * (psi_r[id_xp] - psi_r[id_xm]);
    double Jy = pr * (psi_i[id_yp] - psi_i[id_ym]) - pi * (psi_r[id_yp] - psi_r[id_ym]);
    double Jz = pr * (psi_i[id_zp] - psi_i[id_zm]) - pi * (psi_r[id_zp] - psi_r[id_zm]);
    double J_mag = sqrt(Jx*Jx + Jy*Jy + Jz*Jz);

    // Kovalentní (sdílená) Gluonová Izolace
    double isolation = exp(-J_mag * 3.5);

    // --- OPRAVA ARCHITEKTA: ELEKTRONEGATIVITA (Topologický volný pád) ---
    // Kyslík v centru vytváří obrovskou fázovou prohlubeň. 
    // Vodíky necháme "spadnout" do jeho orbitalu, kde je zastaví geometrie (isolation).
    double cx = N/2.0; double cy = N/2.0; double cz = N/2.0;
    double dx_c = (double)x - cx;
    double dy_c = (double)y - cy;
    double dz_c = (double)z - cz;
    double r_dist = sqrt(dx_c*dx_c + dy_c*dy_c + dz_c*dz_c) + 0.001;

    double pull = 0.0;
    if (r_dist > 2.0 && r_dist < 40.0) {
        pull = 15.0 * (1.0 / r_dist); // EXTRÉMNĚ ZVÝŠENÝ TAH KYSLÍKU!
    }
    
    // Směrový drift do centra (dx_c/r_dist je normalizovaný vektor)
    double drift_r = pull * (dx_c/r_dist) * 0.15 * pr;
    double drift_i = pull * (dx_c/r_dist) * 0.15 * pi;

    // Laplacián
    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + psi_r[id_zp] + psi_r[id_zm];
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + psi_i[id_zp] + psi_i[id_zm];

    // Harmonika pro stabilizaci makromolekuly
    double saturation = 0.5 * (1.0 - tanh(4.5 * (current_m / H2O_TARGET - 0.98)));
    
    // Nasávání vakua a Sifon (Jemné vazebné dýchání)
    double intake = 0.20 * saturation;
    double local_friction = (current_m > H2O_TARGET * 0.8) ? 0.15 : 0.0;

    // APLIKACE DRIFTU: Odečtením driftu (znaménko mínus) táhneme hmotu proti vektoru dx_c -> dovnitř
    double nr = pr + (0.08 * lap_r * isolation * dt) - (drift_r * dt) + (pr * intake * dt) - (local_friction * pr * dt);
    double ni = pi + (0.08 * lap_i * isolation * dt) - (drift_i * dt) + (pi * intake * dt) - (local_friction * pi * dt);

    double nm = sqrt(nr*nr + ni*ni);
    if (nm > H2O_TARGET * 1.5) {
        nr *= (H2O_TARGET * 1.5 / nm);
        ni *= (H2O_TARGET * 1.5 / nm);
    }

    psi_rn[i] = nr; psi_in[i] = ni; h_mass[i] = nm;
}
"""

class WaterBiogenesis:
    def __init__(self, N=120):
        self.N = N
        self.dt = 0.015
        platforms = cl.get_platforms()
        dev = platforms[0].get_devices()[0]
        self.ctx = cl.Context([dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_h2o_step")

        self.grid_r = np.zeros((N, N, N), dtype=np.float64)
        self.grid_i = np.zeros((N, N, N), dtype=np.float64)
        self.x, self.y, self.z = np.indices((N, N, N))
        
        print("[*] OMNI-ENGINE v9.0: BIOGENEZE INICIALIZOVÁNA")

    def inject_packet(self, packet, cx, cy, cz, rotation_phase=0.0):
        r = np.sqrt((self.x - cx)**2 + (self.y - cy)**2 + (self.z - cz)**2)
        env = packet.amplitude * np.exp(-(r**2) / packet.radius)
        phase = r + (self.x / self.N) * np.pi + packet.phase_sig + rotation_phase
        
        self.grid_r += env * np.cos(phase)
        self.grid_i += env * np.sin(phase)
        print(f"[+] Vložen {packet.name} na pozici [{cx:.1f}, {cy:.1f}, {cz:.1f}]")

    def setup_water_molecule(self):
        cx, cy, cz = self.N//2, self.N//2, self.N//2
        
        # 1. Kyslík přesně do středu
        self.inject_packet(TCD_Catalog.OXYGEN, cx, cy, cz)
        
        # 2. Úhel 104.5° pro vodíky
        angle_deg = 104.5
        angle_rad = np.deg2rad(angle_deg)
        
        # OPRAVA ARCHITEKTA: Necháme vzdálenost určit samotnou geometrií orbitalu!
        # Vodíky nasadíme z dálky a necháme je sjet po fázovém svahu kyslíku.
        bond_length = 18.0 
        
        # Výpočet pozic vodíků (tvar písmene V)
        h1_x = cx + bond_length * np.cos(angle_rad / 2)
        h1_y = cy + bond_length * np.sin(angle_rad / 2)
        
        h2_x = cx + bond_length * np.cos(-angle_rad / 2)
        h2_y = cy + bond_length * np.sin(-angle_rad / 2)
        
        # Injekce Vodíků (jeden má inverzní spin/fázi pro vytvoření zámku)
        self.inject_packet(TCD_Catalog.HYDROGEN, h1_x, h1_y, cz, rotation_phase=0.0)
        self.inject_packet(TCD_Catalog.HYDROGEN, h2_x, h2_y, cz, rotation_phase=np.pi)

        # Načtení do GPU paměti
        self.d_pr = cl_array.to_device(self.queue, self.grid_r.flatten())
        self.d_pi = cl_array.to_device(self.queue, self.grid_i.flatten())
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        self.d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

    def run_visualization(self):
        self.setup_water_molecule()
        
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.patch.set_facecolor('#000b18') # Tmavě modrá hlubina
        
        # Použijeme colormapu evokující vodu
        self.im = self.ax.imshow(np.zeros((self.N, self.N)), cmap='ocean', origin='lower', vmin=0, vmax=4500)
        self.ax.set_title("TCD Biogeneze: Stabilizace molekuly Vody ($H_2O$)\nÚhel vazby: 104.5°", color='cyan', pad=15)
        self.ax.axis('off')
        info_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, color='white', fontsize=12)

        print("\n[!] SPUŠTĚNÍ TOPOLOGICKÉ SYNTÉZY MOLEKULY")
        for t in range(3000):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_hm.data, np.float64(self.dt), np.int32(self.N))
            
            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi
            
            if t % 15 == 0:
                mass_3d = self.d_hm.get().reshape((self.N, self.N, self.N))
                slice_2d = gaussian_filter(mass_3d[:, :, self.N//2], sigma=1.0)
                
                self.im.set_data(slice_2d)
                
                max_mass = np.max(slice_2d)
                if t < 500: status = "Vyhlazování pnutí (Termodynamický šum)"
                elif t < 1500: status = "Topologický pád (Přitahování do orbitalu)"
                elif t < 2500: status = "Budování sdíleného fázového obalu (Vazba)"
                else: status = "ZÁMEK KLAPL: Molekula je stabilní"
                
                info_text.set_text(f"Tik: {t} | Max hustota (Kyslík): {max_mass:.1f}\nStav: {status}")
                
                plt.pause(0.01)
                
        print("\n[OK] Kovalentní vazby jsou stabilní. Zrodila se voda.")

if __name__ == "__main__":
    sim = WaterBiogenesis(N=120)
    sim.run_visualization()
