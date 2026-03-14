import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

# =====================================================================
# TKV ŽIVÝ MIKROSKOP: KNIHA I - ZROD PROTONU V PŘÍMÉM PŘENOSU
# Včetně exportu kompletního 3D stavu vesmíru po zavření okna.
# =====================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void tkv_incubator_step(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *Ax, __global double *Ay, __global double *Az,
    __global double *h_total_energy,
    const double dt, const int N, const double time,
    const int physics_mode,     // 0 = 1D Trubice, 1 = Plné 3D
    const double vacuum_heat)   // Síla varu vakua
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

    const double LAMBDA_W = 0.0833333333; 
    const double ALPHA_G = 1.19998161486;

    double pr = psi_r[i]; double pi = psi_i[i];
    double ax = Ax[i]; double ay = Ay[i]; double az = Az[i];

    double dx = (double)x - (double)N/2.0;
    double dy = (double)y - (double)N/2.0;
    double dz = (double)z - (double)N/2.0;
    double r = sqrt(dx*dx + dy*dy + dz*dz);
    double siphon_friction = 0.5 * exp(-r / 1.5);

    double rxp, ixp, rxm, ixm, ryp, iyp, rym, iym, rzp, izp, rzm, izm;
    double Jx = 0.0; double Jy = 0.0; double Jz = 0.0;
    
    // Z-osa (Páteř trubice)
    rzp = psi_r[id_zp]*cos(az) - psi_i[id_zp]*sin(az);
    izp = psi_r[id_zp]*sin(az) + psi_i[id_zp]*cos(az);
    rzm = psi_r[id_zm]*cos(-az) - psi_i[id_zm]*sin(-az);
    izm = psi_r[id_zm]*sin(-az) + psi_i[id_zm]*cos(-az);
    Jz = pr * (psi_i[id_zp] - psi_i[id_zm]) - pi * (psi_r[id_zp] - psi_r[id_zm]);

    if (physics_mode == 0) {
        // 1D Fúze
        rxp = psi_r[id_xp]; ixp = psi_i[id_xp]; rxm = psi_r[id_xm]; ixm = psi_i[id_xm];
        ryp = psi_r[id_yp]; iyp = psi_i[id_yp]; rym = psi_r[id_ym]; iym = psi_i[id_ym];
    } else {
        // Plné 3D
        rxp = psi_r[id_xp]*cos(ax) - psi_i[id_xp]*sin(ax);
        ixp = psi_r[id_xp]*sin(ax) + psi_i[id_xp]*cos(ax);
        rxm = psi_r[id_xm]*cos(-ax) - psi_i[id_xm]*sin(-ax);
        ixm = psi_r[id_xm]*sin(-ax) + psi_i[id_xm]*cos(-ax);
        
        ryp = psi_r[id_yp]*cos(ay) - psi_i[id_yp]*sin(ay);
        iyp = psi_r[id_yp]*sin(ay) + psi_i[id_yp]*cos(ay);
        rym = psi_r[id_ym]*cos(-ay) - psi_i[id_ym]*sin(-ay);
        iym = psi_r[id_ym]*sin(-ay) + psi_i[id_ym]*cos(-ay);
        
        Jx = pr * (psi_i[id_xp] - psi_i[id_xm]) - pi * (psi_r[id_xp] - psi_r[id_xm]);
        Jy = pr * (psi_i[id_yp] - psi_i[id_ym]) - pi * (psi_r[id_yp] - psi_r[id_ym]);
    }

    double lap_r = -6.0*pr + rxp + rxm + ryp + rym + rzp + rzm;
    double lap_i = -6.0*pi + ixp + ixm + iyp + iym + izp + izm;

    Ax[i] = ax + (Jx * LAMBDA_W * ALPHA_G) * dt;
    Ay[i] = ay + (Jy * LAMBDA_W * ALPHA_G) * dt;
    Az[i] = az + (Jz * LAMBDA_W * ALPHA_G) * dt;

    double m_old = sqrt(pr*pr + pi*pi);
    double tenze = m_old * (1.0 - m_old*m_old);
    
    // Kvantová pěna
    double jitter = sin((double)x*1.618 + (double)y*3.14 + (double)z*2.71 + time*15.0) * vacuum_heat;

    double nr = pr + (0.05 * lap_r * dt) + (tenze * pr * dt) - (siphon_friction * pr * dt) + jitter;
    double ni = pi + (0.05 * lap_i * dt) + (tenze * pi * dt) - (siphon_friction * pi * dt) + jitter;

    double nm = sqrt(nr*nr + ni*ni);
    double u = (nm > 3.5) ? (0.0) : (nm > 1.0 ? 1.0/nm : 1.0);
    
    psi_rn[i] = nr * u; psi_in[i] = ni * u;

    // Pro mikroskop vracíme CELKOVOU energii (aby byl vidět i uzel i pole)
    double mass_energy = nm * nm;
    double gluon_energy = LAMBDA_W * ALPHA_G * (Ax[i]*Ax[i] + Ay[i]*Ay[i] + Az[i]*Az[i]);
    
    // Násobeno pro lepší viditelnost na monitoru
    h_total_energy[i] = mass_energy + (gluon_energy * 20.0);
}
"""

class TKV_Live_Microscope:
    def __init__(self, size=100):
        self.N = size
        self.dt = 0.005
        self.global_time = 0.0
        self.ticks = 0
        
        platforms = cl.get_platforms()
        dev = platforms[0].get_devices()[0]
        self.ctx = cl.Context([dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_incubator_step")
        print(f"[+] Mikroskop online (GPU: {dev.name})")

        self.d_pr = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_pi = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        
        self.d_ax = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_ay = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_az = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_total_energy = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

        self.inject_quarks(energy=5.0)

    def inject_quarks(self, energy):
        print("[+] Vkládám 3 fázově posunuté kvarky...")
        x, y, z = np.indices((self.N, self.N, self.N))
        dx, dy, dz = x - self.N/2.0, y - self.N/2.0, z - self.N/2.0
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        
        obalka = 0.1 + 0.9 * np.exp(-(r**2) / 8.0)
        
        # OPRAVA TOPOLOGIE: Skutočné víry (rotácia okolo osí) namiesto rovných vĺn!
        # Kvark 1: Rotácia v rovine XY
        phase_xy = np.arctan2(dy, dx) * energy 
        # Kvark 2: Rotácia v rovine XZ + fázový posun (farba)
        phase_xz = np.arctan2(dz, dx) * energy + (2.0/3.0)*np.pi
        # Kvark 3: Rotácia v rovine YZ + fázový posun (farba)
        phase_yz = np.arctan2(dz, dy) * energy + (4.0/3.0)*np.pi
        
        # Súčet troch ortogonálnych vírov vytvorí topologický uzol v strede
        total_phase = phase_xy + phase_xz + phase_yz
        
        pr = (obalka * np.cos(total_phase)).flatten().astype(np.float64)
        pi = (obalka * np.sin(total_phase)).flatten().astype(np.float64)
        
        self.d_pr = cl_array.to_device(self.queue, pr)
        self.d_pi = cl_array.to_device(self.queue, pi)

    def step(self):
        physics_mode = 0
        vacuum_heat = 0.0
        phase_name = "Fáze 1: Vynucená 1D Fúze"

        if self.ticks > 1500:
            physics_mode = 1
            vacuum_heat = 0.001
            phase_name = "Fáze 2: 3D Prostor (Uvolnění)"
        if self.ticks > 3000:
            physics_mode = 1
            vacuum_heat = 0.015
            phase_name = "Fáze 3: Vroucí QCD Pěna!"

        self.knl(self.queue, (self.N, self.N, self.N), None, 
                 self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                 self.d_ax.data, self.d_ay.data, self.d_az.data, self.d_total_energy.data, 
                 np.float64(self.dt), np.int32(self.N), np.float64(self.global_time),
                 np.int32(physics_mode), np.float64(vacuum_heat))
        
        self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
        self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi
        self.global_time += self.dt
        self.ticks += 1
        
        return phase_name

    def get_2d_slice(self):
        mid_z = self.N // 2
        return self.d_total_energy.get().reshape((self.N, self.N, self.N))[:, :, mid_z]

    def export_data(self, filename="tkv_proton_saved_state.npz"):
        """Uloží kompletní 3D stav do souboru po zavření simulace."""
        print(f"\n[*] Příprava k exportu dat... Stahuji data z grafické karty.")
        np.savez_compressed(
            filename,
            psi_r=self.d_pr.get().reshape((self.N, self.N, self.N)),
            psi_i=self.d_pi.get().reshape((self.N, self.N, self.N)),
            Ax=self.d_ax.get().reshape((self.N, self.N, self.N)),
            Ay=self.d_ay.get().reshape((self.N, self.N, self.N)),
            Az=self.d_az.get().reshape((self.N, self.N, self.N)),
            total_energy=self.d_total_energy.get().reshape((self.N, self.N, self.N)),
            ticks=self.ticks,
            global_time=self.global_time
        )
        print(f"[+] Kompletní 3D stav vesmíru byl úspěšně uložen do: {filename}")
        print("[+] Tento soubor můžeš v budoucnu načíst (pomocí np.load) a analyzovat proton v klidu.")


def run_live_microscope():
    sim = TKV_Live_Microscope(size=100) 
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.canvas.manager.set_window_title('TKV Live Microscope')
    
    im = ax.imshow(np.zeros((sim.N, sim.N)), cmap='magma', vmin=0, vmax=3.0, interpolation='bicubic')
    ax.axis('off')
    
    title_text = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center", va="bottom", color="white", fontsize=14, fontweight='bold')

    def update(frame):
        phase_name = ""
        for _ in range(15): 
            phase_name = sim.step()
        
        slice_2d = sim.get_2d_slice()
        enhanced_slice = np.log1p(slice_2d * 5.0) 
        
        im.set_array(enhanced_slice)
        title_text.set_text(f"{phase_name} | Tik: {sim.ticks}")
        
        return im, title_text

    print("[*] Spouštím okno mikroskopu... Sledujte monitor.")
    print("[i] INFO: Až budete chtít simulaci ukončit a ULOŽIT DATA, zavřete okno (kliknutím na X).")
    
    ani = animation.FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
    plt.tight_layout()
    
    # Kód se zde zablokuje, dokud uživatel nezavře okno
    plt.show() 
    
    # Jakmile uživatel zavře okno grafu, spustí se automaticky export
    sim.export_data()

if __name__ == "__main__":
    run_live_microscope()
