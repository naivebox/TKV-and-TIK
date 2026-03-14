import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import matplotlib.animation as animation
import os

# =====================================================================
# TKV KOSMOLOGICKÝ INKUBÁTOR S VIDEO VÝSTUPEM
# Cíl: Záznam zrodu Protonu v Kvantové Pěně a generování animace
# =====================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void tkv_incubator_step(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *Ax, __global double *Ay, __global double *Az,
    __global double *h_total_energy,
    const double dt, const int N, const double time,
    const int physics_mode,     // 0 = 1D Trubice (Fúze), 1 = Plné 3D (QCD pěna)
    const double vacuum_heat)   // Síla tepelného šumu (excitace vakua)
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
    
    // Z-osa je vždy aktivní
    rzp = psi_r[id_zp]*cos(az) - psi_i[id_zp]*sin(az);
    izp = psi_r[id_zp]*sin(az) + psi_i[id_zp]*cos(az);
    rzm = psi_r[id_zm]*cos(-az) - psi_i[id_zm]*sin(-az);
    izm = psi_r[id_zm]*sin(-az) + psi_i[id_zm]*cos(-az);
    Jz = pr * (psi_i[id_zp] - psi_i[id_zm]) - pi * (psi_r[id_zp] - psi_r[id_zm]);

    if (physics_mode == 0) {
        // 1D Trubice
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
    
    double jitter = sin((double)x*1.618 + (double)y*3.14 + (double)z*2.71 + time*15.0) * vacuum_heat;

    double nr = pr + (0.05 * lap_r * dt) + (tenze * pr * dt) - (siphon_friction * pr * dt) + jitter;
    double ni = pi + (0.05 * lap_i * dt) + (tenze * pi * dt) - (siphon_friction * pi * dt) + jitter;

    double nm = sqrt(nr*nr + ni*ni);
    double u = (nm > 3.5) ? (0.0) : (nm > 1.0 ? 1.0/nm : 1.0);
    
    psi_rn[i] = nr * u; psi_in[i] = ni * u;

    double mass_energy = nm * nm;
    double kinetic_energy = 0.5 * (lap_r*lap_r + lap_i*lap_i);
    double gluon_energy = LAMBDA_W * ALPHA_G * (Ax[i]*Ax[i] + Ay[i]*Ay[i] + Az[i]*Az[i]);

    h_total_energy[i] = mass_energy + kinetic_energy + gluon_energy;
}
"""

class TKV_Incubator_Video:
    def __init__(self, size=64):
        self.N = size
        self.dt = 0.005
        self.global_time = 0.0
        
        platforms = cl.get_platforms()
        dev = platforms[0].get_devices()[0]
        self.ctx = cl.Context([dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_incubator_step")
        
        print(f"[+] Inkubátor online (GPU: {dev.name})")

        self.d_pr = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_pi = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        
        self.d_ax = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_ay = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_az = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_total_energy = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        
        # Pole pro uchování dat jednotlivých snímků
        self.frames_data = []

    def inject_quarks(self, energy):
        print(f"[+] Injektuji 3 kvarky (Energie: {energy})...")
        x, y, z = np.indices((self.N, self.N, self.N))
        dx, dy, dz = x - self.N/2.0, y - self.N/2.0, z - self.N/2.0
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        
        obalka = 0.1 + 0.9 * np.exp(-(r**2) / 8.0)
        
        phase_z = (z / self.N) * np.pi * energy 
        phase_x = (x / self.N) * np.pi * energy + (2.0/3.0)*np.pi
        phase_y = (y / self.N) * np.pi * energy + (4.0/3.0)*np.pi
        total_phase = phase_z + phase_x + phase_y
        
        pr = (obalka * np.cos(total_phase)).flatten().astype(np.float64)
        pi = (obalka * np.sin(total_phase)).flatten().astype(np.float64)
        
        self.d_pr = cl_array.to_device(self.queue, pr)
        self.d_pi = cl_array.to_device(self.queue, pi)
        self.d_ax.fill(0.0); self.d_ay.fill(0.0); self.d_az.fill(0.0)

    def capture_frame(self, frame_id, phase_name):
        """Stáhne aktuální data z GPU a uloží je pro video."""
        psi_r_np = self.d_pr.get().reshape((self.N, self.N, self.N))
        psi_i_np = self.d_pi.get().reshape((self.N, self.N, self.N))
        amplitude = np.sqrt(psi_r_np**2 + psi_i_np**2)
        
        ax_np = self.d_ax.get().reshape((self.N, self.N, self.N))
        ay_np = self.d_ay.get().reshape((self.N, self.N, self.N))
        az_np = self.d_az.get().reshape((self.N, self.N, self.N))
        gluon_field = np.sqrt(ax_np**2 + ay_np**2 + az_np**2)
        
        self.frames_data.append({
            'id': frame_id,
            'phase': phase_name,
            'amplitude': amplitude.copy(),
            'gluon': gluon_field.copy()
        })
        print(f"  -> Snímek {frame_id} uložen ({phase_name})")

    def run_phase(self, ticks, physics_mode, vacuum_heat, phase_name, capture_interval=200):
        print(f"\n--- {phase_name} ---")
        print(f"Mód: {'1D Trubice' if physics_mode == 0 else 'Plné 3D'} | Pěna: {vacuum_heat}")
        
        frame_count = 0
        for t in range(ticks):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_ax.data, self.d_ay.data, self.d_az.data, self.d_total_energy.data, 
                     np.float64(self.dt), np.int32(self.N), np.float64(self.global_time),
                     np.int32(physics_mode), np.float64(vacuum_heat))
            
            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi
            self.global_time += self.dt
            
            if t > 0 and t % capture_interval == 0:
                self.queue.finish()
                self.capture_frame(len(self.frames_data), phase_name)

    def generate_video(self, filename="tkv_proton_birth.mp4", level_quarks=1.5, level_gluons=0.05):
        print(f"\n[*] Generuji video ze {len(self.frames_data)} snímků...")
        if len(self.frames_data) == 0:
            print("[!] Žádná data pro video.")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update_graph(frame_index):
            # ax.clear() dělá ve 3D animacích chyby (IndexError). 
            # Místo toho bezpečně smažeme jen objekty (kolekce):
            while ax.collections:
                for c in ax.collections:
                    c.remove()
                    
            data = self.frames_data[frame_index]
            ax.set_title(f"TKV Zrod Protonu\n{data['phase']} | Snímek: {data['id']}")
            ax.set_xlim(0, self.N); ax.set_ylim(0, self.N); ax.set_zlim(0, self.N)
            ax.set_axis_off() # Vypneme mřížku pro lepší vzhled

            # Rotace kamery v průběhu videa
            ax.view_init(elev=20, azim=-60 + (frame_index * 2))

            # Kreslení gluonů
            try:
                verts_g, faces_g, _, _ = measure.marching_cubes(data['gluon'], level_gluons)
                if len(verts_g) > 0 and len(faces_g) > 0:
                    mesh_g = Poly3DCollection(verts_g[faces_g], alpha=0.15, facecolor='#00BFFF')
                    ax.add_collection3d(mesh_g)
            except Exception:
                pass # Ignoruj, pokud mrak ještě neexistuje nebo je prázdný

            # Kreslení kvarků
            try:
                verts_q, faces_q, _, _ = measure.marching_cubes(data['amplitude'], level_quarks)
                if len(verts_q) > 0 and len(faces_q) > 0:
                    mesh_q = Poly3DCollection(verts_q[faces_q], alpha=0.8, facecolor='#FF4500')
                    ax.add_collection3d(mesh_q)
            except Exception:
                pass

            return ax.collections

        ani = animation.FuncAnimation(fig, update_graph, frames=len(self.frames_data), interval=200, blit=False)
        
        try:
            print(f"[*] Ukládám video do: {filename}")
            ani.save(filename, writer='ffmpeg', fps=10, dpi=100)
            print("[+] Video úspěšně uloženo!")
        except Exception as e:
            print(f"[!] Chyba při ukládání videa pomocí ffmpeg: {e}")
            print("[*] Zkouším uložit jako GIF...")
            try:
                gif_filename = filename.replace('.mp4', '.gif')
                ani.save(gif_filename, writer='pillow', fps=10, dpi=80)
                print(f"[+] GIF úspěšně uložen jako {gif_filename}!")
            except Exception as e2:
                print(f"[!] Nepodařilo se uložit ani GIF: {e2}")

if __name__ == "__main__":
    incubator = TKV_Incubator_Video(size=64)
    
    # KROK 0: Suroviny
    incubator.inject_quarks(energy=5.0)
    
    # Zaznamenáme úplný začátek
    incubator.capture_frame(0, "Počáteční stav")
    
    # Fáze 1: Vynucená fúze (1D)
    # Záznam každých 250 tiků
    incubator.run_phase(ticks=1500, physics_mode=0, vacuum_heat=0.000, phase_name="Fáze 1: 1D Fúze", capture_interval=250)
    
    # Fáze 2: Plné 3D otevření (bez šumu)
    incubator.run_phase(ticks=1500, physics_mode=1, vacuum_heat=0.001, phase_name="Fáze 2: 3D Expanze", capture_interval=250)
    
    # Fáze 3: Vroucí vakuum (QCD pěna)
    incubator.run_phase(ticks=2000, physics_mode=1, vacuum_heat=0.015, phase_name="Fáze 3: Vroucí Kvantová Pěna", capture_interval=250)
    
    # VYTVOŘENÍ VIDEA
    # Může to trvat několik minut, než se všechny snímky vyrenderují!
    incubator.generate_video(filename="proton_evolution.mp4", level_quarks=1.8, level_gluons=0.08)
