import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.pyplot as plt

# =====================================================================
# OMNI-ENGINE v5.4: PROJEKT VODÍK (ATOMOVÁ GENEZE)
# Simulace interakce stabilního Protonu a Elektronového mraku
# =====================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void tkv_hydrogen_step(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *Ax, __global double *Ay, __global double *Az,
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

    const double LAMBDA_W = 0.0833333333; 
    const double ALPHA_G = 1.19998161486;
    const double TARGET_RATIO = 275.4258; // Proton Homeostasis

    double pr = psi_r[i]; double pi = psi_i[i];
    double ax = Ax[i]; double ay = Ay[i]; double az = Az[i];

    // 1. TOPOLOGICKÝ PROUD
    double Jx = pr * (psi_i[id_xp] - psi_i[id_xm]) - pi * (psi_r[id_xp] - psi_r[id_xm]);
    double Jy = pr * (psi_i[id_yp] - psi_i[id_ym]) - pi * (psi_r[id_yp] - psi_r[id_ym]);
    double Jz = pr * (psi_i[id_zp] - psi_i[id_zm]) - pi * (psi_r[id_zp] - psi_r[id_zm]);
    double J_mag = sqrt(Jx*Jx + Jy*Jy + Jz*Jz);

    // 2. GLUONOVÁ IZOLACE & HARMONIKA (Nasycení)
    double isolation = exp(-J_mag * 5.0);
    double current_m = sqrt(pr*pr + pi*pi);
    double saturation = 0.5 * (1.0 - tanh(5.0 * (current_m / TARGET_RATIO - 0.95)));

    // 3. KOVARIANTNÍ LAPLACIÁN
    double rxp = psi_r[id_xp]*cos(ax) - psi_i[id_xp]*sin(ax);
    double ixp = psi_r[id_xp]*sin(ax) + psi_i[id_xp]*cos(ax);
    double rxm = psi_r[id_xm]*cos(-ax) - psi_i[id_xm]*sin(-ax);
    double ixm = psi_r[id_xm]*sin(-ax) + psi_i[id_xm]*cos(-ax);
    double ryp = psi_r[id_yp]*cos(ay) - psi_i[id_yp]*sin(ay);
    double iyp = psi_r[id_yp]*sin(ay) + psi_i[id_yp]*cos(ay);
    double rym = psi_r[id_ym]*cos(-ay) - psi_i[id_ym]*sin(-ay);
    double iym = psi_r[id_ym]*sin(-ay) + psi_i[id_ym]*cos(-ay);
    double rzp = psi_r[id_zp]*cos(az) - psi_i[id_zp]*sin(az);
    double izp = psi_r[id_zp]*sin(az) + psi_i[id_zp]*cos(az);
    double rzm = psi_r[id_zm]*cos(-az) - psi_i[id_zm]*sin(-az);
    double izm = psi_r[id_zm]*sin(-az) + psi_i[id_zm]*cos(-az);

    double lap_r = -6.0*pr + rxp + rxm + ryp + rym + rzp + rzm;
    double lap_i = -6.0*pi + ixp + ixm + iyp + iym + izp + izm;

    // 4. INDUKCE POLE (Magnetická vazba)
    Ax[i] = ax + (Jx * LAMBDA_W * ALPHA_G) * dt;
    Ay[i] = ay + (Jy * LAMBDA_W * ALPHA_G) * dt;
    Az[i] = az + (Jz * LAMBDA_W * ALPHA_G) * dt;

    // 5. MASTER ROVNICE (Homeostáza protonu + Kondenzace elektronu)
    double awakening = 1.0 - exp(-J_mag * 30.0);
    double vacuum_intake = 0.2 * awakening * saturation;
    
    // Lokální sifon tlumí jen centra částic
    double local_friction = (current_m > (TARGET_RATIO * 0.7)) ? 0.1 : 0.0;

    double nr = pr + (0.05 * lap_r * isolation * dt) + (vacuum_intake * pr * dt) - (local_friction * pr * dt);
    double ni = pi + (0.05 * lap_i * isolation * dt) + (vacuum_intake * pi * dt) - (local_friction * pi * dt);

    double nm = sqrt(nr*nr + ni*ni);
    double u = (nm > (TARGET_RATIO * 1.5)) ? (TARGET_RATIO * 1.5 / nm) : 1.0;
    
    psi_rn[i] = nr * u; 
    psi_in[i] = ni * u;
    h_mass[i] = nm;
}
"""

class HydrogenSimulator:
    def __init__(self, size=64):
        self.N = size
        self.dt = 0.005
        platforms = cl.get_platforms()
        dev = platforms[0].get_devices()[0]
        self.ctx = cl.Context([dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_hydrogen_step")
        print(f"[+] OMNI-ENGINE v5.4: Vodíkový generátor spuštěn na {dev.name}")

        self.d_pr = cl_array.to_device(self.queue, np.full((self.N**3), 0.1, dtype=np.float64))
        self.d_pi = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        self.d_ax = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_ay = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_az = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.im = self.ax.imshow(np.zeros((self.N, self.N)), cmap='magma', origin='lower')
        self.ax.set_title("Projekt VODÍK: Protonové jádro a Elektronový mrak")

    def inject_atom(self):
        print("[*] Formuji atomové jádro (Proton) a periferní fázovou smyčku (Elektron)...")
        x, y, z = np.indices((self.N, self.N, self.N))
        dx, dy, dz = x - self.N//2, y - self.N//2, z - self.N//2
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Proton (Trefoil uzel)
        p_env = 10.0 * np.exp(-(r**2) / 10.0)
        p_phase = (dx/self.N)*np.pi*120 + (dy/self.N)*np.pi*120 + (dz/self.N)*np.pi*120
        
        # Elektron (Slabý periferní uzel - pravděpodobnostní mrak)
        e_env = 1.0 * np.exp(-((r - 15.0)**2) / 25.0) # Prstenec ve vzdálenosti 15 voxelů
        e_phase = np.arctan2(dy, dx) * 2.0
        
        pr = p_env * np.cos(p_phase) + e_env * np.cos(e_phase)
        pi = p_env * np.sin(p_phase) + e_env * np.sin(e_phase)
        
        self.d_pr = cl_array.to_device(self.queue, pr.flatten().astype(np.float64))
        self.d_pi = cl_array.to_device(self.queue, pi.flatten().astype(np.float64))

    def run(self, ticks=8000):
        self.inject_atom()
        for t in range(ticks):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_ax.data, self.d_ay.data, self.d_az.data, self.d_hm.data, 
                     np.float64(self.dt), np.int32(self.N))
            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi
            if t % 100 == 0:
                mass_3d = self.d_hm.get().reshape((self.N, self.N, self.N))
                self.im.set_data(mass_3d[self.N//2, :, :])
                self.im.set_clim(vmin=0, vmax=20) # Zvýraznění elektronového mraku
                self.ax.set_title(f"Atom Vodíku | Tik: {t} | Střed: {np.max(mass_3d):.2f}")
                plt.pause(0.01)

if __name__ == "__main__":
    sim = HydrogenSimulator(size=64)
    sim.run()
