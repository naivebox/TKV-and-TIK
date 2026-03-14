import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import time

# =====================================================================
# OMNI-ENGINE v5.1: PROJEKT HARMONIKA - HADRON AWAKENING
# Oprava detekce koherence pro 3-fázové systémy (Kvarky)
# =====================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void tkv_harmonika_step(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *Ax, __global double *Ay, __global double *Az,
    __global double *h_mass,
    const double dt, const int N, const double time)
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
    const double PI = 3.14159265359;
    const double TARGET_RATIO = 183.615; 

    double pr = psi_r[i]; double pi = psi_i[i];
    double ax = Ax[i]; double ay = Ay[i]; double az = Az[i];

    double dx = (double)x - (double)N/2.0;
    double dy = (double)y - (double)N/2.0;
    double dz = (double)z - (double)N/2.0;

    // 1. KOVARIANTNÍ LAPLACIÁN A PROUDY
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

    double Jx = pr * (psi_i[id_xp] - psi_i[id_xm]) - pi * (psi_r[id_xp] - psi_r[id_xm]);
    double Jy = pr * (psi_i[id_yp] - psi_i[id_ym]) - pi * (psi_r[id_yp] - psi_r[id_ym]);
    double Jz = pr * (psi_i[id_zp] - psi_i[id_zm]) - pi * (psi_r[id_zp] - psi_r[id_zm]);
    double J_mag = sqrt(Jx*Jx + Jy*Jy + Jz*Jz);

    // 2. DETEKCE PROBUZENÍ PŘES TOPOLOGICKÝ STRES (Fix pro 3-kvarky)
    // Už nehledáme fázový součet (ten je 0), ale hledáme "vrtuli" (J_mag).
    double awakening = 1.0 - exp(-J_mag * 20.0); 

    // 3. NELINEÁRNÍ SATURAČNÍ ZÁMEK
    double current_m = sqrt(pr*pr + pi*pi);
    double saturation = 0.5 * (1.0 - tanh(5.0 * (current_m / TARGET_RATIO - 0.98)));
    if (saturation < 0.0) saturation = 0.0;

    // 4. INDUKCE
    double lambda_dyn = LAMBDA_W * (1.0 + current_m * 0.2);
    Ax[i] = ax + (Jx * lambda_dyn * ALPHA_G) * dt;
    Ay[i] = ay + (Jy * lambda_dyn * ALPHA_G) * dt;
    Az[i] = az + (Jz * lambda_dyn * ALPHA_G) * dt;

    // 5. MASTER ROVNICE S POSÍLENÝM SÁNÍM
    // Zvýšena konstanta z 0.015 na 0.1, aby proton "nechcípl" v zárodku.
    double vacuum_intake = 0.1 * awakening * saturation;
    
    double dist = sqrt(dx*dx + dy*dy + dz*dz);
    double friction = 0.4 * exp(-dist / 2.0);

    double nr = pr + (0.05 * lap_r * dt) + (vacuum_intake * pr * dt) - (friction * pr * dt);
    double ni = pi + (0.05 * lap_i * dt) + (vacuum_intake * pi * dt) - (friction * pi * dt);

    double nm = sqrt(nr*nr + ni*ni);
    double u = (nm > (TARGET_RATIO * 1.5)) ? (TARGET_RATIO * 1.5 / nm) : 1.0;
    
    psi_rn[i] = nr * u; 
    psi_in[i] = ni * u;
    h_mass[i] = nm;
}
"""

class TKV_Harmonika:
    def __init__(self, size=64):
        self.N = size
        self.dt = 0.005
        self.global_time = 0.0
        platforms = cl.get_platforms()
        dev = platforms[0].get_devices()[0]
        self.ctx = cl.Context([dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_harmonika_step")
        print(f"[+] OMNI-ENGINE v5.1 online. Ellesmere zvyšuje tah sání.")

        bg = np.full((self.N, self.N, self.N), 0.1, dtype=np.float64)
        empty = np.zeros((self.N, self.N, self.N), dtype=np.float64)
        self.d_pr = cl_array.to_device(self.queue, bg.flatten())
        self.d_pi = cl_array.to_device(self.queue, empty.flatten())
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        self.d_ax = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_ay = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_az = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

    def inject_hadron_seed(self):
        print("[*] Vstřikuji vysokoenergetické kvarkové embryo...")
        x, y, z = np.indices((self.N, self.N, self.N))
        dx, dy, dz = x - self.N//2, y - self.N//2, z - self.N//2
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        p1 = (dx/self.N) * np.pi * 100.0 # Vyšší energie pro start
        p2 = (dy/self.N) * np.pi * 100.0 + (2*np.pi/3)
        p3 = (dz/self.N) * np.pi * 100.0 + (4*np.pi/3)
        envelope = 0.5 + 2.5 * np.exp(-(r**2) / 12.0)
        pr = envelope * (np.cos(p1) + np.cos(p2) + np.cos(p3))
        pi = envelope * (np.sin(p1) + np.sin(p2) + np.sin(p3))
        self.d_pr = cl_array.to_device(self.queue, pr.flatten().astype(np.float64))
        self.d_pi = cl_array.to_device(self.queue, pi.flatten().astype(np.float64))

    def run(self, ticks):
        for t in range(ticks):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_ax.data, self.d_ay.data, self.d_az.data, self.d_hm.data, 
                     np.float64(self.dt), np.int32(self.N), np.float64(self.global_time))
            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi
            self.global_time += self.dt
            if t % 500 == 0:
                self.queue.finish()
                max_m = np.max(self.d_hm.get())
                print(f"    Tik {t:05d} | Lokální hmotnost: {max_m:.4f} / 183.6")

if __name__ == "__main__":
    sim = TKV_Harmonika(size=64)
    sim.inject_hadron_seed()
    sim.run(ticks=20000)
