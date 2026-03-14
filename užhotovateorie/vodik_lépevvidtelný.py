import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter

# =====================================================================
# OMNI-ENGINE v5.4.3: PROJEKT VODÍK (ULTRA-JEMNÉ ROZLIŠENÍ & FIX)
# Oprava chyby 'QuadContourSet' a vylepšení vizuální hloubky
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
    const double TARGET_RATIO = 275.4258; 

    double pr = psi_r[i]; double pi = psi_i[i];
    double ax = Ax[i]; double ay = Ay[i]; double az = Az[i];

    // 1. TOPOLOGICKÝ PROUD (Vrtule)
    double Jx = pr * (psi_i[id_xp] - psi_i[id_xm]) - pi * (psi_r[id_xp] - psi_r[id_xm]);
    double Jy = pr * (psi_i[id_yp] - psi_i[id_ym]) - pi * (psi_r[id_yp] - psi_r[id_ym]);
    double Jz = pr * (psi_i[id_zp] - psi_i[id_zm]) - pi * (psi_r[id_zp] - psi_r[id_zm]);
    double J_mag = sqrt(Jx*Jx + Jy*Jy + Jz*Jz);

    // 2. GLUONOVÁ IZOLACE & HARMONIKA
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

    // 4. INDUKCE POLE
    Ax[i] = ax + (Jx * LAMBDA_W * ALPHA_G) * dt;
    Ay[i] = ay + (Jy * LAMBDA_W * ALPHA_G) * dt;
    Az[i] = az + (Jz * LAMBDA_W * ALPHA_G) * dt;

    // 5. MASTER ROVNICE S PLNOU LOGIKOU
    double awakening = 1.0 - exp(-J_mag * 30.0);
    double vacuum_intake = 0.2 * awakening * saturation;
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
        
        # OpenCL Setup
        platforms = cl.get_platforms()
        dev = platforms[0].get_devices()[0]
        self.ctx = cl.Context([dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_hydrogen_step")
        print(f"[+] OMNI-ENGINE v5.4.3: Vizualizační fix aplikován. Zařízení: {dev.name}")

        self.d_pr = cl_array.to_device(self.queue, np.full((self.N**3), 0.05, dtype=np.float64))
        self.d_pi = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        self.d_ax = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_ay = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_az = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_hm = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)

        # Vizualizační okno
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.patch.set_facecolor('#050505')
        
        # PowerNorm gamma 0.15 pro maximální "zjemnění" temných oblastí
        self.im = self.ax.imshow(np.zeros((self.N, self.N)), 
                                 cmap='magma', 
                                 origin='lower',
                                 norm=colors.PowerNorm(gamma=0.15, vmin=0.001, vmax=300),
                                 interpolation='bicubic') # Bicubic pro maximální hladkost
        
        self.cbar = self.fig.colorbar(self.im, label='Topologická energie (Jemné měřítko)')
        self.ax.set_title("Atom Vodíku: Detailní řez fázovým polem", color='white')
        self.ax.axis('off')
        
        self.contours = None

    def inject_atom(self):
        print("[*] Injektuji embryo vodíku s vysokou fázovou koherencí...")
        x, y, z = np.indices((self.N, self.N, self.N))
        dx, dy, dz = x - self.N//2, y - self.N//2, z - self.N//2
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        
        p_env = 15.0 * np.exp(-(r**2) / 7.0)
        p_phase = (dx/self.N)*np.pi*150 + (dy/self.N)*np.pi*150
        
        e_env = 1.5 * np.exp(-((r - 22.0)**2) / 35.0) 
        e_phase = np.arctan2(dy, dx) * 2.0
        
        pr = p_env * np.cos(p_phase) + e_env * np.cos(e_phase)
        pi = p_env * np.sin(p_phase) + e_env * np.sin(e_phase)
        
        self.d_pr = cl_array.to_device(self.queue, pr.flatten().astype(np.float64))
        self.d_pi = cl_array.to_device(self.queue, pi.flatten().astype(np.float64))

    def update_visuals(self, t):
        mass_3d = self.d_hm.get().reshape((self.N, self.N, self.N))
        slice_data = mass_3d[self.N//2, :, :]
        
        # Jemné vyhlazení Gaussovým filtrem pro odstranění voxelových hran
        slice_data = gaussian_filter(slice_data, sigma=0.6)
        slice_data = np.maximum(slice_data, 0.001)
        
        self.im.set_data(slice_data)
        
        # OPRAVA CHYBY: Nový způsob odstraňování kontur
        if self.contours:
            self.contours.remove()
        
        # Vykreslení jemných topologických vrstevnic
        # Úrovně: od ultra-slabého mraku (0.1) až po jádro
        levels = [0.2, 0.8, 2.5, 10, 50, 150, 250]
        self.contours = self.ax.contour(slice_data, levels=levels, colors='cyan', alpha=0.2, linewidths=0.4)
        
        max_val = np.max(slice_data)
        self.ax.set_title(f"Vodík | Tik: {t} | Hustota jádra: {max_val:.2f}", color='white', fontsize=12)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self, ticks=20000):
        self.inject_atom()
        for t in range(ticks):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_ax.data, self.d_ay.data, self.d_az.data, self.d_hm.data, 
                     np.float64(self.dt), np.int32(self.N))
            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi
            
            if t % 50 == 0:
                self.queue.finish()
                self.update_visuals(t)

if __name__ == "__main__":
    sim = HydrogenSimulator(size=64)
    sim.run()
