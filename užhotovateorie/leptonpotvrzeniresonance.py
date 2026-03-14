import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import csv
import sys

# =====================================================================
# TKV OMNI-ENGINE v4.2 - VERTIKÁLNÍ SKENER S TEPEm (HEARTBEAT)
# =====================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void omni_magnetic_spectrum(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *Ax, __global double *Ay, __global double *Az,
    __global double *h_mass,
    const double dt, const int N, const double energy_input, const double time)
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

    double pr = psi_r[i]; double pi = psi_i[i];
    double m_old = sqrt(pr*pr + pi*pi);
    double ax = Ax[i]; double ay = Ay[i]; double az = Az[i];

    // Magnetická rotace
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

    // Indukce
    double Jx = pr * (psi_i[id_xp] - psi_i[id_xm]) - pi * (psi_r[id_xp] - psi_r[id_xm]);
    double Jy = pr * (psi_i[id_yp] - psi_i[id_ym]) - pi * (psi_r[id_yp] - psi_r[id_ym]);
    double Jz = pr * (psi_i[id_zp] - psi_i[id_zm]) - pi * (psi_r[id_zp] - psi_r[id_zm]);

    Ax[i] = ax + (Jx * LAMBDA_W) * dt;
    Ay[i] = ay + (Jy * LAMBDA_W) * dt;
    Az[i] = az + (Jz * LAMBDA_W) * dt;

    double tenze = m_old * (1.0 - m_old*m_old);
    double nr = pr + (0.05 * lap_r * dt) + (tenze * pr * dt);
    double ni = pi + (0.05 * lap_i * dt) + (tenze * pi * dt);

    double nm = sqrt(nr*nr + ni*ni);
    double u = (nm > 1.0) ? (1.0/nm) : 1.0;
    psi_rn[i] = nr * u; psi_in[i] = ni * u;
    h_mass[i] = nm;
}
"""

class TKV_Scanner_v42:
    def __init__(self, size=64): # Zmenšeno na 64 pro stabilitu, uvidíme to i tak
        self.size = size
        platforms = cl.get_platforms()
        dev = platforms[0].get_devices()[0]
        self.ctx = cl.Context([dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "omni_magnetic_spectrum")

    def scan_energy(self, energy):
        N = self.size
        x, y, z = np.indices((N,N,N))
        phase = (z / N) * np.pi * energy 
        dist = np.sqrt((x-N//2)**2 + (y-N//2)**2 + (z-N//2)**2)
        obalka = 0.1 + 0.9 * np.exp(-(dist**2) / 8.0)
        
        d_pr = cl_array.to_device(self.queue, (obalka * np.cos(phase)).flatten().astype(np.float64))
        d_pi = cl_array.to_device(self.queue, (obalka * np.sin(phase)).flatten().astype(np.float64))
        d_pr_n, d_pi_n = cl_array.empty_like(d_pr), cl_array.empty_like(d_pi)
        d_ax, d_ay, d_az = cl_array.zeros(self.queue, N**3, dtype=np.float64), cl_array.zeros(self.queue, N**3, dtype=np.float64), cl_array.zeros(self.queue, N**3, dtype=np.float64)
        d_hm = cl_array.zeros(self.queue, N**3, dtype=np.float64)

        print(f"[*] Testuji energii {energy:.2f}: ", end="")
        sys.stdout.flush()

        for t in range(1500): # Rychlejší test
            self.knl(self.queue, (N,N,N), None, d_pr.data, d_pi.data, d_pr_n.data, d_pi_n.data, 
                     d_ax.data, d_ay.data, d_az.data, d_hm.data, 
                     np.float64(0.005), np.int32(N), np.float64(energy), np.float64(t*0.1))
            d_pr, d_pr_n = d_pr_n, d_pr
            d_pi, d_pi_n = d_pi_n, d_pi
            
            if t % 300 == 0:
                print(".", end="") # Srdeční tep
                sys.stdout.flush()
                self.queue.finish()

        m = np.sum(d_hm.get()) / N - (N**2 * 0.1)
        print(f" Hotovo. Hmota: {m:.2f}")
        return m

scanner = TKV_Scanner_v42()
for e in np.arange(0.2, 4.0, 0.4):
    scanner.scan_energy(e)
