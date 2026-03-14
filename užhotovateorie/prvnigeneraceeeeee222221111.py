import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import sys

# =====================================================================
# TKV OMNI-ENGINE v4.8: 3D MAGNETISMUS + SIPHON
# Cíl: Oprava zborcené symetrie, zrod toroidálního Mionu (E2)
# =====================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void siphon_hunter_3d(
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
    const double ALPHA_G = 1.19998161486;

    double pr = psi_r[i]; double pi = psi_i[i];
    double ax = Ax[i]; double ay = Ay[i]; double az = Az[i];

    // --- 1. SIPHON EFEKT ---
    double dx = (double)x - (double)N/2.0;
    double dy = (double)y - (double)N/2.0;
    double dz = (double)z - (double)N/2.0;
    double r = sqrt(dx*dx + dy*dy + dz*dz);
    double siphon_friction = 0.5 * exp(-r / 1.5);

    // --- 2. PLNÁ 3D PEIERLSOVA SUBSTITUCE ---
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

    // --- 3. PLNÝ 3D MAGNETISMUS A INDUKCE ---
    // Toto chybělo! Proud ve všech směrech stáčí částici do Torusu
    double Jx = pr * (psi_i[id_xp] - psi_i[id_xm]) - pi * (psi_r[id_xp] - psi_r[id_xm]);
    double Jy = pr * (psi_i[id_yp] - psi_i[id_ym]) - pi * (psi_r[id_yp] - psi_r[id_ym]);
    double Jz = pr * (psi_i[id_zp] - psi_i[id_zm]) - pi * (psi_r[id_zp] - psi_r[id_zm]);

    Ax[i] = ax + (Jx * LAMBDA_W * ALPHA_G) * dt;
    Ay[i] = ay + (Jy * LAMBDA_W * ALPHA_G) * dt;
    Az[i] = az + (Jz * LAMBDA_W * ALPHA_G) * dt;

    // --- 4. ZLOM SYMETRIE (Jitter) ---
    // Malinký fázový šum, který umožní mřížce "rozhodnout se", jak se uzel zatočí
    double jitter = sin((double)x*1.618 + (double)y*3.14 + (double)z*2.71 + time*10.0) * 0.0005;

    // --- 5. MASTER ROVNICE ---
    double m_old = sqrt(pr*pr + pi*pi);
    double tenze = m_old * (1.0 - m_old*m_old);
    
    double nr = pr + (0.05 * lap_r * dt) + (tenze * pr * dt) - (siphon_friction * pr * dt) + jitter;
    double ni = pi + (0.05 * lap_i * dt) + (tenze * pi * dt) - (siphon_friction * pi * dt) + jitter;

    double nm = sqrt(nr*nr + ni*ni);
    double u = (nm > 1.8) ? (0.0) : (nm > 1.0 ? 1.0/nm : 1.0);
    
    psi_rn[i] = nr * u; psi_in[i] = ni * u;
    h_mass[i] = nm;
}
"""

def run_simulation():
    N = 64
    platforms = cl.get_platforms()
    dev = platforms[0].get_devices()[0]
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kernel_code).build()
    knl = cl.Kernel(prg, "siphon_hunter_3d")

    E1_REF = 3683.0

    print(f"\n[*] Inicializován 3D Siphon-Stabilizer na: {dev.name}")
    print("--- LOV NA MION (E2) S PLNOU TOPOLOGIÍ ---")
    print("E-Tlak | Čistá Hmota | Poměr k E1 | Status")
    print("-" * 65)

    for energy in [0.5, 2.0, 10.0, 30.0, 80.0, 150.0]:
        print(f"{energy:6.1f} | ", end="")
        sys.stdout.flush()

        x, y, z = np.indices((N,N,N))
        phase = (z / N) * np.pi * energy 
        obalka = 0.1 + 0.9 * np.exp(-((x-N//2)**2 + (y-N//2)**2 + (z-N//2)**2) / 8.0)

        d_pr = cl_array.to_device(queue, (obalka * np.cos(phase)).flatten().astype(np.float64))
        d_pi = cl_array.to_device(queue, (obalka * np.sin(phase)).flatten().astype(np.float64))
        d_pr_n, d_pi_n = cl_array.empty_like(d_pr), cl_array.empty_like(d_pi)
        d_ax, d_ay, d_az = cl_array.zeros(queue, N**3, dtype=np.float64), cl_array.zeros(queue, N**3, dtype=np.float64), cl_array.zeros(queue, N**3, dtype=np.float64)
        d_hm = cl_array.zeros(queue, N**3, dtype=np.float64)

        safe_dt = 0.005 / (1.0 + energy * 0.1)
        
        for t in range(2500):
            knl(queue, (N,N,N), None, d_pr.data, d_pi.data, d_pr_n.data, d_pi_n.data, 
                d_ax.data, d_ay.data, d_az.data, d_hm.data, 
                np.float64(safe_dt), np.int32(N), np.float64(energy), np.float64(t*0.1))
            
            d_pr, d_pr_n = d_pr_n, d_pr
            d_pi, d_pi_n = d_pi_n, d_pi
            
            if t % 500 == 0: 
                print(".", end="")
                sys.stdout.flush()
                queue.finish()

        current_mass = np.sum(d_hm.get()) / N - (N**2 * 0.1)
        ratio = current_mass / E1_REF
        
        status = "Stabilní E1"
        if ratio > 5.0: status = "POTENCIÁLNÍ E2 (MION)!"
        if ratio > 50.0: status = "POTENCIÁLNÍ E3 (TAUON)!"
        if current_mass < 1.0: status = "SINGULARITA"

        print(f" {current_mass:10.2f} | {ratio:10.2f} | {status}")

    print("-" * 65)

if __name__ == "__main__":
    run_simulation()
