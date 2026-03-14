import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import sys

# =====================================================================
# TKV RENORMALIZOVANÝ MOTOR + EXPORT 3D DAT
# =====================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void tkv_energy_evaluator(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *Ax, __global double *Ay, __global double *Az,
    __global double *h_total_energy,
    const double dt, const int N, const double energy_input, const double time,
    const int is_1d_tube) 
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
    
    rzp = psi_r[id_zp]*cos(az) - psi_i[id_zp]*sin(az);
    izp = psi_r[id_zp]*sin(az) + psi_i[id_zp]*cos(az);
    rzm = psi_r[id_zm]*cos(-az) - psi_i[id_zm]*sin(-az);
    izm = psi_r[id_zm]*sin(-az) + psi_i[id_zm]*cos(-az);

    if (is_1d_tube == 0) {
        rxp = psi_r[id_xp]*cos(ax) - psi_i[id_xp]*sin(ax);
        ixp = psi_r[id_xp]*sin(ax) + psi_i[id_xp]*cos(ax);
        rxm = psi_r[id_xm]*cos(-ax) - psi_i[id_xm]*sin(-ax);
        ixm = psi_r[id_xm]*sin(-ax) + psi_i[id_xm]*cos(-ax);
        ryp = psi_r[id_yp]*cos(ay) - psi_i[id_yp]*sin(ay);
        iyp = psi_r[id_yp]*sin(ay) + psi_i[id_yp]*cos(ay);
        rym = psi_r[id_ym]*cos(-ay) - psi_i[id_ym]*sin(-ay);
        iym = psi_r[id_ym]*sin(-ay) + psi_i[id_ym]*cos(-ay);
    } else {
        rxp = psi_r[id_xp]; ixp = psi_i[id_xp];
        rxm = psi_r[id_xm]; ixm = psi_i[id_xm];
        ryp = psi_r[id_yp]; iyp = psi_i[id_yp];
        rym = psi_r[id_ym]; iym = psi_i[id_ym];
    }

    double lap_r = -6.0*pr + rxp + rxm + ryp + rym + rzp + rzm;
    double lap_i = -6.0*pi + ixp + ixm + iyp + iym + izp + izm;

    double Jz = pr * (psi_i[id_zp] - psi_i[id_zm]) - pi * (psi_r[id_zp] - psi_r[id_zm]);
    Az[i] = az + (Jz * LAMBDA_W * ALPHA_G) * dt;
    
    if (is_1d_tube == 0) {
        double Jx = pr * (psi_i[id_xp] - psi_i[id_xm]) - pi * (psi_r[id_xp] - psi_r[id_xm]);
        double Jy = pr * (psi_i[id_yp] - psi_i[id_ym]) - pi * (psi_r[id_yp] - psi_r[id_ym]);
        Ax[i] = ax + (Jx * LAMBDA_W * ALPHA_G) * dt;
        Ay[i] = ay + (Jy * LAMBDA_W * ALPHA_G) * dt;
    }

    double m_old = sqrt(pr*pr + pi*pi);
    double tenze = m_old * (1.0 - m_old*m_old);
    
    double jitter = sin((double)x*1.618 + (double)y*3.14 + (double)z*2.71 + time*10.0) * 0.001;
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

def run_renormalization_and_export():
    N = 64
    platforms = cl.get_platforms()
    dev = platforms[0].get_devices()[0]
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kernel_code).build()
    knl = cl.Kernel(prg, "tkv_energy_evaluator")
    
    safe_dt = 0.005
    ratio = 0.0 # Pojistka pro proměnnou

    print(f"\n[*] TKV RENORMALIZÁTOR ONLINE (GPU: {dev.name})")

    # FÁZE 0
    print("--- FÁZE 0: Tárování vakua ---")
    x, y, z = np.indices((N,N,N))
    d_pr = cl_array.to_device(queue, np.full((N,N,N), 0.1, dtype=np.float64).flatten())
    d_pi = cl_array.zeros(queue, N**3, dtype=np.float64)
    d_pr_n, d_pi_n = cl_array.empty_like(d_pr), cl_array.empty_like(d_pi)
    d_ax, d_ay, d_az = cl_array.zeros(queue, N**3, dtype=np.float64), cl_array.zeros(queue, N**3, dtype=np.float64), cl_array.zeros(queue, N**3, dtype=np.float64)
    d_total_energy = cl_array.zeros(queue, N**3, dtype=np.float64)

    for t in range(500):
        knl(queue, (N,N,N), None, d_pr.data, d_pi.data, d_pr_n.data, d_pi_n.data, 
            d_ax.data, d_ay.data, d_az.data, d_total_energy.data, 
            np.float64(safe_dt), np.int32(N), np.float64(0.0), np.float64(t*0.1), np.int32(0))
        d_pr, d_pr_n = d_pr_n, d_pr; d_pi, d_pi_n = d_pi_n, d_pi

    vacuum_energy = np.sum(d_total_energy.get()) / N

    # FÁZE 1
    print("--- FÁZE 1: Měření Elektronu ---")
    energy = 0.5
    phase = (z / N) * np.pi * energy 
    obalka = 0.1 + 0.9 * np.exp(-((x-N//2)**2 + (y-N//2)**2 + (z-N//2)**2) / 8.0)
    
    d_pr = cl_array.to_device(queue, (obalka * np.cos(phase)).flatten().astype(np.float64))
    d_pi = cl_array.to_device(queue, (obalka * np.sin(phase)).flatten().astype(np.float64))
    d_ax.fill(0.0); d_ay.fill(0.0); d_az.fill(0.0); d_total_energy.fill(0.0)

    for t in range(2000):
        knl(queue, (N,N,N), None, d_pr.data, d_pi.data, d_pr_n.data, d_pi_n.data, 
            d_ax.data, d_ay.data, d_az.data, d_total_energy.data, 
            np.float64(safe_dt), np.int32(N), np.float64(energy), np.float64(t*0.1), np.int32(0))
        d_pr, d_pr_n = d_pr_n, d_pr; d_pi, d_pi_n = d_pi_n, d_pi

    renormalized_electron = (np.sum(d_total_energy.get()) / N) - vacuum_energy

    # FÁZE 2
    print("--- FÁZE 2: Vznět Protonu ---")
    energy_baryon = 1.0 
    total_phase = (z / N) * np.pi * energy_baryon + ((x / N) * np.pi * energy_baryon + (2.0/3.0)*np.pi) + ((y / N) * np.pi * energy_baryon + (4.0/3.0)*np.pi)
    
    d_pr = cl_array.to_device(queue, (obalka * np.cos(total_phase)).flatten().astype(np.float64))
    d_pi = cl_array.to_device(queue, (obalka * np.sin(total_phase)).flatten().astype(np.float64))
    d_ax.fill(0.0); d_ay.fill(0.0); d_az.fill(0.0); d_total_energy.fill(0.0)

    for t in range(4001):
        knl(queue, (N,N,N), None, d_pr.data, d_pi.data, d_pr_n.data, d_pi_n.data, 
            d_ax.data, d_ay.data, d_az.data, d_total_energy.data, 
            np.float64(safe_dt), np.int32(N), np.float64(energy_baryon), np.float64(t*0.1), np.int32(1))
        d_pr, d_pr_n = d_pr_n, d_pr; d_pi, d_pi_n = d_pi_n, d_pi
        
        if t % 1000 == 0: 
            renormalized_proton = (np.sum(d_total_energy.get()) / N) - vacuum_energy
            ratio = renormalized_proton / renormalized_electron if renormalized_electron > 0 else 0
            print(f"Tik {t:04d} | Poměr mp/me = {ratio:.2f}")
            queue.finish()

    print("-" * 55)
    print(f"[*] Finální Leptonová Konstanta (mp/me) = {ratio:.2f}")

    # ==========================================
    # FÁZE 3: EXPORT DAT
    # ==========================================
    print("\n--- FÁZE 3: Ukládání 3D struktury ---")
    psi_r_np = d_pr.get().reshape((N, N, N))
    psi_i_np = d_pi.get().reshape((N, N, N))
    amplitude_np = np.sqrt(psi_r_np**2 + psi_i_np**2)
    az_np = d_az.get().reshape((N, N, N))
    gluon_energy_np = (az_np**2)

    np.save("proton_quarks_amplitude.npy", amplitude_np)
    np.save("proton_gluon_tube.npy", gluon_energy_np)
    print("[+] Soubory 'proton_quarks_amplitude.npy' a 'proton_gluon_tube.npy' uloženy!")

if __name__ == "__main__":
    run_renormalization_and_export()
