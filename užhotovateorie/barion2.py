import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import sys

# =====================================================================
# TKV PROTON COLLIDER (BARYON ENGINE v2.0)
# Cíl: Srážka 3 vírů (kvarků) v 1D trubici pro stvoření Protonu
# =====================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void proton_hunter(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *Az, // 1D Flux Tube (Gluonová struna)
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
    double az = Az[i]; 

    // --- 1. JÁDRO (Sifonový Chladič) ---
    double dx = (double)x - (double)N/2.0;
    double dy = (double)y - (double)N/2.0;
    double dz = (double)z - (double)N/2.0;
    double r = sqrt(dx*dx + dy*dy + dz*dz);
    double siphon_friction = 0.5 * exp(-r / 1.5);

    // --- 2. 1D PEIERLSOVA SUBSTITUCE (Z-osa) ---
    // Částice nemůže "uhnout" do X a Y, je chycena v silokřivce
    double rzp = psi_r[id_zp]*cos(az) - psi_i[id_zp]*sin(az);
    double izp = psi_r[id_zp]*sin(az) + psi_i[id_zp]*cos(az);
    double rzm = psi_r[id_zm]*cos(-az) - psi_i[id_zm]*sin(-az);
    double izm = psi_r[id_zm]*sin(-az) + psi_i[id_zm]*cos(-az);

    double lap_r = -6.0*pr + psi_r[id_xp] + psi_r[id_xm] + psi_r[id_yp] + psi_r[id_ym] + rzp + rzm;
    double lap_i = -6.0*pi + psi_i[id_xp] + psi_i[id_xm] + psi_i[id_yp] + psi_i[id_ym] + izp + izm;

    // --- 3. 1D INDUKCE ---
    double Jz = pr * (psi_i[id_zp] - psi_i[id_zm]) - pi * (psi_r[id_zp] - psi_r[id_zm]);
    Az[i] = az + (Jz * LAMBDA_W * ALPHA_G) * dt;

    // --- 4. MASTER ROVNICE BARYONU ---
    double m_old = sqrt(pr*pr + pi*pi);
    double tenze = m_old * (1.0 - m_old*m_old);
    
    // Tři víry vytvářejí obrovský zmatek, přidáme mírný termální jitter
    double jitter = sin((double)x*1.618 + (double)y*3.14 + (double)z*2.71 + time*10.0) * 0.001;

    double nr = pr + (0.05 * lap_r * dt) + (tenze * pr * dt) - (siphon_friction * pr * dt) + jitter;
    double ni = pi + (0.05 * lap_i * dt) + (tenze * pi * dt) - (siphon_friction * pi * dt) + jitter;

    double nm = sqrt(nr*nr + ni*ni);
    
    // Baryon v trubici vydrží obrovskou amplitudu (nastaveno na 3.5 před prasknutím)
    double u = (nm > 3.5) ? (0.0) : (nm > 1.0 ? 1.0/nm : 1.0);
    
    psi_rn[i] = nr * u; psi_in[i] = ni * u;
    h_mass[i] = nm;
}
"""

def run_collider():
    N = 64
    platforms = cl.get_platforms()
    dev = platforms[0].get_devices()[0]
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kernel_code).build()
    knl = cl.Kernel(prg, "proton_hunter")

    E1_REF = 3683.0 # Změřená hmotnost elektronu v našem vesmíru

    print(f"\n[*] PROTON COLLIDER ONLINE (GPU: {dev.name})")
    print("--- SRÁŽKA 3 VÍRŮ V GLUONOVÉ TRUBICI ---")
    print("E-Tlak | Čistá Hmota | Poměr k E1 (mp/me) | Status")
    print("-" * 75)

    # Obrovské tlaky. Proton je 1836x těžší než elektron.
    for energy in [10.0, 50.0, 100.0, 250.0, 500.0, 1000.0]:
        print(f"{energy:6.1f} | ", end="")
        sys.stdout.flush()

        x, y, z = np.indices((N,N,N))
        
        # --- SRÁŽKA TŘÍ KVARKŮ (VÍRŮ) ---
        # 1. Vítr Z (Základní fáze)
        phase_z = (z / N) * np.pi * energy 
        # 2. Vítr X (Posunutá fáze o 120 stupňů - simulace barevného náboje)
        phase_x = (x / N) * np.pi * energy + (2.0/3.0)*np.pi
        # 3. Vítr Y (Posunutá fáze o 240 stupňů)
        phase_y = (y / N) * np.pi * energy + (4.0/3.0)*np.pi
        
        # Sečteme všechny tři fáze do obrovské interference
        total_phase = phase_z + phase_x + phase_y
        
        obalka = 0.1 + 0.9 * np.exp(-((x-N//2)**2 + (y-N//2)**2 + (z-N//2)**2) / 8.0)

        # Inicializace interference do mřížky
        d_pr = cl_array.to_device(queue, (obalka * np.cos(total_phase)).flatten().astype(np.float64))
        d_pi = cl_array.to_device(queue, (obalka * np.sin(total_phase)).flatten().astype(np.float64))
        d_pr_n, d_pi_n = cl_array.empty_like(d_pr), cl_array.empty_like(d_pi)
        
        d_az = cl_array.zeros(queue, N**3, dtype=np.float64)
        d_hm = cl_array.zeros(queue, N**3, dtype=np.float64)

        # Časový krok musí být extrémně malý kvůli gigantickým tlakům
        safe_dt = 0.005 / (1.0 + energy * 0.02)
        
        for t in range(2500):
            knl(queue, (N,N,N), None, d_pr.data, d_pi.data, d_pr_n.data, d_pi_n.data, 
                d_az.data, d_hm.data, 
                np.float64(safe_dt), np.int32(N), np.float64(energy), np.float64(t*0.1))
            
            d_pr, d_pr_n = d_pr_n, d_pr
            d_pi, d_pi_n = d_pi_n, d_pi
            
            if t % 500 == 0: 
                print(".", end="")
                sys.stdout.flush()
                queue.finish()

        current_mass = np.sum(d_hm.get()) / N - (N**2 * 0.1)
        ratio = current_mass / E1_REF
        
        # Analýza Baryonu
        status = "Stabilní uzel"
        if ratio > 50.0: status = "Lehký Hadron (Pion?)"
        if ratio > 500.0: status = "STŘEDNÍ BARYON!"
        if ratio > 1500.0 and ratio < 2000.0: status = "*** PROTON NALEZEN! ***"
        if ratio > 2000.0: status = "TĚŽKÁ HYPERONOVÁ REZONANCE"
        if current_mass < 1.0: status = "KOLAPS TRUBICE (Singularita)"

        print(f" {current_mass:12.2f} | {ratio:18.2f} | {status}")

    print("-" * 75)

if __name__ == "__main__":
    run_collider()
