import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import sys

# =====================================================================
# TKV PROJEKT 1836: ASYMPTOTICKÁ VOLNOST (OPRAVENO)
# Cíl: Skutečná asymptotická volnost tlumící runaway efekt při vysokém tlaku
# =====================================================================

kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void tkv_dynamic_step(
    __global const double *psi_r, __global const double *psi_i,
    __global double *psi_rn, __global double *psi_in,
    __global double *Ax, __global double *Ay, __global double *Az,
    __global double *h_total_energy,
    const double dt, const int N, const double time,
    const int physics_mode, const double vacuum_heat,
    const double coupling_max, const double critical_pressure) // Opravené parametry
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

    const double LAMBDA_W = 0.0833333333; // Základní EM síla (1/12)
    const double ALPHA_G = 1.19998161486;

    double pr = psi_r[i]; double pi = psi_i[i];
    double ax = Ax[i]; double ay = Ay[i]; double az = Az[i];

    // Centrální Sifon (udržuje uzel před zborcením do singularity)
    double dx = (double)x - (double)N/2.0;
    double dy = (double)y - (double)N/2.0;
    double dz = (double)z - (double)N/2.0;
    double r = sqrt(dx*dx + dy*dy + dz*dz);
    double siphon_friction = 0.5 * exp(-r / 1.5);

    double rxp, ixp, rxm, ixm, ryp, iyp, rym, iym, rzp, izp, rzm, izm;
    double Jx = 0.0; double Jy = 0.0; double Jz = 0.0;
    
    rzp = psi_r[id_zp]*cos(az) - psi_i[id_zp]*sin(az);
    izp = psi_r[id_zp]*sin(az) + psi_i[id_zp]*cos(az);
    rzm = psi_r[id_zm]*cos(-az) - psi_i[id_zm]*sin(-az);
    izm = psi_r[id_zm]*sin(-az) + psi_i[id_zm]*cos(-az);
    Jz = pr * (psi_i[id_zp] - psi_i[id_zm]) - pi * (psi_r[id_zp] - psi_r[id_zm]);

    if (physics_mode == 0) {
        rxp = psi_r[id_xp]; ixp = psi_i[id_xp]; rxm = psi_r[id_xm]; ixm = psi_i[id_xm];
        ryp = psi_r[id_yp]; iyp = psi_i[id_yp]; rym = psi_r[id_ym]; iym = psi_i[id_ym];
    } else {
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

    // =============================================================
    // SKUTEČNÁ ASYMPTOTICKÁ VOLNOST (TLUMENÍ POD TLAKEM)
    // =============================================================
    double local_density = pr*pr + pi*pi;
    double local_curvature = sqrt(lap_r*lap_r + lap_i*lap_i); // Tohle je tlak uvnitř uzlu
    
    // Asymptotická volnost: Jakmile tlak (curvature) začne stoupat směrem ke critical_pressure,
    // "lepidlo" se postupně oslabuje, dokud nespadne zpět na nulu (resp. LAMBDA_W).
    // Používáme klesající exponenciálu:
    double suppression_factor = exp(-(local_curvature * local_curvature) / (critical_pressure * critical_pressure));
    
    // Lepidlo (coupling_max) zabírá jen tam, kde je dost hmoty, ale tlak ještě není extrémní.
    double dynamic_lambda = LAMBDA_W + (coupling_max * local_density * suppression_factor);
    
    // Indukce pole
    Ax[i] = ax + (Jx * dynamic_lambda * ALPHA_G) * dt;
    Ay[i] = ay + (Jy * dynamic_lambda * ALPHA_G) * dt;
    Az[i] = az + (Jz * dynamic_lambda * ALPHA_G) * dt;

    double m_old = sqrt(pr*pr + pi*pi);
    double tenze = m_old * (1.0 - m_old*m_old);
    
    double jitter = sin((double)x*1.618 + (double)y*3.14 + (double)z*2.71 + time*15.0) * vacuum_heat;

    double nr = pr + (0.05 * lap_r * dt) + (tenze * pr * dt) - (siphon_friction * pr * dt) + jitter;
    double ni = pi + (0.05 * lap_i * dt) + (tenze * pi * dt) - (siphon_friction * pi * dt) + jitter;

    double nm = sqrt(nr*nr + ni*ni);
    double u = (nm > 3.5) ? (0.0) : (nm > 1.0 ? 1.0/nm : 1.0);
    
    psi_rn[i] = nr * u; psi_in[i] = ni * u;

    // HAMILTONIÁN
    double mass_energy = nm * nm;
    double kinetic_energy = 0.5 * local_curvature * local_curvature;
    double gluon_energy = dynamic_lambda * ALPHA_G * (Ax[i]*Ax[i] + Ay[i]*Ay[i] + Az[i]*Az[i]);

    h_total_energy[i] = mass_energy + kinetic_energy + gluon_energy;
}
"""

class TKV_Analyzer_1836:
    def __init__(self, size=64):
        self.N = size
        self.dt = 0.005
        self.global_time = 0.0
        
        platforms = cl.get_platforms()
        dev = platforms[0].get_devices()[0]
        self.ctx = cl.Context([dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "tkv_dynamic_step")
        
        print(f"[+] Projekt 1836 (Opraveno) online na: {dev.name}")

        self.d_pr = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_pi = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_pr_n = cl_array.empty_like(self.d_pr)
        self.d_pi_n = cl_array.empty_like(self.d_pi)
        
        self.d_ax = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_ay = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_az = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        self.d_total_energy = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        
        self.vacuum_energy_tara = 0.0
        self.electron_energy_ref = 3683.0

    def tare_vacuum(self, ticks=500):
        print("\n[*] FÁZE 0: Měření a renormalizace vakua...")
        d_pr_vac = cl_array.to_device(self.queue, np.full((self.N, self.N, self.N), 0.1, dtype=np.float64).flatten())
        d_pi_vac = cl_array.zeros(self.queue, self.N**3, dtype=np.float64)
        d_pr_vac_n, d_pi_vac_n = cl_array.empty_like(d_pr_vac), cl_array.empty_like(d_pi_vac)
        
        for t in range(ticks):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     d_pr_vac.data, d_pi_vac.data, d_pr_vac_n.data, d_pi_vac_n.data, 
                     self.d_ax.data, self.d_ay.data, self.d_az.data, self.d_total_energy.data, 
                     np.float64(self.dt), np.int32(self.N), np.float64(self.global_time),
                     np.int32(0), np.float64(0.0), np.float64(1.0), np.float64(10.0))
            d_pr_vac, d_pr_vac_n = d_pr_vac_n, d_pr_vac
            d_pi_vac, d_pi_vac_n = d_pi_vac_n, d_pi_vac
            
        self.vacuum_energy_tara = np.sum(self.d_total_energy.get()) / self.N
        print(f"[+] Tára uložena: {self.vacuum_energy_tara:.2f}")

    def measure_electron(self):
        print("\n[*] FÁZE 1: Měření referenčního Elektronu (slabé pole)...")
        self.d_ax.fill(0.0); self.d_ay.fill(0.0); self.d_az.fill(0.0); self.d_total_energy.fill(0.0)
        
        x, y, z = np.indices((self.N, self.N, self.N))
        dx, dy, dz = x - self.N/2.0, y - self.N/2.0, z - self.N/2.0
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        obalka = 0.1 + 0.9 * np.exp(-(r**2) / 8.0)
        phase = (z / self.N) * np.pi * 0.5 
        
        self.d_pr = cl_array.to_device(self.queue, (obalka * np.cos(phase)).flatten().astype(np.float64))
        self.d_pi = cl_array.to_device(self.queue, (obalka * np.sin(phase)).flatten().astype(np.float64))

        for t in range(2000):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_ax.data, self.d_ay.data, self.d_az.data, self.d_total_energy.data, 
                     np.float64(self.dt), np.int32(self.N), np.float64(self.global_time),
                     np.int32(1), np.float64(0.0), np.float64(1.0), np.float64(10.0))
            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi
            
        raw_e = np.sum(self.d_total_energy.get()) / self.N
        self.electron_energy_ref = raw_e - self.vacuum_energy_tara
        print(f"[+] Skutečná hmotnost elektronu (me): {self.electron_energy_ref:.2f}")

    def run_proton_fusion(self, coupling_max, critical_pressure):
        print(f"\n[*] FÁZE 2: Zrod Protonu (Max. Síla: {coupling_max}, Kritický tlak: {critical_pressure})")
        self.d_ax.fill(0.0); self.d_ay.fill(0.0); self.d_az.fill(0.0); self.d_total_energy.fill(0.0)
        
        x, y, z = np.indices((self.N, self.N, self.N))
        dx, dy, dz = x - self.N/2.0, y - self.N/2.0, z - self.N/2.0
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        obalka = 0.1 + 0.9 * np.exp(-(r**2) / 8.0)
        
        energy = 5.0
        phase_xy = np.arctan2(dy, dx) * energy 
        phase_xz = np.arctan2(dz, dx) * energy + (2.0/3.0)*np.pi
        phase_yz = np.arctan2(dz, dy) * energy + (4.0/3.0)*np.pi
        total_phase = phase_xy + phase_xz + phase_yz
        
        self.d_pr = cl_array.to_device(self.queue, (obalka * np.cos(total_phase)).flatten().astype(np.float64))
        self.d_pi = cl_array.to_device(self.queue, (obalka * np.sin(total_phase)).flatten().astype(np.float64))

        print("Fáze | Tik   | Hmotnost Protonu (Renormalizovaná) | Poměr mp/me")
        print("-" * 65)

        physics_mode = 0
        vacuum_heat = 0.0
        for t in range(1501):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_ax.data, self.d_ay.data, self.d_az.data, self.d_total_energy.data, 
                     np.float64(self.dt), np.int32(self.N), np.float64(self.global_time),
                     np.int32(physics_mode), np.float64(vacuum_heat), np.float64(coupling_max), np.float64(critical_pressure))
            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi
            self.global_time += self.dt
            
            if t > 0 and t % 500 == 0:
                self.queue.finish()
                raw_p = np.sum(self.d_total_energy.get()) / self.N
                renorm_p = raw_p - self.vacuum_energy_tara
                ratio = renorm_p / self.electron_energy_ref
                print(f"1D Fz| {t:04d}  | {renorm_p:34.2f} | {ratio:11.2f}")

        physics_mode = 1
        vacuum_heat = 0.015
        for t in range(4001):
            self.knl(self.queue, (self.N, self.N, self.N), None, 
                     self.d_pr.data, self.d_pi.data, self.d_pr_n.data, self.d_pi_n.data, 
                     self.d_ax.data, self.d_ay.data, self.d_az.data, self.d_total_energy.data, 
                     np.float64(self.dt), np.int32(self.N), np.float64(self.global_time),
                     np.int32(physics_mode), np.float64(vacuum_heat), np.float64(coupling_max), np.float64(critical_pressure))
            self.d_pr, self.d_pr_n = self.d_pr_n, self.d_pr
            self.d_pi, self.d_pi_n = self.d_pi_n, self.d_pi
            self.global_time += self.dt
            
            if t > 0 and t % 500 == 0:
                self.queue.finish()
                raw_p = np.sum(self.d_total_energy.get()) / self.N
                renorm_p = raw_p - self.vacuum_energy_tara
                ratio = renorm_p / self.electron_energy_ref
                print(f"3D Pě| {t+1500:04d}  | {renorm_p:34.2f} | {ratio:11.2f}")

        print("-" * 65)
        print(f"[*] FINÁLNÍ LEPTONOVÁ KONSTANTA PROTONU: {ratio:.2f}")


if __name__ == "__main__":
    analyzer = TKV_Analyzer_1836(size=64)
    analyzer.tare_vacuum()
    analyzer.measure_electron()
    
    # NOVÁ KALIBRACE
    # coupling_max: Udává, jak moc pole houstne v protonu
    # critical_pressure: Určuje tlak, při kterém pole praskne (asymptotická volnost)
    analyzer.run_proton_fusion(coupling_max=10.0, critical_pressure=3.0)
