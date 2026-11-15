import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import time
import os

print("=== QuasiCluster: Python Multithreading Test ===")
# Verify that the script respects the threads allocated by Slurm
threads = os.environ.get('OMP_NUM_THREADS', 'Not Set')
print(f"Allocated Threads (OMP_NUM_THREADS): {threads}")
start_time = time.time()

# ==============================================================================
# GAUSSIAN PULSE VARIATIONS (20 VARIATIONS IN INCREASING TIME ORDER)
# ==============================================================================

# ----------------------------- physical constants ---------------------------
hbar = 6.582119569e-4          # eV * ps

# ------------------------- device D5 parameters -----------------------------
Delta1, Delta2, DeltaT, Deltac = 1.80, 1.98, 1.20, 1.82   # eV (CT removed)
g_bare = 500e-9                # eV
N      = 5.0e10                
G      = g_bare * np.sqrt(N)   # collective coupling (eV)

kappa, gamma_m, gamma_T = 29.0, 0.5, 0.01     # ps^-1
gamma_z, gamma_ISC      = 20.0, 5.0           # ps^-1

# Pulse parameters
r          = 0.5
t0         = 0.10                                       # ps (minimum peak center)
sigma_orig = (35.0 / (2*np.sqrt(2*np.log(2)))) * 1e-3   # ~0.01486 ps (FWHM = 35.0 fs)
sigma_narrow    = sigma_orig * 0.5
sigma_medium    = sigma_orig * 1.75
sigma_wide      = sigma_orig * 2.5
sigma_very_wide = sigma_orig * 3.5

# Reference total pulse energy (photon fluence): I0 = \int_{-\infty}^{\infty} |\eta_{orig}(t)|^2 dt
# Karena \eta_{orig}(t) = (\sqrt{r} / (\sigma_{orig}\sqrt{2\pi})) * exp(-(t-t0)^2 / (2\sigma_{orig}^2)),
# maka I0 = r / (2 * \sqrt{\pi} * \sigma_{orig})
I0 = r / (2.0 * np.sqrt(np.pi) * sigma_orig)            # ~9.4934 ps^-1

# 20 Gaussian pulse width variations (5 existing + 15 wider increasing by +0.5*sigma_orig)
gaussian_multipliers = [
    0.50,   # 1. Narrow Gaussian    (FWHM = 17.5 fs)  [Original Narrow]
    1.00,   # 2. Original Baseline  (FWHM = 35.0 fs)  [Original Baseline]
    1.75,   # 3. Medium Gaussian    (FWHM = 61.2 fs)
    2.50,   # 4. Wide Gaussian      (FWHM = 87.5 fs)  [Original Wide]
    3.50,   # 5. Very Wide Gaussian (FWHM = 122.5 fs)
    4.00,   # 6. Gaussian 4.0x      (FWHM = 140.0 fs)
    4.50,   # 7. Gaussian 4.5x      (FWHM = 157.5 fs)
    5.00,   # 8. Gaussian 5.0x      (FWHM = 175.0 fs)
    5.50,   # 9. Gaussian 5.5x      (FWHM = 192.5 fs)
    6.00,   # 10. Gaussian 6.0x     (FWHM = 210.0 fs)
    6.50,   # 11. Gaussian 6.5x     (FWHM = 227.5 fs)
    7.00,   # 12. Gaussian 7.0x     (FWHM = 245.0 fs)
    7.50,   # 13. Gaussian 7.5x     (FWHM = 262.5 fs)
    8.00,   # 14. Gaussian 8.0x     (FWHM = 280.0 fs)
    8.50,   # 15. Gaussian 8.5x     (FWHM = 297.5 fs)
    9.00,   # 16. Gaussian 9.0x     (FWHM = 315.0 fs)
    9.50,   # 17. Gaussian 9.5x     (FWHM = 332.5 fs)
    10.00,  # 18. Gaussian 10.0x    (FWHM = 350.0 fs)
    10.50,  # 19. Gaussian 10.5x    (FWHM = 367.5 fs)
    11.00,  # 20. Gaussian 11.0x    (FWHM = 385.0 fs)
]

M_pol = np.array([[Deltac, G,      G     ],
                  [G,      Delta1, 0.0   ],
                  [G,      0.0,    Delta2]])
nu = np.linalg.eigvalsh(M_pol)[0]

d1, d2, dc = (Delta1 - nu)/hbar, (Delta2 - nu)/hbar, (Deltac - nu)/hbar
Gw = G / hbar

def build_model():
    """Operators, Hamiltonian, collapse and measurement operators (without CT)."""
    Na, Nb = 5, 3              
    Ia, Ib, I2 = qt.qeye(Na), qt.qeye(Nb), qt.qeye(2)

    def emb(op, slot):
        ops = [Ia, Ib, Ib, I2, I2, I2]
        ops[slot] = op
        return qt.tensor(*ops)

    a   = emb(qt.destroy(Na), 0)            
    b1  = emb(qt.destroy(Nb), 1)            
    b2  = emb(qt.destroy(Nb), 2)            
    dk1 = emb(qt.destroy(2), 3)            
    dk2 = emb(qt.destroy(2), 4)            
    Tm  = emb(qt.destroy(2), 5)            
    
    H0 = (dc * a.dag()*a + d1 * b1.dag()*b1 + d2 * b2.dag()*b2
          + Gw * (a.dag()*(b1 + b2) + (b1.dag() + b2.dag())*a))
    Hdrive = 1j * (a.dag() - a)            

    c_ops = [np.sqrt(kappa)     * a,
             np.sqrt(gamma_m)   * b1,               
             np.sqrt(gamma_m)   * b2,               
             np.sqrt(gamma_m)   * dk1,              
             np.sqrt(gamma_m)   * dk2,              
             np.sqrt(gamma_T)   * Tm,               
             np.sqrt(gamma_z)   * dk1.dag()*b1,     
             np.sqrt(gamma_z)   * dk2.dag()*b2,
             np.sqrt(gamma_ISC) * Tm.dag()*b1,      
             np.sqrt(gamma_ISC) * Tm.dag()*b2,
             np.sqrt(gamma_ISC) * Tm.dag()*dk1,     
             np.sqrt(gamma_ISC) * Tm.dag()*dk2]             

    e_ops = [b1.dag()*b1 + dk1.dag()*dk1,           # <X11>  (S1^0)
             b2.dag()*b2 + dk2.dag()*dk2,           # <X22>  (S1^1)
             Tm.dag()*Tm,                           # <XTT>  (T1)
             a.dag()*a]                             # photons per molecule

    H_batt = (Delta1 * (b1.dag()*b1 + dk1.dag()*dk1)
              + Delta2 * (b2.dag()*b2 + dk2.dag()*dk2)
              + DeltaT * Tm.dag()*Tm)

    # Molecular battery Hamiltonian (tracing out cavity slot 0) for ergotropy calculation
    b1_mol  = qt.tensor(qt.destroy(Nb), Ib, I2, I2, I2)
    b2_mol  = qt.tensor(Ib, qt.destroy(Nb), I2, I2, I2)
    dk1_mol = qt.tensor(Ib, Ib, qt.destroy(2), I2, I2)
    dk2_mol = qt.tensor(Ib, Ib, I2, qt.destroy(2), I2)
    Tm_mol  = qt.tensor(Ib, Ib, I2, I2, qt.destroy(2))
    H_batt_mol = (Delta1 * (b1_mol.dag()*b1_mol + dk1_mol.dag()*dk1_mol)
                  + Delta2 * (b2_mol.dag()*b2_mol + dk2_mol.dag()*dk2_mol)
                  + DeltaT * Tm_mol.dag()*Tm_mol)

    psi0 = qt.tensor(qt.basis(Na, 0), qt.basis(Nb, 0), qt.basis(Nb, 0),
                     qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0))
    return H0, Hdrive, c_ops, e_ops, psi0, H_batt, H_batt_mol


def compute_thermodynamics(res, tlist, eps_sorted):
    """Compute stored energy E(t), charging power P(t), and ergotropy E_ext(t)."""
    P_S10 = np.real(res.expect[0])
    P_S11 = np.real(res.expect[1])
    P_T1  = np.real(res.expect[2])
    
    # Stored energy E(t) = <H_batt> (in eV)
    E_stored = Delta1 * P_S10 + Delta2 * P_S11 + DeltaT * P_T1
    # Instantaneous charging power P(t) = dE(t)/dt (in eV/ps)
    P_charge = np.gradient(E_stored, tlist)
    
    # Ergotropy (extractable work) of the molecular battery subsystem:
    # W_ext(t) = Tr[rho_mol(t) H_batt_mol] - sum_k r_k(t) * eps_k
    ergotropy = np.zeros(len(tlist))
    for i, state in enumerate(res.states):
        rho_mol = state.ptrace([1, 2, 3, 4, 5])
        r_vals = np.sort(np.real(rho_mol.eigenenergies()))[::-1]
        ergotropy[i] = max(0.0, E_stored[i] - np.sum(r_vals * eps_sorted))
        
    return E_stored, P_charge, ergotropy


def make_eta_gaussian(sigma_val):
    """
    Return a Gaussian drive function normalized to total pulse energy I0.
    Uses adaptive peak center t0_val = max(t0, 4.0 * sigma_val) so wide Gaussian
    pulses (> 3.5x) are not truncated at t < 0 and retain 100% of pulse energy I0.
    """
    prefactor = np.sqrt(I0 / (sigma_val * np.sqrt(np.pi)))
    t0_val = max(t0, 4.0 * sigma_val)
    def eta_fn(t, **kwargs):
        return prefactor * np.exp(-(t - t0_val)**2 / (2.0 * sigma_val**2))
    return eta_fn, t0_val


H0, Hdrive, c_ops, e_ops, psi0, H_batt, H_batt_mol = build_model()
eps_batt_sorted = np.sort(np.real(H_batt_mol.eigenenergies()))

tlist = np.linspace(0.0, 4.0, 801)     # ps
# max_step wajib: tanpa ini solver adaptif bisa "melompati" pulsa sempit (sigma ~7 fs)
opts = {"atol": 1e-9, "rtol": 1e-7, "nsteps": 1000000, "store_states": True,
        "max_step": sigma_narrow / 5}   # ~1.5 fs

fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0)) * 1e3
# Preserve the 5 original distinct colors for the first 5 variations
base_colors_g = ["#2ca02c", "#d62728", "#ff7f0e", "#1f77b4", "#9467bd"]
colors_g = base_colors_g + [plt.cm.viridis(x) for x in np.linspace(0.15, 0.95, 15)]

gaussian_names_base = [
    "1. Narrow Gaussian (0.5x)",
    "2. Original Baseline (1.0x)",
    "3. Medium Gaussian (1.75x)",
    "4. Wide Gaussian (2.5x)",
    "5. Very Wide Gaussian (3.5x)",
] + [f"{i}. Gaussian ({m:.1f}x)" for i, m in enumerate(gaussian_multipliers[5:], start=6)]

gaussian_sigmas = [m * sigma_orig for m in gaussian_multipliers]
gaussian_pairs  = [make_eta_gaussian(s) for s in gaussian_sigmas]
gaussian_etas   = [pair[0] for pair in gaussian_pairs]
gaussian_t0s    = [pair[1] for pair in gaussian_pairs]
gaussian_results = []
gaussian_thermos = []

for idx, (title, m, s, eta_fn, t0_v) in enumerate(zip(gaussian_names_base, gaussian_multipliers, gaussian_sigmas, gaussian_etas, gaussian_t0s), start=1):
    fwhm_fs = s * fwhm_factor
    print(f"Running {idx}/20: {title} [FWHM = {fwhm_fs:.1f} fs, sigma = {s*1e3:.1f} fs, t0 = {t0_v:.3f} ps]...")
    res = qt.mesolve([H0, [Hdrive, eta_fn]], psi0, tlist, c_ops, e_ops=e_ops, options=opts)
    gaussian_results.append(res)

print("Computing thermodynamic observables for all 20 Gaussian variations...")
for res in gaussian_results:
    gaussian_thermos.append(compute_thermodynamics(res, tlist, eps_batt_sorted))

# ------------------------------------------------------------------------------
# 1. PLOT GAUSSIAN LASER PULSE PROFILES (20 VARIATIONS: 5x4 GRID)
# ------------------------------------------------------------------------------
print("Plotting Gaussian laser pulse profiles...")
t_pulse_g = np.linspace(0.0, 1.35, 1500)

fig_pulse_g, axs_pg = plt.subplots(5, 4, figsize=(18, 16), sharex=True)
fig_pulse_g.suptitle(r"Gaussian Laser Pulse Profiles $\eta(t)$ — 20 Variations (Energy-Normalized: $\int |\eta(t)|^2\,dt = I_0$)", y=0.99, fontsize=14)

for ax, title, s, eta_fn, t0_v, color in zip(axs_pg.flat, gaussian_names_base, gaussian_sigmas, gaussian_etas, gaussian_t0s, colors_g):
    y_arr = np.array([eta_fn(t) for t in t_pulse_g])
    label_str = f"FWHM={s*fwhm_factor:.1f} fs ($\\sigma={s*1e3:.1f}$ fs, $t_0={t0_v:.2f}$ ps)"
    ax.plot(t_pulse_g, y_arr, color=color, lw=2.0, label=label_str)
    ax.fill_between(t_pulse_g, 0, y_arr, color=color, alpha=0.18)
    ax.axvline(t0_v, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0.0, 1.35)
    ax.set_ylim(bottom=0, top=y_arr.max() * 1.18)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fontsize=7.2)

for col in range(4):
    axs_pg[4, col].set_xlabel("Time (ps)")
for row in range(5):
    axs_pg[row, 0].set_ylabel(r"$\eta(t)$ ($\mathrm{ps}^{-1}$)")

os.makedirs("figures/gaussian", exist_ok=True)
os.makedirs("data", exist_ok=True)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("figures/gaussian/pulse_profiles_gaussian_comparison.png", dpi=200)
plt.show()

# ------------------------------------------------------------------------------
# 2. PLOT GAUSSIAN POPULATION DYNAMICS (20 VARIATIONS: 5x4 GRID)
# ------------------------------------------------------------------------------
print("Plotting Gaussian population dynamics...")
def plot_populations(ax, res, title):
    P_S10 = np.real(res.expect[0])
    P_S11 = np.real(res.expect[1])
    P_T1  = np.real(res.expect[2])
    ax.plot(tlist, P_T1,  color="#9467bd", lw=2.0, label=r"$T_1$")
    ax.plot(tlist, P_S10, color="#e6a817", lw=2.0, label=r"$S_1^0$")
    ax.plot(tlist, P_S11, color="#17a2b8", lw=2.0, label=r"$S_1^1$")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

fig_pop_g, axs_pop_g = plt.subplots(5, 4, figsize=(18, 16), sharex=True, sharey=True)
fig_pop_g.suptitle("Excited-State Population Dynamics of CuPc in Device D5 — 20 Gaussian Pulses", y=0.99, fontsize=14)

for ax, title, res in zip(axs_pop_g.flat, gaussian_names_base, gaussian_results):
    plot_populations(ax, res, title)

pop_max_g = max(np.real(res.expect[k]).max() for res in gaussian_results for k in range(3))
axs_pop_g[0, 0].set_ylim(0, pop_max_g * 1.1)
axs_pop_g[0, 3].legend(loc="upper right", frameon=False, fontsize=8.5)

for col in range(4):
    axs_pop_g[4, col].set_xlabel("Time (ps)")
for row in range(5):
    axs_pop_g[row, 0].set_ylabel("Population")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("figures/gaussian/population_dynamics_gaussian_comparison.png", dpi=200)
plt.show()

# ------------------------------------------------------------------------------
# 3. PLOT GAUSSIAN ENERGY STORED, CHARGING POWER, AND ERGOTROPY (5x4 GRIDS)
# ------------------------------------------------------------------------------
print("Plotting Gaussian Energy Stored, Charging Power, and Ergotropy...")

# 3a. Energy Stored E(t)
E_lim_g = 1.1 * 1e3 * max(th[0].max() for th in gaussian_thermos)   # meV
fig_eg, axs_eg = plt.subplots(5, 4, figsize=(18, 16), sharex=True, sharey=True)
fig_eg.suptitle(r"Energy Stored $E(t) = \langle H_{\mathrm{batt}} \rangle$ per Molecule — 20 Gaussian Pulses", y=0.99, fontsize=14)
for ax, title, (E_st, _, _), color in zip(axs_eg.flat, gaussian_names_base, gaussian_thermos, colors_g):
    E_meV = E_st * 1e3
    ax.plot(tlist, E_meV, color=color, lw=2.0, label=f"$E_{{\\max}} = {E_meV.max():.1f}$ meV")
    ax.fill_between(tlist, 0, E_meV, color=color, alpha=0.14)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, E_lim_g)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fontsize=8.0)
for col in range(4):
    axs_eg[4, col].set_xlabel("Time (ps)")
for row in range(5):
    axs_eg[row, 0].set_ylabel("Energy Stored (meV)")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("figures/gaussian/energy_stored_gaussian_comparison.png", dpi=200)
plt.show()

# 3b. Charging Power P(t)
fig_pg, axs_pwg = plt.subplots(5, 4, figsize=(18, 16), sharex=True)
fig_pg.suptitle(r"Instantaneous Charging Power $P(t) = dE(t)/dt$ per Molecule — 20 Gaussian Pulses", y=0.99, fontsize=14)
for ax, title, (_, P_ch, _), color in zip(axs_pwg.flat, gaussian_names_base, gaussian_thermos, colors_g):
    ax.plot(tlist, P_ch, color=color, lw=2.0, label=f"$P_{{\\max}} = {P_ch.max():.2f}$ eV/ps")
    ax.fill_between(tlist, 0, P_ch, where=(P_ch >= 0), color=color, alpha=0.18)
    ax.axhline(0, color="gray", ls="--", lw=0.9, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fontsize=8.0)
for col in range(4):
    axs_pwg[4, col].set_xlabel("Time (ps)")
for row in range(5):
    axs_pwg[row, 0].set_ylabel("Power (eV / ps)")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("figures/gaussian/charging_power_gaussian_comparison.png", dpi=200)
plt.show()

# 3c. Ergotropy (Extractable Work)
fig_ergg, axs_ergg = plt.subplots(5, 4, figsize=(18, 16), sharex=True, sharey=True)
fig_ergg.suptitle(r"Ergotropy (Extractable Work) $\mathcal{E}(t)$ vs. Stored Energy $E(t)$ — 20 Gaussian Pulses", y=0.99, fontsize=14)
for ax, title, (E_st, _, Erg), color in zip(axs_ergg.flat, gaussian_names_base, gaussian_thermos, colors_g):
    E_meV = E_st * 1e3
    Erg_meV = Erg * 1e3
    ax.plot(tlist, E_meV, color="gray", ls="--", lw=1.4, alpha=0.75, label="Stored Energy $E(t)$")
    ax.plot(tlist, Erg_meV, color=color, lw=2.0, label=f"Ergotropy ($\\mathcal{{E}}_{{\\max}} = {Erg_meV.max():.1f}$ meV)")
    ax.fill_between(tlist, 0, Erg_meV, color=color, alpha=0.20)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, E_lim_g)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fontsize=7.5)
for col in range(4):
    axs_ergg[4, col].set_xlabel("Time (ps)")
for row in range(5):
    axs_ergg[row, 0].set_ylabel("Energy / Ergotropy (meV)")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("figures/gaussian/ergotropy_gaussian_comparison.png", dpi=200)
plt.show()

# 3d. Peak Population Saturation Analysis Across 20 Gaussian Variations
print("Plotting Gaussian Peak Population Saturation Analysis...")
mult_arr_g = np.array(gaussian_multipliers, dtype=float)
fwhms_g = np.array([s * fwhm_factor for s in gaussian_sigmas])
peak_T1_g  = np.array([np.real(res.expect[2]).max() for res in gaussian_results])
peak_S10_g = np.array([np.real(res.expect[0]).max() for res in gaussian_results])
peak_S11_g = np.array([np.real(res.expect[1]).max() for res in gaussian_results])
peak_E_g   = np.array([th[0].max() * 1e3 for th in gaussian_thermos])
peak_Erg_g = np.array([th[2].max() * 1e3 for th in gaussian_thermos])

fig_sat_g, axs_sat_g = plt.subplots(2, 2, figsize=(16, 11.5))
fig_sat_g.suptitle("Peak Population Saturation Analysis Across 20 Energy-Normalized Gaussian Pulses", fontsize=14, fontweight="bold", y=0.98)

# Panel (a): Absolute Peak Populations vs FWHM
ax_s1 = axs_sat_g[0, 0]
ax_s1.plot(fwhms_g, peak_T1_g, "o-", color="#9467bd", lw=2.2, ms=6, label=r"Peak $T_1$ ($\max_t P_{T_1}$)")
ax_s1.plot(fwhms_g, peak_S10_g, "s-", color="#e6a817", lw=2.2, ms=6, label=r"Peak $S_1^0$ ($\max_t P_{S_1^0}$)")
ax_s1.plot(fwhms_g, peak_S11_g, "^-", color="#17a2b8", lw=2.2, ms=6, label=r"Peak $S_1^1$ ($\max_t P_{S_1^1}$)")

idx_t1_max = int(np.argmax(peak_T1_g))
idx_s10_max = int(np.argmax(peak_S10_g))
idx_s11_max = int(np.argmax(peak_S11_g))

ax_s1.plot(fwhms_g[idx_t1_max], peak_T1_g[idx_t1_max], "o", color="#9467bd", ms=11, mfc="none", mew=2.2)
ax_s1.annotate(f"Saturation: {peak_T1_g[idx_t1_max]:.4f}\n({mult_arr_g[idx_t1_max]:.1f}x, {fwhms_g[idx_t1_max]:.0f} fs)",
               xy=(fwhms_g[idx_t1_max], peak_T1_g[idx_t1_max]), xytext=(fwhms_g[idx_t1_max]-115, peak_T1_g[idx_t1_max]-0.038),
               arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.2), fontsize=8.5, color="#5a3582", fontweight="bold")

ax_s1.plot(fwhms_g[idx_s10_max], peak_S10_g[idx_s10_max], "s", color="#e6a817", ms=11, mfc="none", mew=2.2)
ax_s1.annotate(f"Peak $S_1^0$: {peak_S10_g[idx_s10_max]:.4f}\n({mult_arr_g[idx_s10_max]:.1f}x, {fwhms_g[idx_s10_max]:.0f} fs)",
               xy=(fwhms_g[idx_s10_max], peak_S10_g[idx_s10_max]), xytext=(fwhms_g[idx_s10_max]-25, peak_S10_g[idx_s10_max]+0.024),
               arrowprops=dict(arrowstyle="->", color="#c48a00", lw=1.2), fontsize=8.5, color="#9c6d00", fontweight="bold")

ax_s1.plot(fwhms_g[idx_s11_max], peak_S11_g[idx_s11_max], "^", color="#17a2b8", ms=11, mfc="none", mew=2.2)
ax_s1.annotate(f"Peak $S_1^1$: {peak_S11_g[idx_s11_max]:.4f} ({mult_arr_g[idx_s11_max]:.1f}x)",
               xy=(fwhms_g[idx_s11_max], peak_S11_g[idx_s11_max]), xytext=(fwhms_g[idx_s11_max]-35, peak_S11_g[idx_s11_max]+0.018),
               arrowprops=dict(arrowstyle="->", color="#117a8b", lw=1.2), fontsize=8.0, color="#0d5e6b", fontweight="bold")

ax_s1.axvline(200.0, color="gray", ls="--", lw=1.1, alpha=0.7, label=r"ISC Timescale $\tau_{\mathrm{ISC}} = 200$ fs")
ax_s1.set_title("(a) Peak Excited-State Populations vs. Pulse FWHM", fontsize=11, fontweight="bold", pad=26)
ax_s1.set_xlabel("Gaussian Pulse Width FWHM (fs)")
ax_s1.set_ylabel("Peak Population")
ax_s1.set_ylim(0, 0.275)
ax_s1.grid(True, alpha=0.3)
ax_s1.legend(loc="upper left", frameon=True, fontsize=8.0)

ax_top = ax_s1.twiny()
ax_top.set_xlim(ax_s1.get_xlim())
tick_mults = [0.5, 1.0, 2.5, 4.0, 5.0, 7.0, 9.0, 11.0]
ax_top.set_xticks([m * (sigma_orig * fwhm_factor) for m in tick_mults])
ax_top.set_xticklabels([f"{m:g}x" for m in tick_mults], fontsize=8)
ax_top.set_xlabel(r"Width Multiplier ($\sigma / \sigma_{\mathrm{orig}}$)", fontsize=8.5, labelpad=4)

# Panel (b): Normalized Peak Populations
ax_s2 = axs_sat_g[0, 1]
ax_s2.plot(fwhms_g, peak_T1_g / peak_T1_g.max(), "o-", color="#9467bd", lw=2.2, ms=6, label=r"$T_1$ (Monotonic Saturation at $11.0\times$)")
ax_s2.plot(fwhms_g, peak_S10_g / peak_S10_g.max(), "s-", color="#e6a817", lw=2.2, ms=6, label=r"$S_1^0$ (Peaks at $5.0\times = 175$ fs, then decays)")
ax_s2.plot(fwhms_g, peak_S11_g / peak_S11_g.max(), "^--", color="#17a2b8", lw=2.0, ms=6, label=r"$S_1^1$ (Peaks at $5.0\times = 175$ fs, then decays)")
ax_s2.axvline(fwhms_g[idx_s10_max], color="#e6a817", ls=":", lw=1.5, alpha=0.85, label=f"Singlet Turnover (FWHM = {fwhms_g[idx_s10_max]:.0f} fs)")
ax_s2.axhline(1.0, color="black", ls="--", lw=0.9, alpha=0.5)
ax_s2.set_title(r"(b) Normalized Peak Population ($P_{\max} / P_{\max}^{\mathrm{global}}$) — Saturation vs. Turnover", fontsize=11, fontweight="bold", pad=12)
ax_s2.set_xlabel("Gaussian Pulse Width FWHM (fs)")
ax_s2.set_ylabel("Normalized Peak Population")
ax_s2.set_ylim(0.15, 1.06)
ax_s2.grid(True, alpha=0.3)
ax_s2.legend(loc="lower right", frameon=True, fontsize=8.5)

# Panel (c): Saturation Rate d(P_max) / d(Multiplier)
ax_s3 = axs_sat_g[1, 0]
dm_g = np.diff(mult_arr_g)
fwhm_mid_g = 0.5 * (fwhms_g[:-1] + fwhms_g[1:])
ax_s3.plot(fwhm_mid_g, (np.diff(peak_T1_g) / dm_g) * 1e3, "o-", color="#9467bd", lw=2.0, ms=5.5, label=r"$dP_{\max}(T_1)/d(\sigma/\sigma_0)$ ($\times 10^{-3}$)")
ax_s3.plot(fwhm_mid_g, (np.diff(peak_S10_g) / dm_g) * 1e3, "s-", color="#e6a817", lw=2.0, ms=5.5, label=r"$dP_{\max}(S_1^0)/d(\sigma/\sigma_0)$ ($\times 10^{-3}$)")
ax_s3.plot(fwhm_mid_g, (np.diff(peak_S11_g) / dm_g) * 1e3, "^-", color="#17a2b8", lw=2.0, ms=5.5, label=r"$dP_{\max}(S_1^1)/d(\sigma/\sigma_0)$ ($\times 10^{-3}$)")
ax_s3.axhline(0.0, color="black", ls="--", lw=1.1, alpha=0.7, label=r"Saturation / Turnover Threshold ($dP_{\max}/dm = 0$)")
ax_s3.axvline(fwhms_g[idx_s10_max], color="#e6a817", ls=":", lw=1.4, alpha=0.8)
ax_s3.set_title(r"(c) Saturation Rate $dP_{\max} / d(\sigma/\sigma_{\mathrm{orig}})$ — Asymptotic Convergence", fontsize=11, fontweight="bold")
ax_s3.set_xlabel("Gaussian Pulse Width FWHM (fs)")
ax_s3.set_ylabel(r"Slope $dP_{\max} / dm$ ($\times 10^{-3}$ per $1\times\sigma_{\mathrm{orig}}$)")
ax_s3.grid(True, alpha=0.3)
ax_s3.legend(loc="upper right", frameon=True, fontsize=8.5)

# Panel (d): Peak Stored Energy E_max and Peak Ergotropy vs FWHM
ax_s4 = axs_sat_g[1, 1]
ax_s4.plot(fwhms_g, peak_E_g, "o-", color="#2ca02c", lw=2.2, ms=6, label=r"Peak Stored Energy $E_{\max}$")
ax_s4.plot(fwhms_g, peak_Erg_g, "D-", color="#d62728", lw=2.2, ms=6, label=r"Peak Ergotropy $\mathcal{E}_{\max}$")

idx_e_max = int(np.argmax(peak_E_g))
idx_erg_max = int(np.argmax(peak_Erg_g))
ax_s4.plot(fwhms_g[idx_e_max], peak_E_g[idx_e_max], "o", color="#2ca02c", ms=11, mfc="none", mew=2.2)
ax_s4.annotate(f"$E_{{\\max}}$ Saturation: {peak_E_g[idx_e_max]:.2f} meV\n({mult_arr_g[idx_e_max]:.1f}x, {fwhms_g[idx_e_max]:.1f} fs)",
               xy=(fwhms_g[idx_e_max], peak_E_g[idx_e_max]), xytext=(fwhms_g[idx_e_max]-135, peak_E_g[idx_e_max]-55),
               arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2), fontsize=8.5, color="#1b691b", fontweight="bold")

ax_s4.plot(fwhms_g[idx_erg_max], peak_Erg_g[idx_erg_max], "D", color="#d62728", ms=11, mfc="none", mew=2.2)
ax_s4.annotate(f"Peak Ergotropy: {peak_Erg_g[idx_erg_max]:.2f} meV\n({mult_arr_g[idx_erg_max]:.1f}x, {fwhms_g[idx_erg_max]:.1f} fs)",
               xy=(fwhms_g[idx_erg_max], peak_Erg_g[idx_erg_max]), xytext=(fwhms_g[idx_erg_max]+15, peak_Erg_g[idx_erg_max]+28),
               arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2), fontsize=8.5, color="#a11818", fontweight="bold")

ax_s4.set_title(r"(d) Impact on Battery Thermodynamics: $E_{\max}$ Saturation vs. $\mathcal{E}_{\max}$ Optimum", fontsize=11, fontweight="bold")
ax_s4.set_xlabel("Gaussian Pulse Width FWHM (fs)")
ax_s4.set_ylabel("Energy / Ergotropy (meV)")
ax_s4.set_ylim(50, 430)
ax_s4.grid(True, alpha=0.3)
ax_s4.legend(loc="center right", frameon=True, fontsize=8.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/gaussian/peak_population_saturation_gaussian.png", dpi=200)
plt.show()

# 3e. Post-Peak Exciton Decay Rate Analysis Across 20 Gaussian Variations
print("Plotting Gaussian Post-Peak Exciton Decay Rate Analysis...")
import matplotlib.colors as mcolors

tau_half_s10, tau_1e_s10, k_1e_s10, k_fit_s10, max_dp_s10 = [], [], [], [], []
tau_half_s11, tau_1e_s11, k_1e_s11, k_fit_s11, max_dp_s11 = [], [], [], [], []
k_fit_t1, tau_t1, max_dp_t1 = [], [], []

def extract_decay_metrics(p, t_arr):
    idx_pk = int(np.argmax(p))
    t_pk = t_arr[idx_pk]
    p_pk = p[idx_pk]
    t_dense = np.linspace(t_pk, t_arr[-1], 6000)
    p_dense = np.interp(t_dense, t_arr[idx_pk:], p[idx_pk:])
    t_half = t_dense[np.where(p_dense <= 0.5 * p_pk)[0][0]] - t_pk
    t_1e   = t_dense[np.where(p_dense <= (1.0 / np.e) * p_pk)[0][0]] - t_pk
    dp_dt = -np.gradient(p, t_arr)
    v_max = dp_dt[idx_pk:].max()
    mask = (t_arr > t_pk) & (p <= 0.90 * p_pk) & (p >= 0.20 * p_pk)
    k_fit = -np.polyfit(t_arr[mask] - t_pk, np.log(p[mask]), 1)[0]
    return t_half, t_1e, 1.0 / t_1e, k_fit, v_max

for res in gaussian_results:
    s10 = np.real(res.expect[0])
    s11 = np.real(res.expect[1])
    t1  = np.real(res.expect[2])
    th10, te10, ke10, kf10, vm10 = extract_decay_metrics(s10, tlist)
    th11, te11, ke11, kf11, vm11 = extract_decay_metrics(s11, tlist)
    tau_half_s10.append(th10 * 1e3); tau_1e_s10.append(te10 * 1e3); k_1e_s10.append(ke10); k_fit_s10.append(kf10); max_dp_s10.append(vm10)
    tau_half_s11.append(th11 * 1e3); tau_1e_s11.append(te11 * 1e3); k_1e_s11.append(ke11); k_fit_s11.append(kf11); max_dp_s11.append(vm11)
    idx_t1 = int(np.argmax(t1))
    mask_t1 = tlist >= (tlist[idx_t1] + 0.15)
    kt1 = -np.polyfit(tlist[mask_t1] - tlist[idx_t1], np.log(t1[mask_t1]), 1)[0]
    k_fit_t1.append(kt1)
    tau_t1.append(1.0 / kt1)
    max_dp_t1.append((-np.gradient(t1, tlist))[idx_t1:].max())

tau_half_s10 = np.array(tau_half_s10); tau_1e_s10 = np.array(tau_1e_s10); k_1e_s10 = np.array(k_1e_s10); k_fit_s10 = np.array(k_fit_s10); max_dp_s10 = np.array(max_dp_s10)
tau_half_s11 = np.array(tau_half_s11); tau_1e_s11 = np.array(tau_1e_s11); k_1e_s11 = np.array(k_1e_s11); k_fit_s11 = np.array(k_fit_s11); max_dp_s11 = np.array(max_dp_s11)
k_fit_t1 = np.array(k_fit_t1); tau_t1 = np.array(tau_t1); max_dp_t1 = np.array(max_dp_t1)

fig_dec_g, axs_dec_g = plt.subplots(2, 3, figsize=(20, 11.5))
fig_dec_g.suptitle("Post-Peak Exciton Decay Rate Analysis Across 20 Energy-Normalized Gaussian Pulses", fontsize=15, fontweight="bold", y=0.98)

# Panel (a): Effective Singlet Decay Rates k_{1/e} and k_{fit} vs FWHM
ax_d1 = axs_dec_g[0, 0]
ax_d1.plot(fwhms_g, k_1e_s10, "s-", color="#e6a817", lw=2.2, ms=6, label=r"$S_1^0$ Effective Rate $k_{1/e} = 1/\tau_{1/e}$")
ax_d1.plot(fwhms_g, k_fit_s10, "s--", color="#b88200", lw=1.7, ms=4.5, alpha=0.85, label=r"$S_1^0$ Exp-Fit Rate $k_{\mathrm{fit}}$ ($90\% \to 20\%$)")
ax_d1.plot(fwhms_g, k_1e_s11, "^-", color="#17a2b8", lw=2.2, ms=6, label=r"$S_1^1$ Effective Rate $k_{1/e} = 1/\tau_{1/e}$")
ax_d1.plot(fwhms_g, k_fit_s11, "^--", color="#0f6c7a", lw=1.7, ms=4.5, alpha=0.85, label=r"$S_1^1$ Exp-Fit Rate $k_{\mathrm{fit}}$ ($90\% \to 20\%$)")
ax_d1.axhline(5.5, color="gray", ls=":", lw=1.5, label=r"Intrinsic $\gamma_{\mathrm{ISC}} + \gamma_m = 5.5\ \mathrm{ps}^{-1}$")
ax_d1.set_title(r"(a) Singlet Post-Peak Decay Rate $k_{\mathrm{decay}}$ vs. FWHM", fontsize=11, fontweight="bold", pad=24)
ax_d1.set_xlabel("Gaussian Pulse Width FWHM (fs)")
ax_d1.set_ylabel(r"Decay Rate ($\mathrm{ps}^{-1}$)")
ax_d1.set_ylim(1.5, 6.9)
ax_d1.grid(True, alpha=0.3)
ax_d1.legend(loc="upper right", frameon=True, fontsize=8.0)

ax_d1_top = ax_d1.twiny()
ax_d1_top.set_xlim(ax_d1.get_xlim())
tick_mults_d = [0.5, 1.0, 2.5, 5.0, 7.0, 9.0, 11.0]
ax_d1_top.set_xticks([m * (sigma_orig * fwhm_factor) for m in tick_mults_d])
ax_d1_top.set_xticklabels([f"{m:g}x" for m in tick_mults_d], fontsize=8)
ax_d1_top.set_xlabel(r"Width Multiplier ($\sigma / \sigma_{\mathrm{orig}}$)", fontsize=8.5, labelpad=4)

# Panel (b): Characteristic Post-Peak Decay Times tau_{1/e} and tau_{1/2} (fs)
ax_d2 = axs_dec_g[0, 1]
ax_d2.plot(fwhms_g, tau_1e_s10, "s-", color="#e6a817", lw=2.2, ms=6, label=r"$S_1^0$ $1/e$ Decay Time $\tau_{1/e}$")
ax_d2.plot(fwhms_g, tau_half_s10, "s--", color="#b88200", lw=1.7, ms=4.5, label=r"$S_1^0$ Half-Life $\tau_{1/2}$")
ax_d2.plot(fwhms_g, tau_1e_s11, "^-", color="#17a2b8", lw=2.2, ms=6, label=r"$S_1^1$ $1/e$ Decay Time $\tau_{1/e}$")
ax_d2.plot(fwhms_g, tau_half_s11, "^--", color="#0f6c7a", lw=1.7, ms=4.5, label=r"$S_1^1$ Half-Life $\tau_{1/2}$")
ax_d2.axhline(1000.0 / 5.5, color="gray", ls=":", lw=1.5, label=r"Intrinsic $(\gamma_{\mathrm{ISC}} + \gamma_m)^{-1} = 181.8$ fs")
ax_d2.set_title(r"(b) Singlet Post-Peak Lifetime $\tau_{1/e}$ & Half-Life $\tau_{1/2}$ vs. FWHM", fontsize=11, fontweight="bold", pad=12)
ax_d2.set_xlabel("Gaussian Pulse Width FWHM (fs)")
ax_d2.set_ylabel("Post-Peak Decay Time (fs)")
ax_d2.set_ylim(80, 440)
ax_d2.grid(True, alpha=0.3)
ax_d2.legend(loc="upper left", frameon=True, fontsize=8.0)

# Panel (c): Triplet (T1) Post-Peak Decay Rate & Storage Lifetime
ax_d3 = axs_dec_g[0, 2]
ax_d3.plot(fwhms_g, k_fit_t1 * 1e3, "o-", color="#9467bd", lw=2.2, ms=6, label=r"$T_1$ Effective Decay Rate $k_{\mathrm{eff}}(T_1)$")
ax_d3.axhline(10.0, color="gray", ls="--", lw=1.3, label=r"Bare Triplet Rate $\gamma_T = 10.0 \times 10^{-3}\ \mathrm{ps}^{-1}$")
ax_d3.set_title(r"(c) Triplet ($T_1$) Post-Peak Decay Rate & Storage Lifetime", fontsize=11, fontweight="bold", pad=12)
ax_d3.set_xlabel("Gaussian Pulse Width FWHM (fs)")
ax_d3.set_ylabel(r"$T_1$ Decay Rate $k_{\mathrm{eff}}(T_1)$ ($\times 10^{-3}\ \mathrm{ps}^{-1}$)", color="#5a3582")
ax_d3.tick_params(axis="y", labelcolor="#5a3582")
ax_d3.set_ylim(9.2, 10.2)
ax_d3.grid(True, alpha=0.3)

ax_d3_r = ax_d3.twinx()
ax_d3_r.plot(fwhms_g, tau_t1, "d--", color="#2ca02c", lw=1.8, ms=5, label=r"$T_1$ Storage Lifetime $\tau_{T_1} = 1/k_{\mathrm{eff}}$")
ax_d3_r.set_ylabel(r"Effective Storage Lifetime $\tau_{T_1}$ (ps)", color="#1b691b")
ax_d3_r.tick_params(axis="y", labelcolor="#1b691b")
ax_d3_r.set_ylim(98, 108)
lines_d3, labels_d3 = ax_d3.get_legend_handles_labels()
lines_d3r, labels_d3r = ax_d3_r.get_legend_handles_labels()
ax_d3.legend(lines_d3 + lines_d3r, labels_d3 + labels_d3r, loc="center right", frameon=True, fontsize=8.0)

# Panel (d): Maximum Absolute Post-Peak Decay Speed max(-dP/dt)
ax_d4 = axs_dec_g[1, 0]
ax_d4.plot(fwhms_g, max_dp_s10, "s-", color="#e6a817", lw=2.2, ms=6, label=r"$S_1^0$: $\max_{t>t_{\mathrm{peak}}}(-dP_{S_1^0}/dt)$")
ax_d4.plot(fwhms_g, max_dp_s11, "^-", color="#17a2b8", lw=2.2, ms=6, label=r"$S_1^1$: $\max_{t>t_{\mathrm{peak}}}(-dP_{S_1^1}/dt)$")
ax_d4.plot(fwhms_g, max_dp_t1 * 100.0, "o-", color="#9467bd", lw=2.0, ms=5.5, label=r"$T_1$: $\max_{t>t_{\mathrm{peak}}}(-dP_{T_1}/dt) \times 100$")
ax_d4.annotate("Coherent Polariton Spike\n0.650/ps (0.5x, 17.5 fs)", xy=(fwhms_g[0], max_dp_s10[0]), xytext=(fwhms_g[0]+25, max_dp_s10[0]-0.015),
               arrowprops=dict(arrowstyle="->", color="#9c6d00", lw=1.2), fontsize=8.0, color="#9c6d00", fontweight="bold")
ax_d4.annotate(f"Smooth Peak: {max_dp_s10[3]:.3f}/ps\n(2.5x, 87.5 fs)", xy=(fwhms_g[3], max_dp_s10[3]), xytext=(fwhms_g[3]+45, max_dp_s10[3]+0.03),
               arrowprops=dict(arrowstyle="->", color="#9c6d00", lw=1.2), fontsize=8.0, color="#9c6d00", fontweight="bold")
ax_d4.set_title(r"(d) Maximum Post-Peak Decay Velocity $\max(-dP/dt)$ vs. FWHM", fontsize=11, fontweight="bold")
ax_d4.set_xlabel("Gaussian Pulse Width FWHM (fs)")
ax_d4.set_ylabel(r"Max Decay Velocity $\max(-dP/dt)$ ($\mathrm{ps}^{-1}$)")
ax_d4.set_ylim(0, 0.72)
ax_d4.grid(True, alpha=0.3)
ax_d4.legend(loc="upper right", frameon=True, fontsize=8.0)

# Panel (e): Normalized Post-Peak Decay Curves P(t - t_peak) / P_peak (Semilog)
ax_d5 = axs_dec_g[1, 1]
cmap_d = plt.cm.viridis
norm_d = mcolors.Normalize(vmin=fwhms_g.min(), vmax=fwhms_g.max())
for res, fw in zip(gaussian_results, fwhms_g):
    s10 = np.real(res.expect[0])
    idx_pk = int(np.argmax(s10))
    dt_arr = tlist[idx_pk:] - tlist[idx_pk]
    p_norm = s10[idx_pk:] / s10[idx_pk]
    ax_d5.semilogy(dt_arr, p_norm, color=cmap_d(norm_d(fw)), lw=1.6, alpha=0.85)

dt_ref = np.linspace(0, 0.75, 200)
ax_d5.semilogy(dt_ref, np.exp(-5.5 * dt_ref), "k--", lw=1.6, label=r"Fast ISC Slope $\exp(-(\gamma_{\mathrm{ISC}}+\gamma_m)\Delta t)$ ($5.5\ \mathrm{ps}^{-1}$)")
dt_ref2 = np.linspace(0.6, 2.2, 200)
ax_d5.semilogy(dt_ref2, 0.08 * np.exp(-0.5 * (dt_ref2 - 0.6)), "r:", lw=1.8, label=r"Late Tail Slope $\exp(-\gamma_m \Delta t)$ ($0.5\ \mathrm{ps}^{-1}$)")
ax_d5.axhline(1.0 / np.e, color="gray", ls="-.", lw=1.0, alpha=0.7, label=r"$1/e$ Threshold ($P / P_{\mathrm{peak}} = e^{-1}$)")
ax_d5.set_xlim(0, 2.2)
ax_d5.set_ylim(5e-3, 1.2)
ax_d5.set_title(r"(e) Post-Peak Decay Profiles $P_{S_1^0}(\Delta t) / P_{\mathrm{peak}}$ (Semilog)", fontsize=11, fontweight="bold")
ax_d5.set_xlabel(r"Time After Peak $\Delta t = t - t_{\mathrm{peak}}$ (ps)")
ax_d5.set_ylabel(r"Normalized Population $P(\Delta t) / P_{\mathrm{peak}}$")
ax_d5.grid(True, which="both", alpha=0.25)
ax_d5.legend(loc="upper right", frameon=True, fontsize=7.8)
sm_d5 = plt.cm.ScalarMappable(cmap=cmap_d, norm=norm_d)
sm_d5.set_array([])
cbar_d5 = fig_dec_g.colorbar(sm_d5, ax=ax_d5, pad=0.02)
cbar_d5.set_label("Pulse FWHM (fs)", fontsize=8.5)

# Panel (f): Instantaneous Specific Decay Rate \Gamma(\Delta t) = -(1/P) dP/dt
ax_d6 = axs_dec_g[1, 2]
for res, fw in zip(gaussian_results, fwhms_g):
    s10 = np.real(res.expect[0])
    idx_pk = int(np.argmax(s10))
    dt_arr = tlist[idx_pk:] - tlist[idx_pk]
    gamma_inst = -np.gradient(s10, tlist)[idx_pk:] / np.maximum(s10[idx_pk:], 1e-12)
    ax_d6.plot(dt_arr[1:], gamma_inst[1:], color=cmap_d(norm_d(fw)), lw=1.6, alpha=0.85)

ax_d6.axhline(5.5, color="black", ls="--", lw=1.3, label=r"$\gamma_{\mathrm{ISC}} + \gamma_m = 5.5\ \mathrm{ps}^{-1}$")
ax_d6.axhline(0.5, color="#d62728", ls=":", lw=1.5, label=r"Radiative Limit $\gamma_m = 0.5\ \mathrm{ps}^{-1}$ ($T_1$ Pauli-Blocked)")
ax_d6.set_xlim(0, 2.2)
ax_d6.set_ylim(0, 6.8)
ax_d6.set_title(r"(f) Instantaneous Specific Decay Rate $\Gamma_{S_1^0}(\Delta t) = -\frac{1}{P}\frac{dP}{dt}$", fontsize=11, fontweight="bold")
ax_d6.set_xlabel(r"Time After Peak $\Delta t = t - t_{\mathrm{peak}}$ (ps)")
ax_d6.set_ylabel(r"Instantaneous Decay Rate $\Gamma(\Delta t)$ ($\mathrm{ps}^{-1}$)")
ax_d6.grid(True, alpha=0.3)
ax_d6.legend(loc="upper right", frameon=True, fontsize=8.0)
sm_d6 = plt.cm.ScalarMappable(cmap=cmap_d, norm=norm_d)
sm_d6.set_array([])
cbar_d6 = fig_dec_g.colorbar(sm_d6, ax=ax_d6, pad=0.02)
cbar_d6.set_label("Pulse FWHM (fs)", fontsize=8.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/gaussian/decay_rate_analysis_gaussian.png", dpi=200)
plt.show()
print("All 20 Gaussian plots saved successfully.")

# ------------------------------------------------------------------------------
# 4. SAVE GAUSSIAN SIMULATION RESULTS
# ------------------------------------------------------------------------------
print("\nSaving Gaussian simulation results...")
eV_to_J = 1.602176634e-19
ps_to_s = 1.0e-12

save_dict_g = {"tlist": tlist}
csv_columns_g = [tlist]
csv_headers_g = ["time_ps"]

for m, res, (E_st, P_ch, Erg) in zip(gaussian_multipliers, gaussian_results, gaussian_thermos):
    tag = f"Gaussian_{m:.2f}x"
    P_S10 = np.real(res.expect[0])
    P_S11 = np.real(res.expect[1])
    P_T1  = np.real(res.expect[2])
    n_ph  = np.real(res.expect[3])
    
    E_meV = E_st * 1e3
    Erg_meV = Erg * 1e3
    E_pass_meV = np.maximum(0.0, E_meV - Erg_meV)
    P_inst_total_W = np.gradient(N * E_st * eV_to_J, tlist * ps_to_s)
    
    save_dict_g[f"{tag}_P_S10"] = P_S10
    save_dict_g[f"{tag}_P_S11"] = P_S11
    save_dict_g[f"{tag}_P_T1"]  = P_T1
    save_dict_g[f"{tag}_n_ph"]  = n_ph
    save_dict_g[f"{tag}_E_eV"]  = E_st
    save_dict_g[f"{tag}_E_meV"] = E_meV
    save_dict_g[f"{tag}_E_passive_meV"] = E_pass_meV
    save_dict_g[f"{tag}_P_charge_eV_ps"] = P_ch
    save_dict_g[f"{tag}_P_inst_total_W"] = P_inst_total_W
    save_dict_g[f"{tag}_Ergotropy_eV"]  = Erg
    save_dict_g[f"{tag}_Ergotropy_meV"] = Erg_meV
    
    csv_columns_g.extend([P_S10, P_S11, P_T1, n_ph, E_meV, P_ch, Erg_meV])
    csv_headers_g.extend([
        f"{tag}_S10", f"{tag}_S11", f"{tag}_T1", f"{tag}_n_ph",
        f"{tag}_E_meV", f"{tag}_P_eV_ps", f"{tag}_Erg_meV"
    ])

np.savez_compressed("data/protocols_sim_results_gaussian.npz", **save_dict_g)
print("-> Saved Gaussian simulation data to 'data/protocols_sim_results_gaussian.npz'")

csv_matrix_g = np.column_stack(csv_columns_g)
np.savetxt(
    "data/protocols_sim_results_gaussian.csv",
    csv_matrix_g,
    delimiter=",",
    header=",".join(csv_headers_g),
    comments=""
)
end_time = time.time()

# 5. Output the results
print("\n=== Simulation Complete ===")
print(f"Total time elapsed               : {end_time - start_time:.2f} seconds.")
print("-> Saved Gaussian summary CSV to 'data/protocols_sim_results_gaussian.csv'")
