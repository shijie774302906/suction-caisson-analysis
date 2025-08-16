import math

# -------------------------------------------------
# 1. INPUT DATA  (diameter–based, θ fixed 1.5°,注意这里需要修改Zo的系数)
# -------------------------------------------------
PARAM = dict(
    # main caisson
    D   = 0.12,     # diameter D1 (m)
    L   = 0.12,     # skirt length L1 (m)
    t   = 0.002,     # wall thickness t1 (m)

    # circular outer skirt
    R2  = 0.030,     # skirt‑width radius  (m)
    L2  = 0.060,     # skirt wall length   (m)
    t2  = 0.002,     # skirt wall thickness(m)

    # soil & interface
    gamma = 7.85,    # γ'  (kN/m³)
    phi   = 38.6,    # φ'  (deg)
    K0    = 0.38,
    Ka    = 0.25,
    fc    = 0.18,

    # Winkler moduli
    m  = 20e3,       # kN/m⁴
    Kv = 10.053e3    # kN/m³
)
THETA = math.radians(1.5)            # fixed rotation 1.5°
ECC   = 1.5                          # H eccentricity e = M/(H·D)

# -------------------------------------------------
# 2. HELPER FUNCTIONS
# -------------------------------------------------
NqNg = lambda ph: (math.exp(math.pi*math.tan(ph))*math.tan(math.pi/4+ph/2)**2,
                  1.5*(math.exp(math.pi*math.tan(ph))*math.tan(math.pi/4+ph/2)**2-1)*math.tan(ph))
pu = lambda γ,L,t,Nq,Ng: γ*L*Nq + 0.5*γ*t*Ng

# mobilisation factors (skirt only)
F_SKIRT = dict(eta_p=1, fc_eff=1, h_eff=1, Ka_eff=1, tip_eff=1)

# -------------------------------------------------
# 3. CORE SOLVER
# -------------------------------------------------

def horizontal_capacity(data: dict, use_skirt: bool, mobilise: bool=True):
    """return Hu, Mu (kN, kN·m) for given geometry"""
    D1, L1, t1 = data['D'], data['L'], data['t']
    R1 = D1/2

    # --- skirt geometry (may be zero) ---
    if use_skirt:
        R2, L2, t2 = data['R2'], data['L2'], data['t2']
    else:
        R2 = L2 = t2 = 0.0
    Dsk = D1 + 2*R2     # lid radius for Kv term

    # soil constants
    γ , fc, m, Kv = data['gamma'], data['fc'], data['m'], data['Kv']
    K0, Ka        = data['K0'], data['Ka']
    φ  = math.radians(data['phi'])
    Nq, Ng        = NqNg(φ)

    # ---- mobilisation factors for skirt ----
    if use_skirt and mobilise:
        Kp_full = K0 + 2*math.sin(φ)
        Kp = min(2*K0, Kp_full) * F_SKIRT['eta_p']
        fc_s = fc * F_SKIRT['fc_eff']
        L2e  = L2 * F_SKIRT['h_eff']
        Ka_s = F_SKIRT['Ka_eff']
        tipF = F_SKIRT['tip_eff']
    else:
        Kp, fc_s, L2e, Ka_s, tipF = K0+2*math.sin(φ), fc, L2, Ka, 1.0

    # base & skirt tip resistances
    pu1 = pu(γ, L1, t1, Nq, Ng)
    pu2 = pu(γ, L2, t2, Nq, Ng) * tipF if R2 else 0.0

    z0  = 0.55*L1
    C1  = math.pi/24 * m * z0**3 * THETA      # *D
    C2  = math.pi/48 * m * fc  * z0**3 * THETA # *D²
    C3  = math.pi/128* Kv * THETA              # *D³
    C4  = math.pi/24 * m * L2e**3 * THETA      # *D
    C5  = math.pi/48 * m * fc  * L2e**3 * THETA # *D²

    # ---- original caisson M1–M8 ----
    M1 = C1*D1*(L1-z0/2)
    M2 = 1/6*γ*K0*D1*L1**3; 
    M3 = C2*D1**2
    M4 = -1/6*γ*Ka*D1*L1**3
    M5 = 0.25*K0*γ*fc*L1**2*D1**2; M6 = 0.25*Ka*γ*fc*L1**2*D1**2
    M7 = pu1*t1*D1**2/2
    M8 = C3*Dsk**4           # lid spring (Dsk≥D1)

    # ---- skirt additions ----
    M9=M10=M11=M12=M7s=0.0
    if R2>0:
        M9 = m * THETA * math.pi * Dsk * ( L2e**4 / 16- (L1 + z0) * L2e**3 / 12
        + L1 * z0 * L2e**2 / 8) + K0 * γ * Dsk * (L1 * L2e**2 / 2- L2e**3 / 3)
        M10 = (math.pi * m * fc_s * THETA * Dsk**2 * (-L2e**3 / 24 + z0 * L2e**2 / 16)
        + 0.25 * K0 * γ * fc_s * Dsk**2 * L2e**2)
        M11 = -Ka_s * γ * Dsk * (L1 * L2e**2 / 2 - L2e**3 / 3)
        M12 = 0.25*γ*Ka_s*fc_s*L2e**2*Dsk**2
        M7s = pu2*t2*Dsk**2/2
        print("\n--- Skirt Moment Contributions (M9-M12) ---")
        print(f"M1  (Skirt Passive Pressure Moment) = {M1:.4f} kN·m")
        print(f"M2  (Skirt Passive Pressure Moment) = {M2:.4f} kN·m")
        print(f"M3  (Skirt Passive Pressure Moment) = {M3:.4f} kN·m")
        print(f"M4  (Skirt Passive Pressure Moment) = {M4:.4f} kN·m")
        print(f"M5  (Skirt Passive Pressure Moment) = {M5:.4f} kN·m")
        print(f"M6  (Skirt Passive Pressure Moment) = {M6:.4f} kN·m")
        print(f"M7  (Skirt Passive Pressure Moment) = {M7:.4f} kN·m")
        print(f"M8  (Skirt Passive Pressure Moment) = {M8:.4f} kN·m")
        print(f"M9  (Skirt Passive Pressure Moment) = {M9:.4f} kN·m")
        print(f"M10 (Skirt Passive Friction Moment) = {M10:.4f} kN·m")
        print(f"M11 (Skirt Active Pressure Moment)  = {M11:.4f} kN·m")
        print(f"M12 (Skirt Active Friction Moment)  = {M12:.4f} kN·m")
        print(f"M7s (Skirt Active Friction Moment)  = {M7s:.4f} kN·m")
        print("-------------------------------------------")
        Kg=M8/(M1+M2+M3+M4+M5+M6+M7+M8)
        print(f"Kg (lid spring ratio) = {Kg:.4f} (M8/(M1+M2+M3+M4+M5+M6+M7+M8))")
    M0 = M1+M2+M3+M4+M5+M6+M7+M8+M9+M10+M11+M12+M7s
    Mu = (ECC*D1)/(ECC*D1+L1)*M0
    Hu = Mu/(ECC*D1)
    return Hu, Mu

# -------------------------------------------------
# 4. RUN & PRINT
# -------------------------------------------------
Hu_no, Mu_no = horizontal_capacity(PARAM, use_skirt=False)
Hu_sk, Mu_sk = horizontal_capacity(PARAM, use_skirt=True, mobilise=True)

print("θ  = 1.5°  (fixed)")
print(f"H_u  (no‑skirt)      = {Hu_no:.3f} kN | Mu = {Mu_no:.4f} kN·m")
print(f"H_u  (with skirt adj) = {Hu_sk:.3f} kN | Mu = {Mu_sk:.4f} kN·m")
print(f"Increase              = {(Hu_sk/Hu_no-1)*100:6.1f} %")

