import math

# -------------------------------------------------
# 1. INPUT DATA  (diameter–based, θ fixed 1.5°)
# -------------------------------------------------
PARAM = dict(
    # main caisson
    D   = 2.4,     # diameter D1 (m)
    L   = 2.4,     # skirt length L1 (m)
    t   = 0.04,     # wall thickness t1 (m)

    # circular outer skirt
    R2  = 1.2,     # skirt‑width radius  (m)
    L2  = 0,     # skirt wall length   (m)
    t2  = 0,     # skirt wall thickness(m)

    # soil & interface
    gamma = 7.85,    # γ'  (kN/m³)
    phi   = 40,    # φ'  (deg)
    K0    = 0.38,
    Ka    = 0.25,
    fc    = 0.18,

    # Winkler moduli
    m  = 2.15e3,       # kN/m⁴
    Kv = 22e2    # kN/m³
)

# -------------------------------------------------
# Case Configuration (Three Cases Configuration)
# -------------------------------------------------
# z0 coefficients (rotation center coefficients)
Z0_COEFFS = {
    'case1_no_skirt': 0.8,      # Case 1: No skirt
    'case2_full_skirt': 0.50,   # Case 2: Full skirt  
    'case3_simple_skirt': 0.6   # Case 3: Simplified skirt
}

# Case parameter configuration
CASE_CONFIGS = {
    'case1': {
        'name': 'Case 1: No Skirt',
        'R2': 0.0, 'L2': 0.0, 't2': 0.0,
        'z0_coeff': Z0_COEFFS['case1_no_skirt'],
        'use_skirt': False
    },
    'case2': {
        'name': 'Case 2: Full Skirt',
        'R2': 1.2, 'L2': 1.2, 't2': 0.04,
        'z0_coeff': Z0_COEFFS['case2_full_skirt'],
        'use_skirt': True
    },
    'case3': {
        'name': 'Case 3: Simplified Skirt',
        'R2': 1.2, 'L2': 0.0, 't2': 0.0,
        'z0_coeff': Z0_COEFFS['case3_simple_skirt'],
        'use_skirt': True
    }
}
THETA = math.radians(1.5)            # fixed rotation 1.5°
ECC   = 3.58                          # H eccentricity e = M/(H·D)

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

def horizontal_capacity(data: dict, case_config: dict, mobilise: bool=True):
    """
    Return Hu, Mu (kN, kN·m) for given geometry and case configuration
    
    Args:
        data: Basic parameter dictionary
        case_config: Case configuration dictionary containing R2, L2, t2, z0_coeff, use_skirt
        mobilise: Whether to consider mobilisation effects
    """
    D1, L1, t1 = data['D'], data['L'], data['t']
    R1 = D1/2

    # --- Get skirt geometry parameters from case configuration ---
    R2 = case_config['R2']
    L2 = case_config['L2'] 
    t2 = case_config['t2']
    use_skirt = case_config['use_skirt']
    z0_coeff = case_config['z0_coeff']
    
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

    # Use case-specific rotation center coefficient
    z0  = z0_coeff * L1
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
        
    M0 = M1+M2+M3+M4+M5+M6+M7+M8+M9+M10+M11+M12+M7s
    Mu = (ECC*D1)/(ECC*D1+L1)*M0
    Hu = Mu/(ECC*D1)
    return Hu, Mu, z0, M0, (M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M7s)

# -------------------------------------------------
# 4. RUN & PRINT - Three Case Analysis
# -------------------------------------------------
def print_case_results(case_name, Hu, Mu, z0, M0, moments, config):
    """Print results for a single case"""
    M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M7s = moments
    print(f"\n{'='*60}")
    print(f"{case_name}")
    print(f"{'='*60}")
    print(f"Configuration: R2={config['R2']:.1f}m, L2={config['L2']:.1f}m, t2={config['t2']:.3f}m")
    print(f"Rotation center: z0 = {config['z0_coeff']:.2f}*L1 = {z0:.3f}m")
    print(f"Horizontal capacity: H_u = {Hu:.3f} kN")
    print(f"Moment capacity: M_u = {Mu:.4f} kN·m")
    print(f"Total moment: M0 = {M0:.4f} kN·m")
    
    if config['use_skirt'] and config['R2'] > 0:
        print(f"\n--- Moment Component Details ---")
        print(f"M1  (Main caisson passive pressure moment) = {M1:.4f} kN·m")
        print(f"M2  (Main caisson at-rest pressure moment) = {M2:.4f} kN·m") 
        print(f"M3  (Main caisson friction moment) = {M3:.4f} kN·m")
        print(f"M4  (Main caisson active pressure moment) = {M4:.4f} kN·m")
        print(f"M5  (Main caisson at-rest friction moment) = {M5:.4f} kN·m")
        print(f"M6  (Main caisson active friction moment) = {M6:.4f} kN·m")
        print(f"M7  (Main caisson base resistance moment) = {M7:.4f} kN·m")
        print(f"M8  (Lid spring moment) = {M8:.4f} kN·m")
        if config['L2'] > 0:  # Only full skirt has M9-M12
            print(f"M9  (Skirt passive pressure moment) = {M9:.4f} kN·m")
            print(f"M10 (Skirt passive friction moment) = {M10:.4f} kN·m") 
            print(f"M11 (Skirt active pressure moment) = {M11:.4f} kN·m")
            print(f"M12 (Skirt active friction moment) = {M12:.4f} kN·m")
            print(f"M7s (Skirt base resistance moment) = {M7s:.4f} kN·m")
        Kg = M8/(M1+M2+M3+M4+M5+M6+M7+M8) if (M1+M2+M3+M4+M5+M6+M7+M8) > 0 else 0
        print(f"Kg (lid spring ratio) = {Kg:.4f}")

print(f"{'='*80}")
print(f"Suction Caisson Horizontal Capacity - Three Case Comparative Analysis")
print(f"Fixed parameters: θ = 1.5°, ECC = {ECC}")
print(f"{'='*80}")

# Calculate three cases
results = {}
for case_key, config in CASE_CONFIGS.items():
    Hu, Mu, z0, M0, moments = horizontal_capacity(PARAM, config, mobilise=True)
    results[case_key] = {'Hu': Hu, 'Mu': Mu, 'z0': z0, 'M0': M0, 'moments': moments}
    print_case_results(config['name'], Hu, Mu, z0, M0, moments, config)

# Comparative analysis
print(f"\n{'='*60}")
print(f"Three Case Comparative Analysis")
print(f"{'='*60}")
case1_Hu = results['case1']['Hu']
case2_Hu = results['case2']['Hu'] 
case3_Hu = results['case3']['Hu']
case1_Mu = results['case1']['Mu']
case2_Mu = results['case2']['Mu'] 
case3_Mu = results['case3']['Mu']

print(f"Case 1 (No skirt):         H_u = {case1_Hu:.3f} kN | M_u = {case1_Mu:.4f} kN·m")
print(f"Case 2 (Full skirt):       H_u = {case2_Hu:.3f} kN | M_u = {case2_Mu:.4f} kN·m")
print(f"Case 3 (Simplified skirt): H_u = {case3_Hu:.3f} kN | M_u = {case3_Mu:.4f} kN·m")
print(f"\nHorizontal capacity relative improvement:")
print(f"Case 2 vs Case 1:  {(case2_Hu/case1_Hu-1)*100:6.1f} %")
print(f"Case 3 vs Case 1:  {(case3_Hu/case1_Hu-1)*100:6.1f} %")
print(f"Case 2 vs Case 3:  {(case2_Hu/case3_Hu-1)*100:6.1f} %")
print(f"\nMoment capacity relative improvement:")
print(f"Case 2 vs Case 1:  {(case2_Mu/case1_Mu-1)*100:6.1f} %")
print(f"Case 3 vs Case 1:  {(case3_Mu/case1_Mu-1)*100:6.1f} %")
print(f"Case 2 vs Case 3:  {(case2_Mu/case3_Mu-1)*100:6.1f} %")


