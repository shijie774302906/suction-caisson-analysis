import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# 1. INPUT SECTION
# ------------------------
params = {
    "D": 0.12,            # Diameter (m)
    "L": 0.12,            # Skirt length (m)
    "t": 0.002,          # Skirt thickness (m)
    "gamma_prime": 7.85, # Effective unit weight (kN/m³)
    "phi_deg": 38.6,     # Friction angle φ' (deg)
    "K0": 0.38,           # Earth pressure at rest
    "Ka": 0.25,         # Active earth pressure
    "m": 20e3,          # Lateral modulus (kN/m⁴)
    "Kv": 10.053e3,        # Vertical modulus (kN/m³)
    "fc": 0.18          # Soil–skirt friction coeff
}

eccentricities = [1.0, 1.5, 2.6]
measured_Mu = {1.0: 0.048, 1.5: 0.048, 2.6: 0.048}

# ------------------------
# 2. CALC FUNCTIONS
# ------------------------
def bearing_coefficients(phi_rad):
    Nq = math.exp(math.pi * math.tan(phi_rad)) * math.tan(math.pi / 4 + phi_rad / 2)**2
    Ng = 1.8 * (Nq - 1) * math.tan(phi_rad)
    return Nq, Ng

def critical_rotation_and_capacity(e, p):
    D, L, t = p["D"], p["L"], p["t"]
    γp, K0, Ka, m, Kv, fc = (p["gamma_prime"], p["K0"], p["Ka"], p["m"], p["Kv"], p["fc"])
    φ = math.radians(p["phi_deg"])
    z0 = 0.8 * L
    V = 0.6
    Nq, Ng = bearing_coefficients(φ)
    pu = γp * L * Nq + 0.5 * γp * t * Ng

    num = 12*V - 6*math.pi*D*t*pu - 3*(K0-Ka)*γp*fc*L**2*D
    den = Kv*D**3 + 2*m*z0**3*D*fc
    θ0 = 1.5 * (math.pi / 180)
    #θ0 = num/den
    M1 = math.pi/24 * m * D * θ0 * z0**3 * (L - z0/2) # M1ult第一项
    M2 = (1/6) * γp * D * K0 * L**3 # M1ult第二项
    M3 = math.pi/48 * m * fc * D**2 * θ0 * z0**3 # M2ult第一项
    M4 = 0.25 * K0 * γp * fc * L**2 * D**2 # M2ult第二项
    M5 = -γp * D/6 * Ka * L**3 # M3ult
    M6 = 0.25 * Ka * γp * fc * L**2 * D**2 # M4ult 
    M7 = pu * t * D**2 / 2 # M5ult
    M8 = math.pi/128 * Kv * D**4 * θ0   # M6ult

    M0u = M1 + M2 + M3 + M4 + M5 + M6 + M7 + M8
    h = e * D
    Mu = (h / (h + L)) * M0u
    Hu = Mu / (e * D)
    return θ0, Mu, Hu

# ------------------------
# 3. CALCULATE & PRINT
# ------------------------
results = []
for e in eccentricities:
    θ0, Mu_pred, Hu_pred = critical_rotation_and_capacity(e, params)
    Mu_meas = measured_Mu[e]
    Hu_meas = Mu_meas / (e * params["D"])
    results.append({
        "e": e,
        "θ₀ (rad)": θ0,
        "Mu pred (kN·m)": Mu_pred,
        "Mu meas (kN·m)": Mu_meas,
        "Hu pred (kN)": Hu_pred,
        "Hu meas (kN)": Hu_meas
    })

df = pd.DataFrame(results).set_index("e")
print(df.round(3))

# ------------------------
# 4. PLOT θ–H
# ------------------------
plt.figure(figsize=(7,5))
for row in results:
    θ0, Hu = row["θ₀ (rad)"], row["Hu pred (kN)"]
    θ = np.linspace(0, θ0, 50)
    H = Hu / θ0 * θ
    plt.plot(θ, H, label=f"e={row['e']}")

plt.xlabel("Rotation θ (rad)")
plt.ylabel("Horizontal load H (kN)")
plt.title("θ–H curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
