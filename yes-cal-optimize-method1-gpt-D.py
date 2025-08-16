import math, numpy as np, pandas as pd, matplotlib.pyplot as plt

# ---------------- 1. INPUT PANEL ---------------- #
P = dict(
    # Geometry (m)
    D1=0.120, L1=0.240, t1=0.002,
    R2_0=0.030,                # baseline skirt half-width
    L2=0.060,  t2=0.002, t3=0.010,
    # Soil & interface
    phi_deg=38.6,  gamma=7.85, fc=0.18,
    # Winkler & load
    m=60e3, Kv=10.053e3, e_ecc=1.5, theta_deg=1.5,
    # Sweep
    lam_max=2.0, pts=120, export=True
)

phi = math.radians(P['phi_deg'])
theta = math.radians(P['theta_deg'])
K0, Ka = 1-math.sin(phi), math.tan(math.pi/4-phi/2)**2
Nq_th = math.exp(math.pi*math.tan(phi))*math.tan(math.pi/4+phi/2)**2
Ny_th = 1.5*(Nq_th-1)*math.tan(phi)

# ---------- 2. CORE FUNCTIONS (return total & skirt) ---------- #
def bearing_capacity(R2):
    """return (Mu_total, M_skirt_inc)"""
    D1,L1,t1,L2,t2 = P['D1'],P['L1'],P['t1'],P['L2'],P['t2']
    m,Kv,gamma,fc = P['m'],P['Kv'],P['gamma'],P['fc']
    z0=0.6*L1; C1=math.pi/24*m*z0**3*theta; C2=math.pi/48*m*fc*z0**3*theta
    C3=math.pi/128*Kv*theta
    M1=C1*D1*(L1-z0/2); M2=gamma*K0*D1*L1**3/6; M3=C2*D1**2
    M4=-gamma*Ka*D1*L1**3/6; M5=0.25*gamma*K0*fc*L1**2*D1**2
    M6=0.25*gamma*Ka*fc*L1**2*D1**2
    M7=0.5*(gamma*L1*Nq_th+0.5*gamma*t1*Ny_th)*t1*D1**2
    Dsk=D1+2*R2
    Delta_M8=C3*(Dsk**4-D1**4)          # 盖板增量
    M8=C3*Dsk**4
    if R2<1e-9:
        Mu=(P['e_ecc']*D1)/(P['e_ecc']*D1+L1)*(M1+M2+M3+M4+M5+M6+M7+M8)
        return Mu, 0.0
    # skirt terms (empirical)
    mπθ=math.pi*m*theta; L2e=L2
    M9 = mπθ*Dsk*(L2e**4/16-(L1+z0)*L2e**3/12+L1*z0*L2e**2/8)+gamma*K0*Dsk*(L1*L2e**2/2-L2e**3/3)
    M10= mπθ*fc*Dsk**2*(-L2e**3/24+z0*L2e**2/16)+0.25*gamma*K0*fc*Dsk**2*L2e**2
    M11=-gamma*Ka*Dsk*(L1*L2e**2/2-L2e**3/3)
    M12= 0.25*gamma*Ka*fc*L2e**2*Dsk**2
    Nq2=73.9*(1+7.25*(L2/L1)**1.16*(L1/D1)**(-2.09))
    Ny2=1.8*(Nq2-1)*0.9
    M7s=0.5*(gamma*L2*Nq2+0.5*gamma*t2*Ny2)*t2*Dsk**2
    M_sk=M9+M10+M11+M12+M7s+Delta_M8
    M_tot=M1+M2+M3+M4+M5+M6+M7+M8+M9+M10+M11+M12+M7s
    Mu=(P['e_ecc']*D1)/(P['e_ecc']*D1+L1)*M_tot
    return Mu, M_sk

def penetration_resistance(R2):
    """return (R_total, R_skirt)"""
    D1,L1,t1,L2,t2=P['D1'],P['L1'],P['t1'],P['L2'],P['t2']
    gamma=P['gamma']
    Ktan1=(1-math.sin(phi))*math.tan(phi)
    R_main=seg_resistance(L1,D1,D1+2*t1,t1,gamma,Nq_th,Ny_th,Ktan1)
    if R2<1e-9: return R_main,0.0
    Dsi=D1+2*R2; Dso=Dsi+2*t2
    Ktan2=0.478*(1+0.746*(L2/L1)*(L1/D1)**(-1.32))
    Nq2=73.9*(1+7.25*(L2/L1)**1.16*(L1/D1)**(-2.09))
    Ny2=1.8*(Nq2-1)*0.9
    R_sk=seg_resistance(L2,Dsi,Dso,t2,gamma,Nq2,Ny2,Ktan2)
    return R_main+R_sk, R_sk

def material_volume(R2):
    """return (V_total, V_skirt)"""
    D1,L1,t1,L2,t2,t3=P['D1'],P['L1'],P['t1'],P['L2'],P['t2'],P['t3']
    V_main=math.pi*(D1+t1)*t1*L1
    if R2<1e-9:
        V_lid=math.pi/4*(D1+2*t1)**2*t3
        return V_main+V_lid,0.0
    Dskc=D1+2*R2+t2
    V_sk=math.pi*Dskc*t2*L2
    D_out=D1+2*R2+2*t2; D_in=D1+2*t1
    V_lid=math.pi/4*(D_out**2-D_in**2)*t3
    return V_main+V_sk+V_lid, V_sk+V_lid

def seg_resistance(h,Di,Do,t,gamma,Nq,Ny,Ktan):
    if h<=1e-9: return 0.0
    Zo=Do/(4*Ktan); Zi=Di/(4*Ktan)
    Fo=gamma*Zo**2*(math.exp(h/Zo)-1-h/Zo)*Ktan*math.pi*Do
    Fi=gamma*Zi**2*(math.exp(h/Zi)-1-h/Zi)*Ktan*math.pi*Di
    sigma_v=gamma*Zo*(math.exp(h/Zo)-1)
    Q=max(0,sigma_v*Nq+0.5*gamma*t*Ny)*math.pi*(Di+t)*t
    return Fo+Fi+Q

# ---------- 3. BASELINE (λ=1) ---------- #
R2_base=P['R2_0']
Mu0,Msk0=bearing_capacity(R2_base)
Rf0,Rsk0=penetration_resistance(R2_base)
Vs0,Vsk0=material_volume(R2_base)

# ---------- 4. SWEEP λ ---------- #
lam_min=max(1.01*P['t1']/P['R2_0'],0.05)
lam=np.linspace(lam_min,P['lam_max'],P['pts'])
Mu,Rf,Vs,Mu_sk,Rf_sk,Vs_sk=[],[],[],[],[],[]
for la in lam:
    R2=la*R2_base
    mu,msk=bearing_capacity(R2); Mu.append(mu); Mu_sk.append(msk)
    rf,rsk=penetration_resistance(R2); Rf.append(rf); Rf_sk.append(rsk)
    vs,vsk=material_volume(R2); Vs.append(vs); Vs_sk.append(vsk)

Mu=np.array(Mu); Rf=np.array(Rf); Vs=np.array(Vs)
Mu_sk=np.array(Mu_sk); Rf_sk=np.array(Rf_sk); Vs_sk=np.array(Vs_sk)

ratios=dict(
    Mu=Mu/Mu0, R=Rf/Rf0, V=Vs/Vs0,
    Mu_sk=Mu_sk/Msk0, R_sk=Rf_sk/Rsk0, V_sk=Vs_sk/Vsk0
)

# ---------- 5. PLOT ---------- #
plt.figure(figsize=(11,7))
plt.plot(lam, ratios['Mu'],'-o',label='Mu/Mu0'); plt.plot(lam,ratios['R'],'-s',label='Rf/Rf0')
plt.plot(lam, ratios['V'],'-^',label='Vs/Vs0')
plt.plot(lam, ratios['Mu_sk'],'--o',label='M_sk/M_sk0')
plt.plot(lam, ratios['R_sk'],'--s',label='R_sk/R_sk0')
plt.plot(lam, ratios['V_sk'],'--^',label='V_sk/V_sk0')
plt.axvline(1,color='k',ls='--'); plt.axhline(1,color='gray',ls=':')
plt.xlabel('λ = R2 / R2₀'); plt.ylabel('Normalized'); plt.grid(); plt.legend()
plt.title('Overall vs Skirt-only Normalized Metrics'); plt.show()

# ---------- 6. EXPORT ---------- #
if P['export']:
    df=pd.DataFrame({'lambda':lam,**ratios})
    df.to_excel('/mnt/data/ssc_R2_sweep_full.xlsx',index=False)
    print('Excel saved.')
