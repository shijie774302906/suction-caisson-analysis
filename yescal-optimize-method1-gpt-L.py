# -*- coding: utf-8 -*-
"""
固定 R2，参数化 L2 的裙式吸力筒优化分析
------------------------------------------------
输出 2×3 条归一化曲线 + Excel 数据
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. INPUT PANEL
# ------------------------------------------------------------------
P = dict(
    # Geometry (m)
    D1=0.120, L1=0.240, t1=0.002,
    R2_fixed=0.030,          # 固定裙宽
    L2_base =0.060,          # 基准裙长  (≠0)
    t2=0.002, t3=0.010,
    # Soil & interface
    phi_deg=38.6, gamma=7.85, fc=0.18,
    # Winkler & loading
    m=60e3, Kv=10.053e3, theta_deg=1.5, e_ecc=1.5,
    # Sweep
    eta_max=2.0, pts=120,
    export=True
)

# ------------------------------------------------------------------
# 2. PRE-COMPUTED CONSTANTS
# ------------------------------------------------------------------
phi   = math.radians(P['phi_deg'])
theta = math.radians(P['theta_deg'])
K0    = 1 - math.sin(phi)
Ka    = math.tan(math.pi/4 - phi/2)**2
Nq_th = math.exp(math.pi*math.tan(phi))*math.tan(math.pi/4+phi/2)**2
Ny_th = 1.5*(Nq_th-1)*math.tan(phi)          # 与承载公式一致

# ------------------------------------------------------------------
# 3. HELPER FUNCTION
# ------------------------------------------------------------------
def seg_resistance(h, Di, Do, t, g, Nq, Ny, Ktan):
    """李(2011) 闭式：单段圆筒在深度 h 的贯入阻力"""
    if h <= 1e-9:
        return 0.0
    Zo, Zi = Do/(4*Ktan), Di/(4*Ktan)
    Fo = g*Zo**2*(math.exp(h/Zo)-1-h/Zo)*Ktan*math.pi*Do
    Fi = g*Zi**2*(math.exp(h/Zi)-1-h/Zi)*Ktan*math.pi*Di
    sigma_v = g*Zo*(math.exp(h/Zo)-1)
    Qtip = max(0, sigma_v*Nq + 0.5*g*t*Ny)*math.pi*(Di+t)*t
    return Fo + Fi + Qtip

# ------------------------------------------------------------------
# 4. CORE MODELS  (depend on variable L2)
# ------------------------------------------------------------------
def bearing_capacity(L2):
    """返回 (Mu_total, M_skirt_increment)"""
    D1,L1,t1,R2 = P['D1'],P['L1'],P['t1'],P['R2_fixed']
    t2,m,Kv,g,fc = P['t2'],P['m'],P['Kv'],P['gamma'],P['fc']
    z0 = 0.6*L1
    C1 = math.pi/24*m*z0**3*theta
    C2 = math.pi/48*m*fc*z0**3*theta
    C3 = math.pi/128*Kv*theta

    # ---- 主筒常量 (M1–M7) ----
    M1 = C1*D1*(L1 - z0/2)
    M2 = g*K0*D1*L1**3/6
    M3 = C2*D1**2
    M4 = -g*Ka*D1*L1**3/6
    M5 = 0.25*g*K0*fc*L1**2*D1**2
    M6 = 0.25*g*Ka*fc*L1**2*D1**2
    M7 = 0.5*(g*L1*Nq_th + 0.5*g*t1*Ny_th)*t1*D1**2
    M_main = M1+M2+M3+M4+M5+M6+M7

    # ---- 顶板常数 M8 ---- (R2 固定 → Dsk 固定)
    Dsk = D1 + 2*R2
    M8  = C3*D1**4           # 视为主筒常数，不进增量
    
    # ---- 裙边增量 (随 L2) ----
    if R2 <= 1e-9:
        M_sk = 0.0
    else:
        mθ = math.pi*m*theta
        M9  = mθ*Dsk*( L2**4/16 - (L1+z0)*L2**3/12 + L1*z0*L2**2/8 ) \
              + g*K0*Dsk*( L1*L2**2/2 - L2**3/3 )
        M10 = mθ*fc*Dsk**2*(-L2**3/24 + z0*L2**2/16 ) \
              + 0.25*g*K0*fc*Dsk**2*L2**2
        M11 = -g*Ka*Dsk*( L1*L2**2/2 - L2**3/3 )
        M12 = 0.25*g*Ka*fc*L2**2*Dsk**2
        Nq2 = 73.9*(1+7.25*(L2/L1)**1.16*(L1/D1)**(-2.09))
        Ny2 = 1.8*(Nq2-1)*0.9
        M7s = 0.5*(g*L2*Nq2 + 0.5*g*t2*Ny2)*t2*Dsk**2
        ΔM8 = C3*Dsk**4 - C3*D1**4  # ΔM8 = M8(Dsk) - M8(D1)
        M_sk = M9+M10+M11+M12+M7s+ΔM8        # 不含 ΔM8
    # ---- 总矩 & Mu ----
    M_total = M_main + M8 + M_sk
    Mu_tot  = (P['e_ecc']*D1)/(P['e_ecc']*D1+L1) * M_total
    Mu_sk   = (P['e_ecc']*D1)/(P['e_ecc']*D1+L1) * M_sk
    return Mu_tot, Mu_sk

def penetration_resistance(L2):
    """返回 (R_total, R_skirt_increment)"""
    D1,L1,t1,R2 = P['D1'],P['L1'],P['t1'],P['R2_fixed']
    t2,g = P['t2'],P['gamma']
    Ktan1 = (1-math.sin(phi))*math.tan(phi)
    R_main = seg_resistance(L1, D1, D1+2*t1, t1, g, Nq_th, Ny_th, Ktan1)
    if L2 <= 1e-9:
        return R_main, 0.0
    Dsi = D1 + 2*R2
    Dso = Dsi + 2*t2
    Ktan2 = 0.478*(1+0.746*(L2/L1)*(L1/D1)**(-1.32))
    Nq2 = 73.9*(1+7.25*(L2/L1)**1.16*(L1/D1)**(-2.09))
    Ny2 = 1.8*(Nq2-1)*0.9
    R_sk = seg_resistance(L2, Dsi, Dso, t2, g, Nq2, Ny2, Ktan2)
    return R_main+R_sk, R_sk

def material_volume(L2):
    """返回 (V_total, V_skirt_increment)"""
    D1,L1,t1,R2 = P['D1'],P['L1'],P['t1'],P['R2_fixed']
    t2,t3       = P['t2'],P['t3']
    V_main = math.pi*(D1+t1)*t1*L1
    D_out = D1 + 2*R2 + 2*t2
    D_in  = D1 + 2*t1
    V_lid = math.pi/4*(D_out**2 - D_in**2)*t3
    V_sk  = math.pi*(D1+2*R2+t2)*t2*L2
    return V_main+V_lid+V_sk, V_lid+V_sk

# ------------------------------------------------------------------
# 5. BASELINE & SWEEP η
# ------------------------------------------------------------------
L2_0 = max(P['L2_base'], 0.01*P['L1'])     # 确保基准非零
Mu0 , Msk0  = bearing_capacity(L2_0)
Rf0 , Rsk0  = penetration_resistance(L2_0)
Vs0 , Vsk0  = material_volume(L2_0)

eta = np.linspace(0, P['eta_max'], P['pts'])
Mu,Rf,Vs,Mus,Rfs,Vss = [],[],[],[],[],[]
for e in eta:
    L2 = e * L2_0
    mu,ms = bearing_capacity(L2);        Mu.append(mu);   Mus.append(ms)
    rf,rs = penetration_resistance(L2);  Rf.append(rf);   Rfs.append(rs)
    vs,vss= material_volume(L2);         Vs.append(vs);   Vss.append(vss)
Mu,Rf,Vs,Mus,Rfs,Vss = map(np.array,(Mu,Rf,Vs,Mus,Rfs,Vss))

rat = dict(
    Mu   = Mu / Mu0,
    R    = Rf / Rf0,
    V    = Vs / Vs0,
    Mu_s = Mus / Msk0,
    R_s  = Rfs / Rsk0,
    V_s  = Vss / Vsk0
)

# ------------------------------------------------------------------
# 6. PLOT  2×3 CURVES
# ------------------------------------------------------------------
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5),sharey=True)
ax1.plot(eta,rat['Mu'],'-o',label='Mu/Mu0')
ax1.plot(eta,rat['R' ],'-s',label='Rf/Rf0')
ax1.plot(eta,rat['V' ],'-^',label='Vs/Vs0')
ax1.axvline(1,ls='--',c='gray'); ax1.axhline(1,ls=':',c='gray')
ax1.set_title('Overall'); ax1.set_xlabel('η = L2 / L20'); ax1.set_ylabel('Normalized'); ax1.legend(); ax1.grid()

ax2.plot(eta,rat['Mu_s'],'--o',label='Msk/Msk0')
ax2.plot(eta,rat['R_s' ],'--s',label='Rsk/Rsk0')
ax2.plot(eta,rat['V_s' ],'--^',label='Vsk/Vsk0')
ax2.axvline(1,ls='--',c='gray'); ax2.axhline(1,ls=':',c='gray')
ax2.set_title('Skirt + Lid Increment'); ax2.set_xlabel('η = L2 / L20'); ax2.legend(); ax2.grid()

plt.tight_layout(); plt.show()

# ------------------------------------------------------------------
# 7. EXPORT
# ------------------------------------------------------------------
if P['export']:
    save_path = r'C:\Users\Laptop\WPSDrive\248006156\WPS云盘\1-吸力筒事务-第二篇相关材料\1-数据\4-协同优化部分\SSC_L2_sweep_corrected.xlsx'
    df = pd.DataFrame({'eta': eta, **rat})
    df.to_excel(save_path, index=False)
    print(f'Excel saved ➜ {save_path}')
