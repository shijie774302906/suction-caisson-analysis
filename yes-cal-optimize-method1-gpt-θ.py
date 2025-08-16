# -*- coding: utf-8 -*-
"""
SSC θ_r 参数化（R2, L2 固定）
整体 & Skirt+Lid 增量 6 条归一化曲线
"""

import math, numpy as np, pandas as pd, matplotlib.pyplot as plt

# ---------------- 1. INPUTS --------------------
P = dict(
    # Geometry (m)
    D1=0.120,  L1=0.240,  t1=0.002,
    R2=0.030,  L2=0.060,  t2=0.002, t3=0.010,

    # Soil & Winkler
    phi_deg=38.6,  gamma=7.85,  fc=0.18,
    m=60e3,  Kv=1.0053e4,  theta0_deg=1.5,  e_ecc=1.5,

    # θ_r sweep (deg)  —— 基准 π = 180°
    theta_r_deg=np.linspace(0, 180, 91),

    export=True
)

# ---------------- 2. CONSTANTS -----------------
phi = math.radians(P['phi_deg'])
theta0 = math.radians(P['theta0_deg'])

K0 = 1 - math.sin(phi)
Ka = math.tan(math.pi/4 - phi/2)**2
Nq_th = math.exp(math.pi*math.tan(phi))*math.tan(math.pi/4+phi/2)**2
Ny_th = 1.5*(Nq_th-1)*math.tan(phi)

D1,L1,t1 = P['D1'],P['L1'],P['t1']
R2,L2,t2 = P['R2'],P['L2'],P['t2']
Dsk = D1 + 2*R2
z0  = 0.6*L1

C1 = math.pi/24 * P['m'] * z0**3 * theta0
C2 = math.pi/48 * P['m'] * P['fc'] * z0**3 * theta0
C3 = math.pi/128 * P['Kv'] * theta0

# 主筒常量力矩
M1=C1*D1*(L1-z0/2)
M2=P['gamma']*K0*D1*L1**3/6
M3=C2*D1**2
M4=-P['gamma']*Ka*D1*L1**3/6
M5=0.25*P['gamma']*K0*P['fc']*L1**2*D1**2
M6=0.25*P['gamma']*Ka*P['fc']*L1**2*D1**2
M7=0.5*(P['gamma']*L1*Nq_th + 0.5*P['gamma']*t1*Ny_th)*t1*D1**2
M_main = M1+M2+M3+M4+M5+M6+M7
M8_const = C3*D1**4

# 土参数固定后 Nq2, Ny2, Ktan2
Nq2 = 73.9*(1+7.25*(L2/L1)**1.16*(L1/D1)**(-2.09))
Ny2 = 1.8*(Nq2-1)*0.9
Ktan2 = 0.478*(1+0.746*(L2/L1)*(L1/D1)**(-1.32))

# 主筒贯入阻力常量
def seg_res(h, Di, Do, t, g, Nq, Ny, Ktan):
    if h<=1e-9: return 0.0
    Zo, Zi = Do/(4*Ktan), Di/(4*Ktan)
    Fo = g*Zo**2*(math.exp(h/Zo)-1-h/Zo)*Ktan*math.pi*Do
    Fi = g*Zi**2*(math.exp(h/Zi)-1-h/Zi)*Ktan*math.pi*Di
    sig = g*Zo*(math.exp(h/Zo)-1)
    Q   = max(0, sig*Nq + 0.5*g*t*Ny)*math.pi*(Di+t)*t
    return Fo+Fi+Q

Ktan1 = (1-math.sin(phi))*math.tan(phi)
R_main = seg_res(L1, D1, D1+2*t1, t1, P['gamma'], Nq_th, Ny_th, Ktan1)
R_sk_π = seg_res(L2, D1+2*R2, D1+2*R2+2*t2, t2, P['gamma'], Nq2, Ny2, Ktan2)  # θ_r = π

# 主筒 & 盖板常量体积
V_main = math.pi*(D1+t1)*t1*L1

# ---------------- 3. SWEEP θ_r ------------------
theta_list = np.asarray(P['theta_r_deg']) * math.pi/180
Mu_tot, Mu_sk, R_tot, R_sk, V_tot, V_sk = [],[],[],[],[],[]

for th in theta_list:
    sin_half = math.sin(th/2)
    thsin    = th + math.sin(th)

    # -- 裙板增量力矩 M_sk(th) --
    mθ = math.pi*P['m']*theta0
    M13 = (P['Kv']*theta0*(Dsk**4 - D1**4)/128) * thsin
    M9  = mθ*thsin*Dsk*(L2**4/16 - (L1+z0)*L2**3/12 + L1*z0*L2**2/8) \
          + P['gamma']*K0*Dsk*(L1*L2**2/2 - L2**3/3)*sin_half
    M10 = mθ*P['fc']*thsin*Dsk**2*(L2**3/24 + z0*L2**2/16) \
          + 0.25*P['gamma']*K0*P['fc']*Dsk**2*L2**2*sin_half
    M11 = -P['gamma']*Ka*Dsk*(L1*L2**2/2 - L2**3/3)*sin_half
    M12 = 0.25*P['gamma']*Ka*P['fc']*Dsk**2*L2**2*sin_half
    M7s = 0.5*(P['gamma']*L2*Nq2 + 0.5*P['gamma']*t2*Ny2)*t2*Dsk**2*sin_half
    M_sk = M13+M9+M10+M11+M12+M7s

    conv = P['e_ecc']*D1/(P['e_ecc']*D1+L1)
    Mu_sk.append(conv*M_sk)
    Mu_tot.append(conv*(M_main + M8_const + M_sk))

    # -- 贯入阻力 --
    rsk_val = (th/math.pi)*R_sk_π
    R_sk.append(rsk_val)
    R_tot.append(R_main + rsk_val)

    # -- 材料体积 --
    V_lid = (th/8.0) * ((D1+2*R2+2*t2)**2 - (D1+2*t1)**2) * P['t3']
    V_sk_val  = th * (D1+2*R2+t2) * t2 * L2
    V_total = V_main + V_lid + V_sk_val
    V_tot.append(V_total)
    V_sk.append(V_lid + V_sk_val)

Mu_tot, Mu_sk = map(np.array, (Mu_tot, Mu_sk))
R_tot, R_sk   = map(np.array, (R_tot,  R_sk))
V_tot, V_sk   = map(np.array, (V_tot,  V_sk))

# 基准 θ_r = π
base_idx = np.where(np.isclose(theta_list, math.pi))[0][0]
Mu0, Msk0 = Mu_tot[base_idx], Mu_sk[base_idx]
R0 , Rsk0 = R_tot[base_idx], R_sk [base_idx]
V0 , Vsk0 = V_tot[base_idx], V_sk [base_idx]

rat = dict(
    Mu = Mu_tot/Mu0,  R = R_tot/R0,  V = V_tot/V0,
    Mu_s = Mu_sk/Msk0, R_s = R_sk/Rsk0, V_s = V_sk/Vsk0
)

# ---------------- 4. PLOT -----------------------
xdeg = P['theta_r_deg']
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(13,5),sharey=True)

ax1.plot(xdeg,rat['Mu'],'-o',label='Mu/Muπ')
ax1.plot(xdeg,rat['R' ],'-s',label='Rf/Rfπ')
ax1.plot(xdeg,rat['V' ],'-^',label='Vs/Vsπ')
ax1.axvline(180,ls='--',c='gray'); ax1.axhline(1,ls=':',c='gray')
ax1.set_xlabel('θ_r (deg)'); ax1.set_ylabel('Normalized'); ax1.set_title('Overall')
ax1.legend(); ax1.grid()

ax2.plot(xdeg,rat['Mu_s'],'--o',label='Msk/Mskπ')
ax2.plot(xdeg,rat['R_s' ],'--s',label='Rsk/Rskπ')
ax2.plot(xdeg,rat['V_s' ],'--^',label='Vsk/Vskπ')
ax2.axvline(180,ls='--',c='gray'); ax2.axhline(1,ls=':',c='gray')
ax2.set_xlabel('θ_r (deg)'); ax2.set_title('Skirt + Lid Increment')
ax2.legend(); ax2.grid()

plt.tight_layout(); plt.show()

# ---------------- 5. EXPORT ---------------------
if P['export']:
    import os
    save_path = r'D:\工作汇总\1-吸力筒事务-第二篇相关材料\code\SSC_theta_r_sweep.xlsx'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame({'theta_r_deg': xdeg, **rat})
    df.to_excel(save_path, index=False)
    print(f'Excel saved ➜ {save_path}')
