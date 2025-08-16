#这个是用来建立公式的，已经是反算成功的，带入了计算公式。
import numpy as np
import pandas as pd

# ---------------- 1. 单筒段阻力模型 ----------------
def comp_res(h, D_i, D_o, t, γp, β, Nq, Ny):
    """李大勇等(2011) 模型——单筒段摩阻 + 端阻"""
    if h <= 1e-6:
        return 0.0
    Z_o = D_o / (4 * β)
    Z_i = D_i / (4 * β)
    exp_o, exp_i = np.exp(h/Z_o), np.exp(h/Z_i)
    F_o = γp * Z_o**2 * (exp_o - 1 - h/Z_o) * β * np.pi * D_o
    F_i = γp * Z_i**2 * (exp_i - 1 - h/Z_i) * β * np.pi * D_i
    σ_vo = γp * Z_o * (exp_o - 1)
    A_tip = np.pi * (D_i + t) * t
    Q_tip = (σ_vo * Nq + 0.5 * γp * Ny * t) * A_tip
    return F_o + F_i + Q_tip

# -------------- 2. 方法一：输入 φ, K  ----------------
def calc_by_phi(L1, D1, t1, L2, D2, t2, γp, phi_deg, K):
    phi = np.deg2rad(phi_deg)
    β_1 = K * np.tan(phi)              # K·tanδ, δ≈φ
    Nq_main, Ny_main = 73.9, 114.06
    Nq_skirt = np.exp(np.pi*np.tan(phi)) * np.tan(np.deg2rad(45)+phi/2)**2
    Ny_skirt = 1.8*(Nq_skirt-1)*np.tan(phi)
    # 几何
    D1_o = D1 + 2*t1
    D2_i = D1 + 2*D2
    D2_o = D2_i + 2*t2
    h0 = L1 - L2
    depths, R = np.linspace(0, L1, 201), []
    for h in depths:
        Rm = comp_res(h, D1, D1_o, t1, γp, β_1, Nq_main, Ny_main)
        if h>h0 and L2>0:
            Rm += comp_res(h-h0, D2_i, D2_o, t2, γp, β_1, Nq_skirt, Ny_skirt)
        R.append(Rm)
    return np.array(R)/1000  # kN

# -------------- 3. 方法二：输入 β, Nq  ----------------
def calc_by_nq_beta(L1, D1, t1, L2, D2, t2, γp, β, phi_deg):
    """
    MSC裙边模型分阶段计算：主筒用固定Nq/Ny，裙边用phi计算Nq/Ny。
    """
    phi = np.deg2rad(phi_deg)
    # 主筒参数
    Nq_main, Ny_main = 73.9, 114.06
    # 裙边参数（用phi公式）
    Nq_skirt = np.exp(np.pi * np.tan(phi)) * np.tan(np.deg2rad(45) + phi / 2) ** 2
    Ny_skirt = 1.8 * (Nq_skirt - 1) * np.tan(phi)
    # 几何
    D1_o = D1 + 2 * t1
    D2_i = D1 + 2 * D2
    D2_o = D2_i + 2 * t2
    h0 = L1 - L2
    depths, R = np.linspace(0, L1, 201), []
    for h in depths:
        Rm = comp_res(h, D1, D1_o, t1, γp, β, Nq_main, Ny_main)
        if h > h0 and L2 > 0:
            h_skirt = h - h0
            Rm += comp_res(h_skirt, D2_i, D2_o, t2, γp, β, Nq_skirt, Ny_skirt)
        R.append(Rm)
    return np.array(R) / 1000  # kN

# ---------------- 4. 批量计算对比 ----------------
cases = [
    # id, L1(mm), L2(mm), D2(mm), meas kN
    (1, 240, 30, 30, 1.704),
    (2, 240, 60, 30, 1.987),
    (3, 240, 90, 30, 2.306),
    (4, 240, 30, 50, 1.845),
    (5, 240, 60, 50, 2.117),
    (6, 240, 90, 50, 2.430),
    (7, 180, 60, 30, 1.006),
    (8, 120, 60, 30, 0.626),
    (9, 240, 0, 0, 1.104),
    (10, 180, 0, 0, 0.55),
    (11, 120, 0, 0, 0.187),
]

γp, D1, t1, t2 = 7850, 0.120, 0.002, 0.002
phi_deg, K = 41, 0.55

results = []
for cid, L1mm, L2mm, D2mm, meas in cases:
    L1, L2, D2 = L1mm/1000, L2mm/1000, D2mm/1000
    # 方法一
    R1 = calc_by_phi(L1,D1,t1,L2,D2,t2,γp,phi_deg,K)[-1]
    # 方法二：先计算 Nq_skirt, β
    L_D = L1/D1; L2_L1 = L2/L1 if L2>0 else 0
    Nq_sk = 73.9*(1+7.25*(L2_L1)**1.16*(L_D)**(-2.09))
    β_sk  = 0.478*(1+0.746*(L2_L1)*(L_D)**(-1.32))
    R2 = calc_by_nq_beta(L1,D1,t1,L2,D2,t2,γp,β_sk,phi_deg)[-1]
    # 保留三位有效数字
    R2 = float(f"{R2:.3g}")
    err1 = round((R1-meas)/meas*100,1)
    err2 = float(f"{(R2-meas)/meas*100:.3g}")
    results.append((cid, meas, round(R1,3), R2, err1, err2, float(f"{Nq_sk:.3g}"), float(f"{β_sk:.3g}")))

df = pd.DataFrame(results,
    columns=["Case","Measured","Method1 (kN)","Method2 (kN)",
             "Err1 (%)","Err2 (%)","Nq_sk","β_sk"])
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)     # 显示所有行
pd.set_option('display.width', 200)         # 设置输出宽度
print("方法一 vs 方法二 预测对比")
print(df)

