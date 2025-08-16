import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------
# 1. 参数输入区 (Parameter Input Section)
#    您可以在此修改所有工况参数
# --------------------------------------------------------------------------
params = {
    # --- 几何参数 (m) ---
    "D1": 0.120,       # 主筒内径
    "L1": 0.240,       # 主筒长度 / 最终贯入深度
    "t1": 0.002,       # 主筒壁厚
    "L2": 0.060,       # 裙边长度 (固定值)
    "t2": 0.002,       # 裙边壁厚 (固定值)
    "t3": 0.010,       # 盖板厚度 (固定值)

    # --- 土体与界面参数 ---
    "phi_deg": 38.6,   # 内摩擦角 (度)
    "gamma_kN": 7.85,  # 土的有效重度 (kN/m^3)
    "fc": 0.18,        # 承载力模型中的筒-土摩擦系数

    # --- 承载力模型参数 (Winkler) ---
    "m_winkler": 60e3,
    "Kv_winkler": 10.053e3,
    "e_ecc": 1.5,
    "theta_deg": 1.5,

    # --- 归一化分析控制 ---
    "R2_0": 0.030,     # 基准裙边宽度 (m), 此为 λ=1 的点
    "lambda_max": 2.0,
    "num_points": 100,
}

# --------------------------------------------------------------------------
# 2. 核心计算函数 (Core Calculation Functions)
#    (函数已更新，会同时返回“总值”和“增量值”)
# --------------------------------------------------------------------------

# --- 2.1 承载力 Mu 计算 ---
def calculate_bearing_capacity(R2, p):
    """
    计算极限承载力矩。
    返回: (Mu_total, Mu_increment)
          Mu_total: 总承载力矩
          Mu_increment: 裙边贡献的承载力矩增量
    """
    phi_rad = math.radians(p['phi_deg'])
    K0 = 1 - math.sin(phi_rad)
    Ka = math.tan(math.pi / 4 - phi_rad / 2)**2
    Nq = math.exp(math.pi * math.tan(phi_rad)) * math.tan(math.pi / 4 + phi_rad / 2)**2
    Ng = 1.5 * (Nq - 1) * math.tan(phi_rad)

    z0 = 0.6 * p['L1']
    C1 = math.pi / 24 * p['m_winkler'] * z0**3 * math.radians(p['theta_deg'])
    C2 = math.pi / 48 * p['m_winkler'] * p['fc'] * z0**3 * math.radians(p['theta_deg'])
    C3 = math.pi / 128 * p['Kv_winkler'] * math.radians(p['theta_deg'])
    
    M_main_const = (C1 * p['D1'] * (p['L1'] - z0 / 2) +
                    1/6 * p['gamma_kN'] * K0 * p['D1'] * p['L1']**3 +
                    C2 * p['D1']**2 -
                    1/6 * p['gamma_kN'] * Ka * p['D1'] * p['L1']**3 +
                    0.25 * K0 * p['gamma_kN'] * p['fc'] * p['L1']**2 * p['D1']**2 +
                    0.25 * Ka * p['gamma_kN'] * p['fc'] * p['L1']**2 * p['D1']**2 +
                    0.5 * (p['gamma_kN'] * p['L1'] * Nq + 0.5 * p['gamma_kN'] * p['t1'] * Ng) * p['t1'] * p['D1']**2)

    M8_no_skirt = C3 * p['D1']**4
    M0_no_skirt = M_main_const + M8_no_skirt
    
    if R2 < 1e-6:
        M0_increment = 0
        M0_total = M0_no_skirt
    else:
        Dsk = p['D1'] + 2 * R2
        M8_with_skirt = C3 * Dsk**4
        m_term = math.pi * p['m_winkler'] * math.radians(p['theta_deg'])
        M9 = (m_term*Dsk*(p['L2']**4/16 - (p['L1']+z0)*p['L2']**3/12 + p['L1']*z0*p['L2']**2/8) + 
              p['gamma_kN']*K0*Dsk*(p['L1']*p['L2']**2/2 - p['L2']**3/3))
        M10 = (m_term*p['fc']*Dsk**2*(-p['L2']**3/24 + z0*p['L2']**2/16) + 
               0.25*p['gamma_kN']*K0*p['fc']*Dsk**2*p['L2']**2)
        M11 = -p['gamma_kN']*Ka*Dsk*(p['L1']*p['L2']**2/2 - p['L2']**3/3)
        M12 = 0.25*p['gamma_kN']*Ka*p['fc']*p['L2']**2*Dsk**2
        pu2 = p['gamma_kN'] * p['L2'] * Nq + 0.5 * p['gamma_kN'] * p['t2'] * Ng
        M7s = 0.5 * pu2 * p['t2'] * Dsk**2
        
        M_skirt_moment_sum = M9 + M10 + M11 + M12 + M7s
        M_lid_increment = M8_with_skirt - M8_no_skirt
        M0_increment = M_lid_increment + M_skirt_moment_sum
        M0_total = M0_no_skirt + M0_increment

    # 使用转换系数将M0转换为最终的Mu
    conversion_factor = (p['e_ecc'] * p['D1']) / (p['e_ecc'] * p['D1'] + p['L1'])
    Mu_total = M0_total * conversion_factor
    Mu_increment = M0_increment * conversion_factor
    
    return Mu_total, Mu_increment

# --- 2.2 贯入阻力 R 计算 ---
def _calc_segment_resistance(h, Di, Do, t, p, Nq, Ny, K_tan_delta):
    # ... (此内部函数无需修改) ...
    if h <= 1e-9: return 0.0
    Zo = Do / (4 * K_tan_delta); Zi = Di / (4 * K_tan_delta)
    Fo = p['gamma_kN'] * Zo**2 * (math.exp(h / Zo) - 1 - h / Zo) * K_tan_delta * math.pi * Do
    Fi = p['gamma_kN'] * Zi**2 * (math.exp(h / Zi) - 1 - h / Zi) * K_tan_delta * math.pi * Zi
    Atip = math.pi * (Di + t) * t
    sigma_v = p['gamma_kN'] * Zo * (math.exp(h / Zo) - 1)
    sigma_end = max(0, sigma_v * Nq + 0.5 * p['gamma_kN'] * t * Ny)
    Qtip = sigma_end * Atip
    return Fo + Fi + Qtip

def calculate_penetration_resistance(R2, p):
    """
    计算总贯入阻力。
    返回: (R_total, R_increment)
          R_total: 总阻力
          R_increment: 裙壁贡献的阻力增量
    """
    phi_rad = math.radians(p['phi_deg'])
    K_th = 1 - math.sin(phi_rad)
    K_tan_delta_th = K_th * math.tan(phi_rad)
    Nq_th = math.exp(math.pi*math.tan(phi_rad)) * math.tan(math.pi/4 + phi_rad/2)**2
    Ny_th = 1.8 * (Nq_th - 1) * math.tan(phi_rad)
    
    R_main = _calc_segment_resistance(p['L1'], p['D1'], p['D1'] + 2 * p['t1'], p['t1'], p, Nq_th, Ny_th, K_tan_delta_th)
    
    if R2 < 1e-6:
        R_increment = 0
    else:
        K_tan_delta_em = 0.478*(1 + 0.746*(p['L2']/p['L1'])*(p['L1']/p['D1'])**(-1.32))
        Nq_em = 73.9 * (1 + 7.25*(p['L2']/p['L1'])**1.16*(p['L1']/p['D1'])**(-2.09))
        Ny_em = 1.8 * (Nq_em - 1) * 0.9
        D_skirt_inner = p['D1'] + 2 * R2
        D_skirt_outer = D_skirt_inner + 2 * p['t2']
        R_increment = _calc_segment_resistance(p['L2'], D_skirt_inner, D_skirt_outer, p['t2'], p, Nq_em, Ny_em, K_tan_delta_em)
        
    R_total = R_main + R_increment
    return R_total, R_increment

# --- 2.3 材料用量 V 计算 ---
def calculate_material_volume(R2, p):
    """
    计算总材料用量。
    返回: (V_total, V_increment)
          V_total: 总体积
          V_increment: 裙边及相关盖板的体积增量
    """
    V_main = math.pi * (p['D1'] + p['t1']) * p['t1'] * p['L1']
    
    if R2 < 1e-6:
        V_lid_no_skirt = (math.pi / 4) * (p['D1'] + 2*p['t1'])**2 * p['t3']
        V_total = V_main + V_lid_no_skirt
        V_increment = 0
    else:
        V_skirt = math.pi * (p['D1'] + 2 * R2 + p['t2']) * p['t2'] * p['L2']
        D_lid_outer = p['D1'] + 2 * R2 + 2 * p['t2']
        D_lid_inner = p['D1'] + 2 * p['t1']
        V_lid = (math.pi / 4) * (D_lid_outer**2 - D_lid_inner**2) * p['t3']
        V_increment = V_skirt + V_lid
        V_total = V_main + V_increment
        
    return V_total, V_increment

# --------------------------------------------------------------------------
# 3. 归一化分析与可视化 (Normalization Analysis and Visualization)
# --------------------------------------------------------------------------
# 3.1 定义分析范围
lambda_min = (params['t1'] / params['R2_0']) * 1.01
lambda_range = np.linspace(lambda_min, params['lambda_max'], params['num_points'])

# 3.2 计算基准值 (λ=1)
R2_base = params['R2_0']
Mu_total_0, Mu_increment_0 = calculate_bearing_capacity(R2_base, params)
R_total_0, R_increment_0 = calculate_penetration_resistance(R2_base, params)
V_total_0, V_increment_0 = calculate_material_volume(R2_base, params)

# 3.3 循环计算并归一化
# 初始化6个列表来存储比率
total_mu_ratios, total_r_ratios, total_v_ratios = [], [], []
inc_mu_ratios, inc_r_ratios, inc_v_ratios = [], [], []

for lam in lambda_range:
    r2_current = lam * R2_base
    
    mu_total, mu_inc = calculate_bearing_capacity(r2_current, params)
    r_total, r_inc = calculate_penetration_resistance(r2_current, params)
    v_total, v_inc = calculate_material_volume(r2_current, params)
    
    # 计算总性能比率
    total_mu_ratios.append(mu_total / Mu_total_0)
    total_r_ratios.append(r_total / R_total_0)
    total_v_ratios.append(v_total / V_total_0)
    
    # 计算增量性能比率 (防止除以0)
    inc_mu_ratios.append(mu_inc / Mu_increment_0 if Mu_increment_0 else 0)
    inc_r_ratios.append(r_inc / R_increment_0 if R_increment_0 else 0)
    inc_v_ratios.append(v_inc / V_increment_0 if V_increment_0 else 0)

# 3.4 打印分析摘要
print("--- 分析完成 ---")
print(f"基准裙宽 R2_0 = {R2_base*1000:.1f} mm (对应 λ = 1.0)")
print("\n--- 基准点总性能 ---")
print(f"总承载力 Mu_total_0 = {Mu_total_0:.2f} kN·m")
print(f"总贯入阻力 R_total_0 = {R_total_0:.2f} kN")
print(f"总材料用量 V_total_0 = {V_total_0:.4f} m^3")
print("\n--- 基准点增量性能 ---")
print(f"承载力增量 Mu_increment_0 = {Mu_increment_0:.2f} kN·m")
print(f"阻力增量 R_increment_0 = {R_increment_0:.2f} kN")
print(f"材料增量 V_increment_0 = {V_increment_0:.4f} m^3")


# 3.5 结果可视化 (使用子图)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), sharex=True)

# --- Top Plot: Total Performance Analysis ---
ax1.plot(lambda_range, total_mu_ratios, 'b-o', markersize=4, label='总承载力比率 ($M_u / M_{u0}$)')
ax1.plot(lambda_range, total_r_ratios, 'r-s', markersize=4, label='总贯入阻力比率 ($R / R_{0}$)')
ax1.plot(lambda_range, total_v_ratios, 'g-^', markersize=4, label='总材料用量比率 ($V / V_{0}$)')
ax1.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5, label='基准设计 (λ=1.0)')
ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.0)
ax1.set_title('宏观分析：吸力筒总性能随裙边宽度的变化', fontsize=16)
ax1.set_ylabel('总性能比率', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True)

# --- Bottom Plot: Incremental Performance Analysis ---
ax2.plot(lambda_range, inc_mu_ratios, 'b-o', markersize=4, label='承载力增量比率 ($M_{inc} / M_{inc,0}$)')
ax2.plot(lambda_range, inc_r_ratios, 'r-s', markersize=4, label='阻力增量比率 ($R_{inc} / R_{inc,0}$)')
ax2.plot(lambda_range, inc_v_ratios, 'g-^', markersize=4, label='材料增量比率 ($V_{inc} / V_{inc,0}$)')
ax2.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5, label='基准设计 (λ=1.0)')
ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.0)
ax2.set_title('微观分析：裙边贡献增量随其宽度变化的敏感度', fontsize=16)
ax2.set_xlabel('裙边宽度变化倍数 $\lambda = R_2 / R_{2,0}$', fontsize=12)
ax2.set_ylabel('增量性能比率', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.98]) # 调整布局防止标题重叠
fig.suptitle('裙式吸力筒优化分析：宏观与微观视角', fontsize=20, y=0.99)
plt.show()

# --------------------------------------------------------------------------
# 4. 保存分析结果到CSV文件
# --------------------------------------------------------------------------
output_df = pd.DataFrame({
    'lambda': lambda_range,
    'total_mu_ratios': total_mu_ratios,
    'total_r_ratios': total_r_ratios,
    'total_v_ratios': total_v_ratios,
    'inc_mu_ratios': inc_mu_ratios,
    'inc_r_ratios': inc_r_ratios,
    'inc_v_ratios': inc_v_ratios
})
output_path = r'D:\工作汇总\1-吸力筒事务-第二篇相关材料\code\right_SSC_D.csv'
output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"分析结果已保存到: {output_path}")