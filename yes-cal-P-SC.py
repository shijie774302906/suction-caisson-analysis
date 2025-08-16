import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_sc_resistance_li_model_revised(L, D_inner, t, gamma_prime, phi_deg, K, delta_deg=None, export_to_excel=False, filename='penetration_resistance.xlsx'):
    """
    根据李大勇等(2011)模型计算SC的静压贯入阻力（修订版）。
    该版本输入更基础的物理参数，并能自动导出结果到Excel。

    参数:
    L (float): 筒的总长度 (m).
    D_inner (float): 筒的内径 (m).
    t (float): 筒壁厚度 (m).
    gamma_prime (float): 土的有效重度 (N/m^3).
    phi_deg (float): 土的有效内摩擦角 (度).
    K (float): 侧向土压力系数.
    delta_deg (float, optional): 筒-土界面摩擦角 (度). 如果为None, 默认等于phi_deg.
    export_to_excel (bool): 是否导出结果到Excel文件. 默认为False.
    filename (str): 导出的Excel文件名. 默认为'penetration_resistance.xlsx'.

    返回:
    tuple: (depths, resistances_kN) 贯入深度(m)和对应总阻力(kN)的Numpy数组.
    """
    # --- 1. 参数初始化和单位转换 ---
    if delta_deg is None:
        delta_deg = phi_deg  # 如果未指定筒土摩擦角，则假定其等于土的内摩擦角
    
    phi_rad = np.deg2rad(phi_deg)
    delta_rad = np.deg2rad(delta_deg)
    
    # 几何参数
    D_outer = D_inner + 2 * t
    D_avg = D_inner + t
    A_tip = np.pi * D_avg * t
    
    # 计算 K*tan(delta)
    K_tan_delta = K * np.tan(delta_rad)
    
    # --- 2. 自动计算承载力系数 Nq, Ny ---
    # 根据内摩擦角计算 Nq
    Nq = np.exp(np.pi * np.tan(phi_rad)) * np.tan(np.deg2rad(45) + phi_rad / 2)**2
    # 根据 Nq 计算 Ny
    Ny = 1.8 * (Nq - 1) * np.tan(phi_rad)
    
    print("--- 模型内部计算参数 ---")
    print(f"内摩擦角 φ' = {phi_deg}°")
    print(f"侧向土压力系数 K = {K:.3f}")
    print(f"计算得到的 Nq = {Nq:.2f}")
    print(f"计算得到的 Ny = {Ny:.2f}")
    print("------------------------")
    
    # --- 3. 计算特征长度 Z_o, Z_i ---
    Z_o = D_outer / (4 * K_tan_delta)
    Z_i = D_inner / (4 * K_tan_delta)
    
    # --- 4. 循环计算不同深度下的阻力 ---
    depths = np.linspace(0, L, 101) # 增加数据点密度
    resistances_N = []

    for h in depths:
        if h < 1e-6: # 避免深度为0时的计算问题
            resistances_N.append(0)
            continue
        
        # 计算外侧摩擦力 Fo
        exp_term_o = np.exp(h / Z_o)
        F_o = gamma_prime * Z_o**2 * (exp_term_o - 1 - h / Z_o) * K_tan_delta * (np.pi * D_outer)
        
        # 计算内侧摩擦力 Fi
        exp_term_i = np.exp(h / Z_i)
        F_i = gamma_prime * Z_i**2 * (exp_term_i - 1 - h / Z_i) * K_tan_delta * (np.pi * D_inner)
        
        # 计算端承力 Q_tip
        sigma_vo_h = gamma_prime * Z_o * (exp_term_o - 1)
        sigma_end = sigma_vo_h * Nq + 0.5*gamma_prime * Ny * t 
        sigma_end = max(0, sigma_end) 
        Q_tip = sigma_end * A_tip
        
        R_total_N = F_o + F_i + Q_tip
        resistances_N.append(R_total_N)
        
    resistances_N = np.array(resistances_N)
    resistances_kN = resistances_N / 1000
    
    # --- 5. 导出到Excel ---
    if export_to_excel:
        df = pd.DataFrame({
            'Penetration Depth (m)': depths,
            'Side Friction Outer (kN)': [((np.pi * D_outer * gamma_prime * K * np.tan(delta_rad) / 2) * h**2)/1000 for h in depths], # 简化展示，实际使用模型计算
            'Side Friction Inner (kN)': [((np.pi * D_inner * gamma_prime * K * np.tan(delta_rad) / 2) * h**2)/1000 for h in depths], # 简化展示，实际使用模型计算
            'Tip Resistance (kN)': (resistances_N - F_o - F_i)/1000, # 从总阻力反算
            'Total Resistance (kN)': resistances_kN
        })
        try:
            df.to_excel(filename, index=False, float_format="%.4f")
            print(f"\n数据已成功导出至文件: '{filename}'")
        except Exception as e:
            print(f"\n导出Excel文件失败: {e}")

    return depths, resistances_kN

# --- 使用示例 ---
if __name__ == '__main__':
    # --- 基础输入参数 ---
    # 以 SC 工况 I-3-S (L/D=2.0) 为例
    L_test = 0.240  # 贯入总深度 (m)
    D_inner_test = 0.120  # 内径 (m)
    t_test = 0.002 # 筒壁厚度 (m)
    
    # 土体和界面参数 (参考Li et al. 2011论文的工况)
    # 为使其 K*tan(delta) 结果与原文的0.48一致，进行反算
    # tan(40) ≈ 0.839, 故 K = 0.48 / 0.839 ≈ 0.572
    gamma_prime_test = 7850  # 土有效重度 (N/m^3)
    phi_deg_test = 41     # 内摩擦角 (度)
    K_test = 0.5           # 侧向土压力系数 (经推算)
    
    # --- 调用函数计算 ---
    # 设置 export_to_excel=True 来激活Excel导出功能
    depth_array, resistance_array_kN = calculate_sc_resistance_li_model_revised(
        L=L_test, 
        D_inner=D_inner_test, 
        t=t_test, 
        gamma_prime=gamma_prime_test, 
        phi_deg=phi_deg_test, 
        K=K_test,
        export_to_excel=True,
        filename=f'new-SC_LD_1_38_6.xlsx' # 指定有意义的文件名
    )
    
    # --- 打印最终结果 ---
    print(f"\n--- 计算结果摘要 ---")
    print(f"计算工况: L={L_test*1000:.0f}mm, D={D_inner_test*1000:.0f}mm (L/D={L_test/D_inner_test:.1f})")
    print(f"在最大深度 {L_test:.3f} m 处，计算得到的总贯入阻力为: {resistance_array_kN[-1]:.3f} kN")

    # --- 绘制贯入深度-阻力曲线 ---
    plt.figure(figsize=(8, 6))
    plt.plot(resistance_array_kN, depth_array, 'b-', label=f'Calculated Resistance (K={K_test:.3f})')
    # 绘制试验数据点进行对比
    plt.scatter([1.041], [0.240], color='red', s=100, zorder=5, label='Experimental Data (I-3-S)')
    
    plt.gca().invert_yaxis() 
    plt.xlabel('Penetration Resistance (kN)')
    plt.ylabel('Penetration Depth (m)')
    plt.title(f'SC Penetration Resistance (L/D = {L_test/D_inner_test:.1f}) - Li et al. Model')
    plt.grid(True)
    plt.legend()
    plt.show()