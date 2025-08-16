import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _calculate_component_resistance(h, D_i, D_o, t, gamma_prime, K_tan_delta, Nq, Ny):
    """
    内部辅助函数，用于计算单个圆筒结构（主筒或裙边）的贯入阻力。
    使用 李大勇等(2011) 模型。
    """
    if h <= 1e-6:
        return 0

    # 计算特征长度 Z
    Z_o = D_o / (4 * K_tan_delta)
    Z_i = D_i / (4 * K_tan_delta)
    
    # 外侧摩擦力 Fo
    exp_term_o = np.exp(h / Z_o)
    F_o = gamma_prime * Z_o**2 * (exp_term_o - 1 - h / Z_o) * K_tan_delta * (np.pi * D_o)
    
    # 内侧摩擦力 Fi
    exp_term_i = np.exp(h / Z_i)
    F_i = gamma_prime * Z_i**2 * (exp_term_i - 1 - h / Z_i) * K_tan_delta * (np.pi * D_i)
    
    # 端承力 Q_tip
    D_avg = D_i + t
    A_tip = np.pi * D_avg * t
    sigma_vo_h = gamma_prime * Z_o * (exp_term_o - 1)
    # 此处更正：原文公式sigma_end为sigma_vo*Nq + 1/2*gamma*t*Ny，但您图片中的公式为-gamma*t*Ny，此处遵循您的上一张图片公式
    sigma_end = sigma_vo_h * Nq + 0.5* gamma_prime * Ny * t 
    sigma_end = max(0, sigma_end)
    Q_tip = sigma_end * A_tip
    
    return F_o + F_i + Q_tip

def calculate_ssc_resistance(L1, D1, t1, L2, D2, t2, gamma_prime, phi_deg, K, delta_deg=None, export_to_excel=False, filename='try1_ssc_resistance.xlsx'):
    """
    计算裙式吸力筒（SSC）在静压贯入过程中的总阻力。

    参数:
    L1, D1, t1: 主筒的总长度(m), 内径(m), 壁厚(m).
    L2, D2, t2: 裙边的长度(m), 与主筒的径向宽度(m), 壁厚(m).
    gamma_prime, phi_deg, K, delta_deg: 土体和界面参数.
    export_to_excel, filename: Excel导出选项.
    
    返回:
    tuple: (depths, resistances_kN) 贯入深度(m)和对应总阻力(kN)的Numpy数组.
    """
    # --- 1. 参数初始化和单位转换 ---
    if delta_deg is None:
        delta_deg = phi_deg
    
    phi_rad = np.deg2rad(phi_deg)
    delta_rad = np.deg2rad(delta_deg)
    K_tan_delta_1 = 0.478107705798925
    K_tan_delta_2 = 0.478*(1+0.746*(L2/L1)*(L1/D1)**(-1.32))
    # --- 2. 自动计算承载力系数 Nq, Ny ---
    Nq_1 = 73.9
    Ny_1 = 114.06
    
    Nq_2 = 73.9*(1+7.25*(L2/L1)**1.16*(L1/D1)**(-2.09))
    Ny_2 = 1.8 * (Nq_2 - 1) * 0.9
    
    # --- 3. 定义几何尺寸 ---
    D1_outer = D1 + 2 * t1
    D_skirt_inner = D1 + 2 * D2
    D_skirt_outer = D_skirt_inner + 2 * t2
    
    # --- 4. 分阶段循环计算总阻力 ---
    depths = np.linspace(0, L1, 201)
    resistances_N = []
    h_skirt_start = L1 - L2

    for h in depths:
        # 计算主筒阻力
        res_main = _calculate_component_resistance(h, D1, D1_outer, t1, gamma_prime, K_tan_delta_1, Nq_1, Ny_1)
        
        # 判断裙边是否开始贯入
        if h <= h_skirt_start:
            # 阶段一：只有主筒阻力
            total_resistance = res_main
        else:
            # 阶段二：主筒阻力 + 裙边阻力
            h_skirt = h - h_skirt_start
            res_skirt = _calculate_component_resistance(h_skirt, D_skirt_inner, D_skirt_outer, t2, gamma_prime, K_tan_delta_2, Nq_2, Ny_2)
            total_resistance = res_main + res_skirt
            
        resistances_N.append(total_resistance)
        
    resistances_kN = np.array(resistances_N) / 1000

    # --- 导出到Excel ---
    if export_to_excel:
        try:
            df = pd.DataFrame({
                'Penetration Depth (m)': depths,
                'Total Resistance (kN)': resistances_kN
            })
            print(f"正在写入Excel文件到: {filename}")
            df.to_excel(filename, index=False, float_format="%.4f")
            print(f"数据已成功导出至文件: '{filename}'")
        except Exception as e:
            print(f"导出Excel文件失败: {e}")

    return depths, resistances_kN

# --- 使用示例 ---
if __name__ == '__main__':
    # --- 基础输入参数 ---
    # 以 SSC 工况 II-2-S (L1=240, D2=30, L2=60) 为例
    ssc_params = {
        "L1": 0.180, "D1": 0.120, "t1": 0.002,
        "L2": 0.000, "D2": 0.000, "t2": 0.002
    }
    
    soil_params = {
        "gamma_prime": 7850, "phi_deg": 50, "K": 0.55
    }
    
    # --- 调用函数计算 ---
    depths, resistances_kN = calculate_ssc_resistance(
        **ssc_params, 
        **soil_params,
        export_to_excel=True,
        filename = r'd:/工作汇总/1-吸力筒事务-第二篇相关材料/code/SSC_Resistance_L1_240_L2_60_D2_30-Limetod.xlsx'
    )
    
    # --- 打印和绘图 ---
    print(f"\n--- SSC 计算结果摘要 ---")
    print(f"计算工况: L1={ssc_params['L1']*1000:.0f}mm, L2={ssc_params['L2']*1000:.0f}mm, D2={ssc_params['D2']*1000:.0f}mm")
    final_resistance = resistances_kN[-1]
    experimental_resistance = 1.000 # 工况II-2-S的试验值
    print(f"在最大深度 {ssc_params['L1']:.3f} m 处，计算得到的总贯入阻力为: {final_resistance:.3f} kN")
    print(f"该工况试验值为: {experimental_resistance:.3f} kN")
    print(f"模型偏差: {(final_resistance - experimental_resistance) / experimental_resistance:.1%}")

    plt.figure(figsize=(10, 7))
    plt.plot(resistances_kN, depths, 'b-', label=f'Calculated SSC Resistance')
    # 标记裙边开始作用的点
    h_skirt_starts_at = ssc_params["L1"] - ssc_params["L2"]
    resistance_at_skirt_start = np.interp(h_skirt_starts_at, depths, resistances_kN)
    plt.scatter([resistance_at_skirt_start], [h_skirt_starts_at], color='orange', s=100, zorder=5, label='Skirt Penetration Starts')
    plt.axhline(y=h_skirt_starts_at, color='orange', linestyle='--', alpha=0.7)
    
    # 对比试验最终值
    plt.scatter([experimental_resistance], [ssc_params['L1']], color='red', s=100, zorder=5, label='Experimental Final Resistance')

    plt.gca().invert_yaxis() 
    plt.xlabel('Penetration Resistance (kN)')
    plt.ylabel('Penetration Depth (m)')
    plt.title('SSC Static Penetration Resistance vs. Depth')
    plt.grid(True)
    plt.legend()
    plt.show()