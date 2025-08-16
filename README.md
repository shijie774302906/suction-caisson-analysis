# Horizontal Suction Caisson (HSC) Analysis

This repository contains Python code for analyzing the horizontal capacity of suction caissons with different skirt configurations.

## Project Overview

This project implements a comprehensive analysis framework for calculating the horizontal bearing capacity of suction caissons under different configurations:

1. **Case 1: No Skirt** - Basic suction caisson without external skirt (z0 = 0.8×L1)
2. **Case 2: Full Skirt** - Complete skirt configuration with non-zero L2 and t2 (z0 = 0.55×L1)
3. **Case 3: Simplified Skirt** - Simplified skirt with R2 only, L2=0, t2=0 (z0 = 0.6×L1)

## Key Features

- **Multi-case comparative analysis**: Three different suction caisson configurations
- **Configurable rotation center coefficients**: Different z0 values for each case
- **Detailed moment component breakdown**: M1-M12 analysis
- **Winkler foundation model**: Implementation with mobilization factors
- **English documentation**: All comments and outputs in English for international collaboration

## Main Analysis Script

### `yes-MSC-H-right_cenf_test.py`

This is the primary analysis script that compares three different suction caisson configurations.

#### Input Parameters
- **Geometric Parameters**: 
  - D = 2.4m (diameter)
  - L = 2.4m (length) 
  - t = 0.04m (wall thickness)
- **Skirt Parameters**: 
  - R2 = 1.2m (skirt radius)
  - L2 = 0/1.2m (skirt length, case-dependent)
  - t2 = 0/0.04m (skirt thickness, case-dependent)
- **Soil Parameters**: 
  - γ' = 7.85 kN/m³ (effective unit weight)
  - φ' = 40° (friction angle)
  - K0 = 0.38, Ka = 0.25 (earth pressure coefficients)
  - fc = 0.18 (friction coefficient)
- **Winkler Parameters**: 
  - m = 2.2×10³ kN/m⁴ (modulus)
  - Kv = 20.0×10² kN/m³ (vertical modulus)

#### Rotation Center Coefficients
- **Case 1 (No Skirt)**: z0 = 0.8 × L1
- **Case 2 (Full Skirt)**: z0 = 0.55 × L1  
- **Case 3 (Simplified Skirt)**: z0 = 0.6 × L1

#### Output Results
- **Horizontal Capacity (Hu)**: Ultimate horizontal load capacity for each case
- **Moment Capacity (Mu)**: Ultimate moment capacity for each case
- **Comparative Analysis**: Percentage improvements between cases
- **Detailed Breakdown**: Individual moment components (M1-M12, M7s)

## Usage

Run the main analysis script:
```bash
python yes-MSC-H-right_cenf_test.py
```

The script will output:
1. Detailed results for each of the three cases
2. Moment component breakdown
3. Comparative analysis showing percentage improvements

## Research Applications

This code is suitable for:
- **Offshore Foundation Design**: Analysis of suction caisson foundations
- **Parametric Studies**: Investigating the effect of skirt configurations
- **Academic Research**: Comparing different analytical approaches
- **Engineering Practice**: Preliminary design of suction caisson foundations

## References

Based on theoretical framework for suction caisson horizontal capacity analysis with Winkler foundation models and passive/active earth pressure considerations.

## Author

Developed for suction caisson research and engineering applications.

## License

This project is available for academic and research purposes.
