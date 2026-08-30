import pandas as pd

v1 = pd.read_csv('//wsl.localhost/Ubuntu/home/venkat/projects/Heat exchangers/Data/version 1/shell_tube_fouling_final.csv')
v2 = pd.read_csv('//wsl.localhost/Ubuntu/home/venkat/projects/Heat exchangers/Data/version 2/v2_physics_corrected.csv')
cols = ['T_in_C','m_dot_nominal_kg_s','Re','u_m_s','tau_w_Pa','dP_Pa','Rf_m2K_W','U_total_W_m2K','Q_total_W','thermal_efficiency']
print('COLUMN, V1_mean, V1_std, V2_mean, V2_std, V1_min, V1_max, V2_min, V2_max')
for c in cols:
    v1m = v1[c].mean()
    v1s = v1[c].std()
    v2m = v2[c].mean()
    v2s = v2[c].std()
    print(f'{c}, {v1m:.6f}, {v1s:.6f}, {v2m:.6f}, {v2s:.6f}, {v1[c].min():.6f}, {v1[c].max():.6f}, {v2[c].min():.6f}, {v2[c].max():.6f}')
