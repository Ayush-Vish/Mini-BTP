# -------------------------------------------------
# 5) Plot Results
# -------------------------------------------------
window_length = 7  # Must be odd
polyorder = 2

def smooth(data):
      return savgol_filter(data, window_length=window_length, polyorder=polyorder)

# Apply smoothing
Pd_u1_s = smooth(Pd_u1)
Pd_u2_s = smooth(Pd_u2)
Pd_oma_s = smooth(Pd_oma)
Pd_rf_s = smooth(Pd_rf)
Pd_lr_s = smooth(Pd_lr)
Pd_dt_s = smooth(Pd_dt)

plt.figure(figsize=(14, 8))

# Cyclic
plt.plot(SNR_dB, Pd_u1_s, 'b-o', label='U1-Pd (Cyclic)', markevery=2)
plt.plot(SNR_dB, Pd_u2_s, 'r-s', label='U2-Pd (Cyclic)', markevery=2)
plt.plot(SNR_dB, Pd_oma_s, 'g-^', label='OMA (Cyclic)', markevery=2)

# ML
plt.plot(SNR_dB, Pd_rf_s, 'm-D', label='Random Forest', markevery=2, markersize=8)
plt.plot(SNR_dB, Pd_lr_s, 'c-p', label='Logistic Regression', markevery=2, markersize=8)
plt.plot(SNR_dB, Pd_dt_s, 'y-*', label='Decision Tree', markevery=2, markersize=10)

# False Alarm lines
plt.plot(SNR_dB, [Pf_u1] * len(SNR_dB), 'b--', label='U1-Pf')
plt.plot(SNR_dB, [Pf_u2] * len(SNR_dB), 'r--', label='U2-Pf')

plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Smoothed Detection Performance: Cyclic Correlation vs. ML Classifiers\n($\\alpha_1 : \\alpha_2 = 4:1$)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
