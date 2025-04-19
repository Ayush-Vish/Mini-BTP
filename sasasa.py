import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d
from xgboost import XGBClassifier
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# ---------------------
# 1) System Parameters
# ---------------------
SNR_dB = np.arange(-25, 6, 1)  # SNR range in dB
SNR_linear = 10**(SNR_dB / 10)

# Channel gains
h_u1_2 = 0.65
h_u2_2 = 2.7
h_oma_2 = 0.45

# Power allocation
alpha1 = 0.8
alpha2 = 0.2

# Detection parameters
Ns = 50
sigma_n2 = 1.0
alpha_cyclic = 0.1
Pf_u1 = Pf_u2 = Pf_oma = 0.10
M1, M2 = 2, 2

# Monte Carlo parameters
num_samples = 5000  # Reduced for efficiency
test_size = 0.2

# Cyclic lags for CAF (scaled from 370, 310)
cyclic_lags = [18, 15]

# -------------------------------------------------
# 2) Cyclic Correlation Detection Functions
# -------------------------------------------------
def compute_cyclic_threshold(Pf, Ns, sigma_n2):
    return np.sqrt(2 * sigma_n2 / Ns) * norm.ppf(1 - Pf)

def cyclic_correlation_pd(SNR, alpha, h_2, Ns, lambda_val, sigma_n2, alpha_cyclic):
    signal_power = alpha * h_2 * SNR
    var_cyclic_H1 = (sigma_n2 + signal_power)**2 / Ns
    return 1 - norm.cdf((lambda_val - signal_power) / np.sqrt(var_cyclic_H1))

# -------------------------------------------------
# 3) ML Classifiers Setup
# -------------------------------------------------
def generate_ml_data(snr_lin, num_samples):
    X = []
    y = []

    for _ in range(num_samples):
        signal_present = np.random.rand() > 0.5
        if signal_present:
            x_u1 = np.sqrt(alpha1 * h_u1_2 * snr_lin) * np.random.randn(Ns)
            x_u2 = np.sqrt(alpha2 * h_u2_2 * snr_lin) * np.random.randn(Ns)
            signal = x_u1 + x_u2
        else:
            signal = np.zeros(Ns)

        noise = np.sqrt(sigma_n2/2) * (np.random.randn(Ns) + 1j*np.random.randn(Ns))
        received = signal + noise

        # Extract features
        mag = np.abs(received)

        # CAF features
        caf_values = []
        for lag in cyclic_lags:
            if lag < Ns:
                caf = np.mean(received[:-lag] * np.conj(received[lag:]))
                caf_values.append(np.abs(caf))
        caf_peak = max(caf_values) if caf_values else 0.0
        caf_variance = np.var(caf_values) if len(caf_values) > 1 else 0.0

        # Spectral Correlation Density (approximated)
        fft_received = np.fft.fft(received)
        scd = np.abs(np.fft.fftshift(np.correlate(fft_received, fft_received, mode='same')))
        scd_peak = np.max(scd) / Ns

        # Signal power estimate
        signal_power_est = max(np.mean(mag**2) - sigma_n2, 0)

        # New statistical features
        mag_kurtosis = kurtosis(mag)
        mag_skewness = skew(mag)

        # Feature list: Optimized to 10 features
        features = [
            np.mean(mag),           # Mean magnitude
            np.std(mag),            # Std magnitude
            np.sum(mag**2),         # Energy
            np.percentile(mag, 25), # 25th percentile
            np.percentile(mag, 75), # 75th percentile
            caf_peak,               # CAF peak
            caf_variance,           # CAF variance
            signal_power_est,       # Signal power
            mag_kurtosis,           # Kurtosis
            mag_skewness            # Skewness
        ]

        X.append(features)
        y.append(signal_present)

    return np.array(X), np.array(y)

def train_classifiers(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Optimized classifiers with GridSearchCV
    classifiers = {
        'Random Forest': GridSearchCV(
            RandomForestClassifier(random_state=42),
            {'n_estimators': [50, 100], 'max_depth': [None, 10]},
            cv=3
        ),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost': GridSearchCV(
            XGBClassifier(random_state=42, eval_metric='logloss'),
            {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
            cv=3
        )
    }

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_probs = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        results[name] = (fpr, tpr, roc_auc)

    return results, X_test, y_test

# -------------------------------------------------
# 4) Main Simulation with Parallel Processing
# -------------------------------------------------
def process_snr(snr_db, snr_lin):
    # Cyclic correlation
    lambda_noma = compute_cyclic_threshold(Pf_u1, Ns, sigma_n2)
    lambda_oma = compute_cyclic_threshold(Pf_oma, Ns, sigma_n2)

    pd_u1 = cyclic_correlation_pd(snr_lin, alpha1, h_u1_2, Ns, lambda_noma, sigma_n2, alpha_cyclic)
    pd_u2 = cyclic_correlation_pd(snr_lin, alpha2, h_u2_2, Ns, lambda_noma, sigma_n2, alpha_cyclic)
    pd_oma = cyclic_correlation_pd(snr_lin, 1.0, h_oma_2, Ns, lambda_oma, sigma_n2, alpha_cyclic)

    # ML detection
    X, y = generate_ml_data(snr_lin, num_samples)
    results, _, _ = train_classifiers(X, y)

    # Get Pd at Pf=0.10
    fpr_desired = 0.10
    pd_results = {}
    for name, (fpr, tpr, _) in results.items():
        if max(fpr) >= fpr_desired:
            interp_tpr = interp1d(fpr, tpr)
            pd_results[name] = float(interp_tpr(fpr_desired))
        else:
            pd_results[name] = 1.0

    return pd_u1, pd_u2, pd_oma, pd_results

# Parallel execution
results = Parallel(n_jobs=-1)(delayed(process_snr)(snr_db, snr_lin) for snr_db, snr_lin in zip(SNR_dB, SNR_linear))

# Unpack results
Pd_u1 = [r[0] for r in results]
Pd_u2 = [r[1] for r in results]
Pd_oma = [r[2] for r in results]
Pd_rf = [r[3].get('Random Forest', 0) for r in results]
Pd_lr = [r[3].get('Logistic Regression', 0) for r in results]
Pd_xgb = [r[3].get('XGBoost', 0) for r in results]

# Print results
for idx, snr_db in enumerate(SNR_dB):
    print(f"SNR = {snr_db:3d} dB | U1-Pd = {Pd_u1[idx]:.3f} | U2-Pd = {Pd_u2[idx]:.3f} | OMA-Pd = {Pd_oma[idx]:.3f}")
    print(f"ML Results: RF-Pd = {Pd_rf[idx]:.3f}, LR-Pd = {Pd_lr[idx]:.3f}, XGB-Pd = {Pd_xgb[idx]:.3f}\n")

# -------------------------------------------------
# 5) Plot Results
# -------------------------------------------------
plt.figure(figsize=(14, 8))
plt.plot(SNR_dB, Pd_u1, 'b-o', label='U1-Pd (Cyclic)', markevery=2)
plt.plot(SNR_dB, Pd_u2, 'r-s', label='U2-Pd (Cyclic)', markevery=2)
plt.plot(SNR_dB, Pd_oma, 'g-^', label='OMA (Cyclic)', markevery=2)
plt.plot(SNR_dB, Pd_rf, 'm-D', label='Random Forest', markevery=2, markersize=8)
plt.plot(SNR_dB, Pd_lr, 'c-p', label='Logistic Regression', markevery=2, markersize=8)
plt.plot(SNR_dB, Pd_xgb, 'y-*', label='XGBoost', markevery=2, markersize=10)
plt.plot(SNR_dB, [Pf_u1]*len(SNR_dB), 'b--', label='U1-Pf', markevery=5)
plt.plot(SNR_dB, [Pf_u2]*len(SNR_dB), 'r--', label='U2-Pf', markevery=5)
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Optimized Detection: Cyclic Correlation vs. Machine Learning Classifiers\n($\\alpha_1 : \\alpha_2 = 4:1$)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
