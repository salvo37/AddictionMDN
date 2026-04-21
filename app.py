import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import timedelta, date

st.set_page_config(
    page_title="Craving Trajectory Simulator",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# Carica modello e parametri
# ============================================================
@st.cache_resource
def load_model():
    import tensorflow as tf
    mdn = tf.keras.models.load_model('mdn_model_v10.keras', compile=False)
    enc = tf.keras.models.load_model('encoder_mdn_v10.keras', compile=False)
    with open('mdn_params.json') as f:
        params = json.load(f)
    return mdn, enc, params

try:
    mdn_model, encoder_model, params = load_model()
    model_ok = True
except Exception as e:
    model_ok = False
    st.error(f"Errore caricamento modello: {e}")

# ============================================================
# Costanti
# ============================================================
VARS_CRAVING = [
    "desiderio_di__usare_sostanza",
    "intenzione_di_usare_sostanza",
    "anticipazione_effetti_positivi",
    "anticipazione_sollievo_dai_sintomi_d'astinenza_e_disforia",
    "perdita_di_controllo"
]
LABELS_CRA = ['DU — Desire to use', 'IU — Intention to use',
               'APE — Anticipation pos. effects',
               'AR — Anticipation of relief', 'LC — Loss of control']
LABELS_SHORT = ['DU', 'IU', 'APE', 'AR', 'LC']

VARS_RES = ['percezione_se', 'percezione_futuro', 'stile_strutturato',
            'competenze_sociali', 'coesione_familiare', 'risorse_sociali']
LABELS_RES = ['PS — Perceived self', 'PF — Planned future',
              'SS — Structured style', 'SC — Social competence',
              'FC — Family cohesion', 'SR — Social resources']

COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
K = 3
N_OUT = 5

latent_centers = np.array(params['latent_centers'])
DELTA_T_MEAN = params['DELTA_T_MEAN']
DELTA_T_SD = params['DELTA_T_SD']
DELTA_T_SIM_Z = params['DELTA_T_SIM_Z']
T_STEPS = params['T_STEPS']
THETA = params['THETA']

# ============================================================
# Funzioni simulazione
# ============================================================
def clip_z(x, lo=-3.5, hi=3.5):
    return np.clip(x, lo, hi)

def nearest_center(x, cm):
    return int(np.argmin(np.sum((cm - x)**2, axis=1)))

def mdn_sample(x_input):
    pred = mdn_model.predict(x_input.reshape(1, -1), verbose=0)[0]
    pi    = pred[:K]
    mu    = pred[K:K+K*N_OUT].reshape(K, N_OUT)
    sigma = pred[K+K*N_OUT:].reshape(K, N_OUT) + 1e-6
    pi = pi / pi.sum()
    k = np.random.choice(K, p=pi)
    return np.random.normal(mu[k], sigma[k])

def simulate(craving_z, n_seeds=5):
    """Simula n_seeds traiettorie e restituisce la media e le singole."""
    all_traj = []
    for _ in range(n_seeds):
        Z = np.full((T_STEPS + 1, N_OUT), np.nan)
        S = np.zeros(T_STEPS + 1, dtype=int)
        Z[0] = craving_z
        xf = np.append(craving_z, DELTA_T_SIM_Z)
        lat = encoder_model.predict(xf.reshape(1, -1), verbose=0)[0]
        S[0] = nearest_center(lat, latent_centers)
        for t in range(1, T_STEPS + 1):
            xi = np.append(Z[t-1], DELTA_T_SIM_Z)
            xn = clip_z(mdn_sample(xi))
            Z[t] = xn
            xf2 = np.append(xn, DELTA_T_SIM_Z)
            lat = encoder_model.predict(xf2.reshape(1, -1), verbose=0)[0]
            S[t] = nearest_center(lat, latent_centers)
        all_traj.append(Z)
    return np.array(all_traj), S

def get_resilience_group(z, t33, t67):
    if z <= t33: return 'LOW', '#FF5722'
    elif z > t67: return 'HIGH', '#4CAF50'
    else: return 'MID', '#FF9800'

# Terzili approssimativi da dati in-cura
T33_IND, T67_IND = -0.65, 0.42
T33_FAM, T67_FAM = -0.85, 0.25
T33_SOC, T67_SOC = -1.48, -0.48

# ============================================================
# UI
# ============================================================
st.title("🧠 Craving Trajectory Simulator")
st.markdown("*Neural bottleneck MDN v10 — Agent-based simulation over 52 weeks*")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Craving at baseline (z-score)")
    cra_vals = []
    for i, lab in enumerate(LABELS_CRA):
        v = st.slider(lab, min_value=-3.5, max_value=3.5, value=0.0, step=0.1, key=f'cra_{i}')
        cra_vals.append(v)

    st.subheader("Resilience at baseline (z-score)")
    res_vals = []
    for i, lab in enumerate(LABELS_RES):
        v = st.slider(lab, min_value=-3.5, max_value=3.5, value=0.0, step=0.1, key=f'res_{i}')
        res_vals.append(v)

    st.subheader("Simulation settings")
    n_seeds = st.slider("N trajectories (uncertainty band)", 1, 20, 5)
    np.random.seed(st.number_input("Random seed", value=42, step=1))

    # Resilience groups
    ind_z = np.mean(res_vals[:4])
    fam_z = res_vals[4]
    soc_z = res_vals[5]
    grp_ind, col_ind = get_resilience_group(ind_z, T33_IND, T67_IND)
    grp_fam, col_fam = get_resilience_group(fam_z, T33_FAM, T67_FAM)
    grp_soc, col_soc = get_resilience_group(soc_z, T33_SOC, T67_SOC)

    st.markdown("---")
    st.markdown("**Resilience profile**")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div style='text-align:center;background:{col_ind};color:white;padding:6px;border-radius:6px'><b>IND</b><br>{grp_ind}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div style='text-align:center;background:{col_fam};color:white;padding:6px;border-radius:6px'><b>FAM</b><br>{grp_fam}</div>", unsafe_allow_html=True)
    c3.markdown(f"<div style='text-align:center;background:{col_soc};color:white;padding:6px;border-radius:6px'><b>SOC</b><br>{grp_soc}</div>", unsafe_allow_html=True)

with col_right:
    if model_ok:
        craving_z = np.array(cra_vals, dtype=np.float64)
        comp_base = float(np.mean(craving_z))

        with st.spinner("Simulating..."):
            all_traj, states = simulate(craving_z, n_seeds=n_seeds)

        weeks = np.arange(T_STEPS + 1)
        mean_traj = all_traj.mean(axis=0)
        comp_traj = mean_traj.mean(axis=1)

        # ---- Grafico principale ----
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        ax = axes[0]

        for i in range(N_OUT):
            dim_mean = all_traj[:, :, i].mean(axis=0)
            dim_std  = all_traj[:, :, i].std(axis=0)
            ax.plot(weeks, dim_mean, color=COLORS[i], lw=1.5, label=LABELS_SHORT[i], alpha=0.9)
            ax.fill_between(weeks, dim_mean - dim_std, dim_mean + dim_std,
                            color=COLORS[i], alpha=0.1)

        ax.plot(weeks, comp_traj, color='black', lw=2.5, linestyle='--', label='Composite', zorder=5)
        ax.axhline(0, color='gray', lw=0.8, linestyle=':')
        ax.axhline(THETA, color='red', lw=1, linestyle='--', alpha=0.6, label=f'θ = {THETA}')
        ax.axhline(-THETA, color='blue', lw=1, linestyle='--', alpha=0.3)
        ax.axvline(0, color='green', lw=1, alpha=0.5)

        ax.set_xlim(0, T_STEPS)
        ax.set_ylim(-3.5, 3.5)
        ax.set_ylabel('Craving z-score', fontsize=11)
        ax.set_title(f'Simulated craving trajectory — 52 weeks\n'
                     f'Baseline composite: {comp_base:+.2f} | '
                     f'IND:{grp_ind}  FAM:{grp_fam}  SOC:{grp_soc}', fontsize=11)
        ax.legend(loc='upper right', fontsize=9, ncol=3)
        ax.grid(True, alpha=0.2)

        # ---- Latent states ----
        ax2 = axes[1]
        state_colors = {0: '#4CAF50', 1: '#FF9800', 2: '#FF5722'}
        state_labels = {0: 'S0 — low craving', 1: 'S1 — mid', 2: 'S2 — high craving'}
        for t in range(T_STEPS + 1):
            ax2.axvspan(t - 0.5, t + 0.5, color=state_colors.get(states[t], 'gray'), alpha=0.4)
        ax2.set_xlim(0, T_STEPS)
        ax2.set_yticks([])
        ax2.set_xlabel('Week', fontsize=11)
        ax2.set_ylabel('Latent state', fontsize=9)
        patches = [mpatches.Patch(color=state_colors[s], label=state_labels[s]) for s in [0, 1, 2]]
        ax2.legend(handles=patches, loc='upper right', fontsize=8, ncol=3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ---- Metriche ----
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        n_above = int((comp_traj > THETA).sum())
        m1.metric("Composite mean", f"{comp_traj.mean():+.2f}")
        m2.metric("Composite SD", f"{comp_traj.std():.2f}")
        m3.metric(f"Weeks above θ={THETA}", str(n_above))
        peak_dim = LABELS_SHORT[int(mean_traj.mean(axis=0).argmax())]
        m4.metric("Highest dimension (mean)", peak_dim)
    else:
        st.warning("Model not loaded. Check keras files.")

st.markdown("---")
st.caption("MDN v10 | Bonfiglio, Renati, Penna (2026) | AIRS Conference")
