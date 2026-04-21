import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Craving Trajectory Simulator",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open('mdn_weights.json') as f:
        all_weights = json.load(f)
    with open('mdn_params.json') as f:
        params = json.load(f)
    return all_weights['mdn'], all_weights['encoder'], params

mdn_w, enc_w, params = load_model()

def relu(x): return np.maximum(0, x)
def softmax(x): e = np.exp(x - x.max()); return e / e.sum()
def softplus(x): return np.log1p(np.exp(x))
def dense(x, W, b, act): return act(x @ np.array(W) + np.array(b))

def numpy_encoder(x):
    h = dense(x, enc_w['e1'][0], enc_w['e1'][1], relu)
    h = dense(h, enc_w['e2'][0], enc_w['e2'][1], relu)
    h = dense(h, enc_w['latent'][0], enc_w['latent'][1], lambda z: z)
    return h

def numpy_mdn(x):
    h = dense(x, mdn_w['e1'][0], mdn_w['e1'][1], relu)
    h = dense(h, mdn_w['e2'][0], mdn_w['e2'][1], relu)
    h = dense(h, mdn_w['latent'][0], mdn_w['latent'][1], lambda z: z)
    h = dense(h, mdn_w['d1'][0], mdn_w['d1'][1], relu)
    h = dense(h, mdn_w['d2'][0], mdn_w['d2'][1], relu)
    pi    = softmax(h @ np.array(mdn_w['pi'][0]) + np.array(mdn_w['pi'][1]))
    mu    = (h @ np.array(mdn_w['mu'][0]) + np.array(mdn_w['mu'][1])).reshape(3, 5)
    sigma = softplus(h @ np.array(mdn_w['sigma'][0]) + np.array(mdn_w['sigma'][1])).reshape(3, 5) + 1e-6
    return pi, mu, sigma

LABELS_CRA   = ['DU — Desire to use', 'IU — Intention to use',
                'APE — Anticipation pos. effects',
                'AR — Anticipation of relief', 'LC — Loss of control']
LABELS_SHORT = ['DU', 'IU', 'APE', 'AR', 'LC']
COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']

latent_centers  = np.array(params['latent_centers'])
DELTA_T_MEAN    = params['DELTA_T_MEAN']
DELTA_T_SD      = params['DELTA_T_SD']
THETA           = params['THETA']
K, N_OUT        = 3, 5

# Parametri simulazione corretti — Δt coerente con training
STEP_DAYS_SIM   = 32
T_STEPS_SIM     = int(365 / STEP_DAYS_SIM)   # 11 assessment ~1 anno
DELTA_T_SIM_Z   = (STEP_DAYS_SIM - DELTA_T_MEAN) / DELTA_T_SD

def clip_z(x): return np.clip(x, -3.5, 3.5)
def nearest_center(x): return int(np.argmin(np.sum((latent_centers - x)**2, axis=1)))

def mdn_sample(x_input):
    pi, mu, sigma = numpy_mdn(x_input)
    k = np.random.choice(K, p=pi)
    return np.random.normal(mu[k], sigma[k])

def simulate(craving_z, n_seeds=10):
    all_traj, all_states = [], []
    for _ in range(n_seeds):
        Z = np.full((T_STEPS_SIM + 1, N_OUT), np.nan)
        S = np.zeros(T_STEPS_SIM + 1, dtype=int)
        Z[0] = craving_z
        lat = numpy_encoder(np.append(craving_z, DELTA_T_SIM_Z))
        S[0] = nearest_center(lat)
        for t in range(1, T_STEPS_SIM + 1):
            xn = clip_z(mdn_sample(np.append(Z[t-1], DELTA_T_SIM_Z)))
            Z[t] = xn
            S[t] = nearest_center(numpy_encoder(np.append(xn, DELTA_T_SIM_Z)))
        all_traj.append(Z)
        all_states.append(S)
    return np.array(all_traj), all_states[0]

# ============================================================
st.title("🧠 Craving Trajectory Simulator")
st.markdown("*MDN v10 — Autoregressive simulation over ~1 year (11 assessments, Δt ≈ 32 days)*")
st.markdown("---")

col_left, col_right = st.columns([1, 2.5])

with col_left:
    st.subheader("⚙️ Baseline craving (z-score)")
    cra_vals = [st.slider(lab, -3.5, 3.5, 0.0, 0.1, key=f'cra_{i}')
                for i, lab in enumerate(LABELS_CRA)]

    st.subheader("🔧 Settings")
    n_seeds = st.slider("Trajectories (uncertainty band)", 1, 30, 10)
    seed    = st.number_input("Random seed", value=42, step=1)
    np.random.seed(int(seed))

    comp_base = float(np.mean(cra_vals))
    st.markdown("---")
    st.markdown("**Baseline composite**")
    color_base = '#FF5722' if comp_base > 0.5 else '#4CAF50' if comp_base < -0.5 else '#FF9800'
    st.markdown(
        f"<div style='text-align:center;background:{color_base};color:white;"
        f"padding:12px;border-radius:8px;font-size:20px'><b>{comp_base:+.2f}</b></div>",
        unsafe_allow_html=True
    )
    st.markdown(f"<div style='text-align:center;color:gray;font-size:11px;margin-top:4px'>"
                f"Each step ≈ 32 days | {T_STEPS_SIM} assessments over ~1 year</div>",
                unsafe_allow_html=True)

with col_right:
    craving_z = np.array(cra_vals, dtype=np.float64)

    with st.spinner("Simulating..."):
        all_traj, states = simulate(craving_z, n_seeds=n_seeds)

    assessments = np.arange(T_STEPS_SIM + 1)
    mean_traj   = all_traj.mean(axis=0)
    comp_traj   = mean_traj.mean(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8),
                             gridspec_kw={'height_ratios': [3.5, 1]})
    ax = axes[0]

    for i in range(N_OUT):
        dm = all_traj[:,:,i].mean(axis=0)
        ds = all_traj[:,:,i].std(axis=0)
        ax.plot(assessments, dm, color=COLORS[i], lw=1.8,
                label=LABELS_SHORT[i], alpha=0.9)
        ax.fill_between(assessments, dm-ds, dm+ds,
                        color=COLORS[i], alpha=0.12)

    ax.plot(assessments, comp_traj, color='black', lw=2.5,
            linestyle='--', label='Composite', zorder=5)
    ax.axhline(0,      color='gray',      lw=0.8, linestyle=':')
    ax.axhline(THETA,  color='red',       lw=1.2, linestyle='--',
               alpha=0.7, label=f'θ={THETA} (relapse risk)')
    ax.axhline(-THETA, color='steelblue', lw=0.8, linestyle='--', alpha=0.3)
    ax.set_xlim(0, T_STEPS_SIM); ax.set_ylim(-3.5, 3.5)
    ax.set_ylabel('Craving z-score', fontsize=12)
    ax.set_title(
        f'Simulated craving trajectory — ~1 year  '
        f'(baseline composite: {comp_base:+.2f})',
        fontsize=12
    )
    ax.legend(loc='upper right', fontsize=9, ncol=3)
    ax.grid(True, alpha=0.15)
    # Asse X con label mesi approssimativi
    month_labels = [f'M{int(t*32/30)}' for t in assessments]
    ax.set_xticks(assessments)
    ax.set_xticklabels(month_labels, fontsize=8)

    # Latent states
    ax2 = axes[1]
    sc = {0: '#4CAF50', 1: '#FF9800', 2: '#FF5722'}
    sl = {0: 'S0 — low craving', 1: 'S1 — intermediate', 2: 'S2 — high craving'}
    for t in range(T_STEPS_SIM + 1):
        ax2.axvspan(t-0.5, t+0.5, color=sc.get(states[t], 'gray'), alpha=0.45)
    ax2.set_xlim(0, T_STEPS_SIM); ax2.set_yticks([])
    ax2.set_xlabel('Assessment (~32 days each)', fontsize=11)
    ax2.set_ylabel('Latent state', fontsize=9)
    ax2.set_xticks(assessments)
    ax2.set_xticklabels(month_labels, fontsize=8)
    ax2.legend(
        handles=[mpatches.Patch(color=sc[s], label=sl[s]) for s in [0, 1, 2]],
        loc='upper right', fontsize=8, ncol=3
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    n_above  = int((comp_traj > THETA).sum())
    peak_dim = LABELS_SHORT[int(mean_traj.mean(axis=0).argmax())]
    m1.metric("Composite mean (~1y)", f"{comp_traj.mean():+.2f}")
    m2.metric("Composite SD",          f"{comp_traj.std():.2f}")
    m3.metric(f"Assessments above θ={THETA}", str(n_above))
    m4.metric("Most active dimension",   peak_dim)

st.markdown("---")
st.caption("MDN v10 | Bonfiglio, Renati, Penna (2026) | AIRS Conference")
