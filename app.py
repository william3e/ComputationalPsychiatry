import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Integrate-and-Fire Model ---
def integrate_and_fire(I, dt, t_max):
    t = np.arange(0, t_max, dt)
    V = np.zeros_like(t)
    threshold = -50
    V_reset = -65
    R = 10
    tau = 10
    V[0] = V_reset
    for i in range(1, len(t)):
        dV = (-(V[i-1] - V_reset) + R * I) / tau
        V[i] = V[i-1] + dV * dt
        if V[i] >= threshold:
            V[i] = V_reset
    return t, V

# --- Hodgkin-Huxley Model ---
def hodgkin_huxley(I, dt, t_max, Cm, gNa, gK, gL, ENa, EK, EL):
    t = np.arange(0, t_max, dt)
    V = np.zeros_like(t)
    n = np.zeros_like(t)
    m = np.zeros_like(t)
    h = np.zeros_like(t)

    def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    def beta_n(V): return 0.125 * np.exp(-(V + 65) / 80)
    def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    def beta_m(V): return 4.0 * np.exp(-(V + 65) / 18)
    def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
    def beta_h(V): return 1 / (1 + np.exp(-(V + 35) / 10))

    for i in range(1, len(t)):
        n[i] = n[i-1] + dt * (alpha_n(V[i-1]) * (1 - n[i-1]) - beta_n(V[i-1]) * n[i-1])
        m[i] = m[i-1] + dt * (alpha_m(V[i-1]) * (1 - m[i-1]) - beta_m(V[i-1]) * m[i-1])
        h[i] = h[i-1] + dt * (alpha_h(V[i-1]) * (1 - h[i-1]) - beta_h(V[i-1]) * h[i-1])
        gNa_current = gNa * (m[i]**3) * h[i]
        gK_current = gK * (n[i]**4)
        gL_current = gL
        dV = (I - gNa_current * (V[i-1] - ENa) - gK_current * (V[i-1] - EK) - gL_current * (V[i-1] - EL)) / Cm
        V[i] = V[i-1] + dV * dt
    return t, V

# --- Equivalent Circuit Model ---
def equivalent_circuit(I, R, C, dt, t_max):
    t = np.arange(0, t_max, dt)
    V = np.zeros_like(t)
    for i in range(1, len(t)):
        dV = (I - V[i-1] / R) / C
        V[i] = V[i-1] + dV * dt
    return t, V

# --- Streamlit App ---
st.title("Neural Models Simulation")
model = st.sidebar.selectbox("Select a Model", ["Integrate-and-Fire Model", "Hodgkin-Huxley Model", "Equivalent Circuit Model"])

if model == "Integrate-and-Fire Model":
    st.header("Integrate-and-Fire Model")
    st.latex(r"\tau \frac{dV}{dt} = -(V - V_{\text{reset}}) + R I")
    I = st.slider("Input Current (I)", 0.0, 20.0, 10.0)
    dt = st.slider("Time Step (dt)", 0.01, 0.1, 0.05)
    t_max = st.slider("Simulation Time (ms)", 10, 100, 50)
    t, V = integrate_and_fire(I, dt, t_max)
    fig, ax = plt.subplots()
    ax.plot(t, V, label="Membrane Potential")
    ax.axhline(-50, color='red', linestyle='--', label="Threshold (-50 mV)")
    ax.axhline(-65, color='blue', linestyle='--', label="Reset Potential (-65 mV)")
    ax.set_title("Membrane Potential over Time")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane Potential (mV)")
    ax.legend()
    st.pyplot(fig)

elif model == "Hodgkin-Huxley Model":
    st.header("Hodgkin-Huxley Model")
    st.latex(r"""
    C_m \frac{dV}{dt} = I - g_{\text{Na}} m^3 h (V - E_{\text{Na}}) - 
    g_{\text{K}} n^4 (V - E_{\text{K}}) - g_{\text{L}} (V - E_{\text{L}})
    """)
    I = st.slider("Input Current (I)", 0.0, 20.0, 10.0)
    dt = st.slider("Time Step (dt)", 0.01, 0.1, 0.05)
    t_max = st.slider("Simulation Time (ms)", 10, 100, 50)
    Cm = st.slider("Membrane Capacitance (Cm)", 0.5, 2.0, 1.0)
    gNa = st.slider("Na+ Conductance (gNa)", 50.0, 200.0, 120.0)
    gK = st.slider("K+ Conductance (gK)", 10.0, 50.0, 36.0)
    gL = st.slider("Leak Conductance (gL)", 0.1, 1.0, 0.3)
    t, V = hodgkin_huxley(I, dt, t_max, Cm, gNa, gK, gL, 50.0, -77.0, -54.4)
    fig, ax = plt.subplots()
    ax.plot(t, V, label="Membrane Potential")
    ax.set_title("Membrane Potential over Time")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane Potential (mV)")
    ax.legend()
    st.pyplot(fig)

elif model == "Equivalent Circuit Model":
    st.header("Equivalent Circuit Model")
    st.latex(r"\frac{dV}{dt} = \frac{I - V / R}{C}")
    I = st.slider("Input Current (I)", 0.0, 20.0, 10.0)
    R = st.slider("Resistance (R)", 1.0, 100.0, 10.0)
    C = st.slider("Capacitance (C)", 0.1, 10.0, 1.0)
    dt = st.slider("Time Step (dt)", 0.01, 0.1, 0.05)
    t_max = st.slider("Simulation Time (ms)", 10, 100, 50)
    t, V = equivalent_circuit(I, R, C, dt, t_max)
    fig, ax = plt.subplots()
    ax.plot(t, V, label="Membrane Potential")
    ax.set_title("Membrane Potential over Time")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane Potential (mV)")
    ax.legend()
    st.pyplot(fig)
