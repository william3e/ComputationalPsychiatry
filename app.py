import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 関数定義
def integrate_and_fire(I, dt, t_max):
    """積分発火モデルのシミュレーション"""
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

def hodgkin_huxley(I, dt, t_max):
    """ホジキン-ハクスレーモデルのシミュレーション"""
    t = np.arange(0, t_max, dt)
    V = np.zeros_like(t)
    n = np.zeros_like(t)
    m = np.zeros_like(t)
    h = np.zeros_like(t)
    g_Na = 120
    g_K = 36
    g_L = 0.3
    E_Na = 50
    E_K = -77
    E_L = -54.4
    C_m = 1.0

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
        g_Na_current = g_Na * (m[i]**3) * h[i]
        g_K_current = g_K * (n[i]**4)
        g_L_current = g_L
        dV = (I - g_Na_current * (V[i-1] - E_Na) - g_K_current * (V[i-1] - E_K) - g_L_current * (V[i-1] - E_L)) / C_m
        V[i] = V[i-1] + dV * dt
    return t, V

def equivalent_circuit(I, R, C, dt, t_max):
    """等価回路モデルのシミュレーション"""
    t = np.arange(0, t_max, dt)
    V = np.zeros_like(t)
    for i in range(1, len(t)):
        dV = (I - V[i-1] / R) / C
        V[i] = V[i-1] + dV * dt
    return t, V

# Streamlitアプリ
st.title("神経モデルシミュレーション")
model = st.sidebar.selectbox("モデルを選択してください", ["積分発火モデル", "ホジキン-ハクスレーモデル", "等価回路モデル"])

if model == "積分発火モデル":
    st.header("積分発火モデル")
    I = st.slider("入力電流 (I)", 0.0, 20.0, 10.0)
    dt = st.slider("時間ステップ (dt)", 0.01, 0.1, 0.05)
    t_max = st.slider("シミュレーション時間", 10, 100, 50)
    t, V = integrate_and_fire(I, dt, t_max)
    fig, ax = plt.subplots()
    ax.plot(t, V)
    ax.set_xlabel("時間 (ms)")
    ax.set_ylabel("膜電位 (mV)")
    st.pyplot(fig)

elif model == "ホジキン-ハクスレーモデル":
    st.header("ホジキン-ハクスレーモデル")
    I = st.slider("入力電流 (I)", 0.0, 20.0, 10.0)
    dt = st.slider("時間ステップ (dt)", 0.01, 0.1, 0.05)
    t_max = st.slider("シミュレーション時間", 10, 100, 50)
    t, V = hodgkin_huxley(I, dt, t_max)
    fig, ax = plt.subplots()
    ax.plot(t, V)
    ax.set_xlabel("時間 (ms)")
    ax.set_ylabel("膜電位 (mV)")
    st.pyplot(fig)

elif model == "等価回路モデル":
    st.header("等価回路モデル")
    I = st.slider("入力電流 (I)", 0.0, 20.0, 10.0)
    R = st.slider("抵抗 (R)", 1.0, 100.0, 10.0)
    C = st.slider("容量 (C)", 0.1, 10.0, 1.0)
    dt = st.slider("時間ステップ (dt)", 0.01, 0.1, 0.05)
    t_max = st.slider("シミュレーション時間", 10, 100, 50)
    t, V = equivalent_circuit(I, R, C, dt, t_max)
    fig, ax = plt.subplots()
    ax.plot(t, V)
    ax.set_xlabel("時間 (ms)")
    ax.set_ylabel("膜電位 (mV)")
    st.pyplot(fig)
