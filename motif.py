import streamlit as st
import altair as alt
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.markdown(r"""## Feed-Forward loop
Select thresholds for $ Y  \rightarrow Z $ and $ X \rightarrow Z $""")

col_1, col_2 = st.columns(2)

with col_1:

    st.header(r'$K_{YZ}$')
    st.markdown(r'Observe the delay effect on "ON" depends on $K_{YZ}$')
    K_YZ = st.slider('', min_value=0.1,
                     max_value=2.0, value=1.0)

with col_2:
    st.header(r'$K_{XZ}$')
    st.markdown(r'What happens if you increase way too much')
    K_XZ = st.slider('', min_value=0.1,
                     max_value=1.2, value=0.5)


def FFL_Model(t, RNA, K_YZ, K_XZ):

    K_XY = 0.5
    a_Y = 0.2
    a_Z = 0.2
    b_Y = 1
    b_Z = 1

    X, Y, Z = RNA

    return [
        0,
        b_Y * int(X > K_XY) - a_Y * Y,
        b_Z * int(X > K_XZ) * int(Y > K_YZ) - a_Z * Z
    ]


t = np.linspace(0, 14, 100)

sol = solve_ivp(FFL_Model, t_span=[0, 10], y0=(
    1, 0, 0), args=(K_YZ, K_XZ,), dense_output=True)

rise = sol.sol(t)
solutions = rise.T[-1]

_, y, z = solutions
sol = solve_ivp(FFL_Model, t_span=[0, 10], y0=(
    0, y, z), args=(K_YZ, K_XZ,), dense_output=True)
fall = sol.sol(t)


lines = np.vstack((rise.T, fall.T))
data = pd.DataFrame(lines, columns=['X', 'Y', 'Z'])
data['time'] = np.linspace(0, 28, 200)
data = data.melt(value_vars=['X', 'Y', 'Z'], id_vars=['time'])

t_star = data[(data['variable'] == 'Y') & (
    data['value'] > K_YZ)].iloc[0]['time']

line = alt.Chart(pd.DataFrame({'y': [t_star]})).mark_rule(color='orange', strokeDash=(5, 5)).encode(
    x='y')

Chart = alt.Chart(data).mark_line().encode(
    alt.X('time', title='Time'),
    alt.Y('value', title='Value'),
    color='variable',
).properties(
    width=700
)

st.altair_chart(line + Chart)
