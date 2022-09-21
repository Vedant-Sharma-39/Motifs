import streamlit as st
import altair as alt
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.write('Select the thresholds for Y -> Z and X -> Z')

K_YZ = st.slider('K_YZ', min_value=0.1,
                 max_value=2.0, value=1.0)


def FFL_Model(t, RNA, K_YZ):

    K_XY = 0.5

    K_XZ = 0.5
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


t = np.linspace(0, 5, 100)

sol = solve_ivp(FFL_Model, t_span=[0, 10], y0=(
    1, 0, 0), args=(K_YZ,), dense_output=True)

rise = sol.sol(t)
solutions = rise.T[-1]

_, y, z = solutions
sol = solve_ivp(FFL_Model, t_span=[0, 10], y0=(
    0, y, z), args=(K_YZ,), dense_output=True)
fall = sol.sol(t)

sns.set_style('darkgrid')

lines = np.vstack((rise.T, fall.T))

data = pd.DataFrame(lines, columns=['X', 'Y', 'Z'])
st.line_chart(data)
