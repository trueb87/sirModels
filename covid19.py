import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
N = 7000000
# Initial number of infected recovered, and deceased individuals, I0, R0, and D0.
I0 = 503
R0 = 0
D0 = 9
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0 - D0
# Contact rate, recovered rate, and mortality rate
contact_rate = 0.15  # 10% contact rate
recovered_rate = 1. / 14  # Takes 1 person 14 days to recover
mortality_rate = 0.0005  # 0.5% mortality rate
# A grid of time points (in days)
t = np.linspace(0, 365, 365)


# The SIR model differential equations.
def deriv(y, t, N, beta, gamma, theta):
    S, I, R, D = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I - theta * I
    dRdt = gamma * I
    dDdt = theta * I
    return dSdt, dIdt, dRdt, dDdt


# Initial conditions vector
y0 = S0, I0, R0, D0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, contact_rate, recovered_rate, mortality_rate))
S, I, R, D = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, fc="white", axisbelow=True)
ax.set_title("SIRD model without social distancing")
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.plot(t, D/1000, 'k', alpha=0.5, lw=2, label='Deceased')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (thousands)')
ax.set_ylim(0, 8000)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='gray', lw=1, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
print("Susceptible = {}".format(S[-1]))
print("Infected = {}".format(I[-1]))
print("Max Infected = {}".format(max(I)))
print("Recovered = {}".format(R[-1]))
print("Deceased = {}".format(D[-1]))
plt.show()
