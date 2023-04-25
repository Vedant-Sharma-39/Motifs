import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns
import math
sns.set_style('darkgrid')

def update(r_prey, d_prey, e_pred, d_pred):
    def wrap(prey, predator):
        updated_prey = int(r_prey * prey - d_prey * predator)
        updated_predator = int(e_pred * prey - predator * d_pred)
        new_prey = max(updated_prey, 0)
        new_predator = max(updated_predator, 0)
        return (new_prey, new_predator)
    return wrap

class SiteModel:
    def __init__(self, n_sites, initial_pop, update_pop):
        self.n_sites = n_sites
        self.prey_pop = [initial_pop[0]]
        self.pred_pop = [initial_pop[1]]
        self.update_pop = update_pop
        self.pop_history_prey = []
        self.pop_history_pred = []

    def distribute_populations(self):
        probabilities = np.ones(self.n_sites) / self.n_sites
        allotments_prey = np.random.choice(np.arange(1, self.n_sites + 1), size=self.prey_pop[-1], p=probabilities)
        allotments_pred = np.random.choice(np.arange(1, self.n_sites + 1), size=self.pred_pop[-1], p=probabilities)
        counts_prey = Counter(allotments_prey)
        counts_pred = Counter(allotments_pred)
        return counts_prey, counts_pred

    def iterate(self, num_iterations):
        for _ in range(num_iterations):
            counts_prey, counts_pred = self.distribute_populations()
            new_prey_pop = 0
            new_pred_pop = 0
            for site in range(1, self.n_sites + 1):
                prey_site = counts_prey.get(site, 0)
                pred_site = counts_pred.get(site, 0)
                updated_prey, updated_pred = self.update_pop(prey_site, pred_site)
                new_prey_pop += updated_prey
                new_pred_pop += updated_pred
            self.prey_pop.append(new_prey_pop)
            self.pred_pop.append(new_pred_pop)
            populations_prey = np.zeros(self.n_sites)
            for site, count in counts_prey.items():
                populations_prey[site-1] = count
            self.pop_history_prey.append(populations_prey)
            populations_pred = np.zeros(self.n_sites)
            for site, count in counts_pred.items():
                populations_pred[site-1] = count
            self.pop_history_pred.append(populations_pred)


# Define the model parameters and run the simulation
k, r_prey, e_pred, d_prey = 10, 2, 0.5, 0.5
mod = SiteModel(n_sites=10, initial_pop=(30, 5), update_pop=update(r_prey, d_prey, e_pred, d_pred))
mod.iterate(100)

# Create the plot
fig, ax = plt.subplots()
ax.plot(mod.prey_pop, label='Prey Population')
ax.plot(mod.pred_pop, label='Predator Population')
ax.set_xlabel('Iteration')
ax.set_ylabel('Population')
ax.set_title(f'r_prey: {r_prey:.1f}, d_prey: {d_prey:.1f}')
ax.legend()
plt.tight_layout()

# Display the plot on Streamlit
st.pyplot(fig)


