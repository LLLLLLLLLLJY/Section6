#!/usr/bin/env python

import os
import numpy as np
from numpy.linalg import eigh, norm, matrix_power
from scipy.linalg import null_space, logm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 确保保存目录存在
save_dir = "Section6/A/"
os.makedirs(save_dir, exist_ok=True)

# 1. 计算 Heisenberg XXX Hamiltonian 的转移矩阵
def heisenberg_xxx_hamiltonian(num_sites):
    def is_adjacent_hopping(state_a, state_b):
        if state_a == state_b:
            return 0
        diff = state_a ^ state_b
        if state_a & diff == 0 or state_b & diff == 0:
            return 0
        return int(diff % 3 == 0 and diff // 3 & diff // 3 - 1 == 0) + int(diff == (1 << num_sites - 1) + 1)

    def get_site_spin(state, index):
        return 1 / 2 - (state >> index % num_sites & 1)

    def heisenberg_xxx_element(state_a, state_b):
        result = 0
        if state_a == state_b:
            for i in range(num_sites):
                result += 1 / 4 - get_site_spin(state_a, i) * get_site_spin(state_a, i + 1)
        result -= is_adjacent_hopping(state_a, state_b) / 2
        return result

    return [[heisenberg_xxx_element(i, j) for j in range(1 << num_sites)] for i in range(1 << num_sites)]

def boltzmann_distribution(energy_levels, temperature):
    probs = np.exp(-energy_levels / temperature)
    return probs / np.sum(probs)

def transition_matrix(num_sites, temperature=1):
    hamiltonian = heisenberg_xxx_hamiltonian(num_sites)
    energy_levels, eigenvectors = eigh(hamiltonian)
    probabilities = boltzmann_distribution(energy_levels, temperature)
    basis_amp = (eigenvectors.conj() * eigenvectors).real
    return basis_amp.T @ np.tile(probabilities, (1 << num_sites, 1)) @ basis_amp

num_spins = 3
transition_probabilities = transition_matrix(num_spins)
np.savetxt(save_dir + "transition_matrix.txt", transition_probabilities)

# 2. 计算稳态分布
stationary_distribution = null_space(transition_probabilities.T - np.eye(1 << num_spins)).T
stationary_distribution /= stationary_distribution.sum(axis=1, keepdims=True)
np.savetxt(save_dir + "stationary_distribution.txt", stationary_distribution)

# 3. Markov 过程收敛
def markov_iteration(transition, initial_state, max_steps=1000, tolerance=1e-6):
    current_state = initial_state
    for _ in range(max_steps):
        next_state = current_state @ transition
        if norm(next_state - current_state) < tolerance:
            return next_state
        current_state = next_state
    raise ValueError(f"Did not converge in {max_steps} iterations.")

initial_states = [
    (np.zeros(1 << num_spins), "initial_state_1.txt"),
    (np.zeros(1 << num_spins), "initial_state_2.txt"),
    (np.ones(1 << num_spins) / (1 << num_spins), "initial_state_3.txt")
]

initial_states[0][0][0b000] = 1
initial_states[1][0][0b000] = 0.5
initial_states[1][0][0b101] = 0.5

for state, filename in initial_states:
    result = markov_iteration(transition_probabilities, state)
    np.savetxt(save_dir + filename, result)

# 4. 计算 Magnon 变换
def transition_matrix_magnon(num_sites, temperature=1):
    energy_levels = 2 * np.sin(np.pi * np.arange(num_sites) / num_sites) ** 2
    probabilities = boltzmann_distribution(energy_levels, temperature)
    return np.tile(probabilities, (num_sites, 1))

transition_probabilities_magnon = transition_matrix_magnon(num_spins)
np.savetxt(save_dir + "transition_matrix_magnon.txt", transition_probabilities_magnon)

# 5. Magnon 的稳态分布
stationary_distribution_magnon = null_space(transition_probabilities_magnon.T - np.eye(num_spins)).T
stationary_distribution_magnon /= stationary_distribution_magnon.sum(axis=1, keepdims=True)
np.savetxt(save_dir + "stationary_distribution_magnon.txt", stationary_distribution_magnon)

# 6. Magnon 的 Markov 过程
initial_states_magnon = [
    (np.zeros(num_spins), "initial_state_magnon_1.txt"),
    (np.zeros(num_spins), "initial_state_magnon_2.txt"),
    (np.ones(num_spins) / num_spins, "initial_state_magnon_3.txt")
]

initial_states_magnon[0][0][1] = 1
initial_states_magnon[1][0][1] = 0.5
initial_states_magnon[1][0][2] = 0.5

for state, filename in initial_states_magnon:
    result = markov_iteration(transition_probabilities_magnon, state)
    np.savetxt(save_dir + filename, result)

# 7. Master 方程求解
time_steps = 100
rate_matrix = logm(matrix_power(transition_probabilities_magnon, time_steps)) / time_steps
initial_state = np.zeros(num_spins)
initial_state[1] = 1
solution = solve_ivp(lambda t, y: rate_matrix.T @ y, (0, 20), initial_state)

plt.figure(figsize=(8, 6))
for k in range(num_spins):
    plt.plot(solution.t, solution.y[k], label=r"$\pi_{}$".format(k))
plt.xlabel("$t$")
plt.legend()
plt.title("Master Equation Evolution")
plt.savefig(save_dir + "master_equation_evolution.png")
plt.show()