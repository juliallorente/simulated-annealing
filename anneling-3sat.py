import random
import math
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

# Função para carregar o arquivo CNF e extrair as cláusulas
def load_cnf_file(filename):
    clauses = []
    num_variables = 0
    
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('c') or line.startswith('%'):
                continue
            elif line.startswith('p cnf'):
                parts = line.split()
                num_variables = int(parts[2])
            elif line.strip() != "":
                clause = list(map(int, line.split()))
                clause.pop()  # Remove o 0 no final de cada cláusula
                clauses.append(clause)
    return num_variables, [c for c in clauses if len(c) == 3]

# Função que gera uma solução inicial aleatória
def generate_initial_solution(num_variables):
    return [random.choice([True, False]) for _ in range(num_variables)]

# Função para calcular a energia (quantidade de cláusulas não satisfeitas)
def calculate_energy(solution, clauses):
    energy = 0
    for clause in clauses:
        satisfied = False
        for var in clause:
            var_index = abs(var) - 1  # Corrigido o índice
            # Verifique se a variável é verdadeira ou falsa conforme a cláusula
            if solution[var_index] == (var > 0):  # Se a variável for verdadeira
                satisfied = True
                break
        if not satisfied:
            energy += 1
    return energy

# Função para gerar um novo estado vizinho (invertendo uma variável aleatória)
def get_neighbor(solution):
    neighbor = solution[:]
    index = random.randint(0, len(solution) - 1)
    neighbor[index] = not neighbor[index]  # Inverte o valor do bit
    return neighbor

# Função de Simulated Annealing
def simulated_annealing(clauses, num_variables, initial_temp=100000, min_temp=0.00001, cooling_rate=0.990, sa_max=30):
    current_solution = generate_initial_solution(num_variables)
    current_energy = calculate_energy(current_solution, clauses)
    best_solution = current_solution
    best_energy = current_energy
    
    temperature = initial_temp
    energy_history = []  # Para armazenar o histórico da energia durante o processo
    
    print(f"Initial energy: {current_energy}, initial temperature: {temperature}")
    
    while temperature > min_temp:
        iteration = 0
        
        while iteration < sa_max:
            neighbor = get_neighbor(current_solution)
            neighbor_energy = calculate_energy(neighbor, clauses)
            delta_energy = neighbor_energy - current_energy
            
            if delta_energy < 0:
            
                current_solution = neighbor
                current_energy = neighbor_energy

                if current_energy < best_energy:
                    best_solution = current_solution
                    best_energy = current_energy
            elif random.random() < math.exp((-delta_energy) / temperature):
                    current_solution = neighbor
                    current_energy = neighbor_energy

            iteration += 1
        
        # Resfriamento
        temperature *= cooling_rate
        
        # Armazenar a energia a cada iteração
        energy_history.append(current_energy)
    
    return best_solution, best_energy, energy_history

def run_single_execution(exec_num, clauses, num_variables):
    print(f"Execution {exec_num + 1} started...")
    solution, energy, energy_history = simulated_annealing(clauses, num_variables)
    print(f"Execution {exec_num + 1} completed with energy: {energy}\n")
    return solution, energy, energy_history

def run_multiple_executions_parallel(clauses, num_variables, num_executions=30, n_jobs=-1):
    # Execuções paralelas
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_execution)(i, clauses, num_variables)
        for i in range(num_executions)
    )

    best_overall_solution = None
    best_overall_energy = float('inf')
    all_energy_histories = []

    for solution, energy, energy_history in results:
        all_energy_histories.append(energy_history)
        if energy < best_overall_energy:
            best_overall_solution = solution
            best_overall_energy = energy

    # Processamento do histórico de energia
    sa_max = max(len(history) for history in all_energy_histories)
    padded_histories = []

    for history in all_energy_histories:
        if len(history) < sa_max:
            history.extend([history[-1]] * (sa_max - len(history)))
        padded_histories.append(history)

    mean_energy_history = np.mean(padded_histories, axis=0)

    # Plot do gráfico de convergência
    plt.figure(figsize=(10, 6))
    plt.plot(mean_energy_history, label='Média da Energia')
    plt.title('Gráfico de Convergência - Simulated Annealing - 250 cláusulas')
    plt.xlabel('Iterações')
    plt.ylabel('Energia (Número de cláusulas não satisfeitas)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_overall_solution, best_overall_energy


filename = 'uf250-01.cnf'  
num_variables, clauses = load_cnf_file(filename)

best_solution, best_energy = run_multiple_executions_parallel(clauses, num_variables)

print(f"solução ótima encontrada: {best_solution}")
print(f"Quantidade de cláusulas não satisfeitas: {best_energy}")
