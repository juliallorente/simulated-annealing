import random
import math
import matplotlib.pyplot as plt
import numpy as np

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
    print(num_variables, clauses)
    return num_variables, clauses



# Função que gera uma solução inicial aleatória
def generate_initial_solution(num_variables):
    return [random.choice([True, False]) for _ in range(num_variables)]

# Função para calcular a energia (quantidade de cláusulas não satisfeitas)
def calculate_energy(solution, clauses):
    energy = 0
    for clause in clauses:
        satisfied = False
        for var in clause:
            if solution[abs(var) - 1] == (var > 0):
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
def simulated_annealing(clauses, num_variables, initial_temp=1000, min_temp=0.00001, cooling_rate=0.990, max_iterations=1000):
    current_solution = generate_initial_solution(num_variables)
    current_energy = calculate_energy(current_solution, clauses)
    best_solution = current_solution
    best_energy = current_energy
    
    temperature = initial_temp
    iteration = 0
    energy_history = []  # Para armazenar o histórico da energia durante o processo
    
    print(f"Initial energy: {current_energy}, initial temperature: {temperature}")
    
    while temperature > min_temp and iteration < max_iterations:
        neighbor = get_neighbor(current_solution)
        neighbor_energy = calculate_energy(neighbor, clauses)
        
        # Aceitação da mudança com base na temperatura
        if neighbor_energy < current_energy or random.random() < math.exp((current_energy - neighbor_energy) / temperature):
            current_solution = neighbor
            current_energy = neighbor_energy
            
            # Atualizar a melhor solução encontrada
            if current_energy < best_energy:
                best_solution = current_solution
                best_energy = current_energy
        
        # Resfriamento
        temperature *= cooling_rate
        iteration += 1
        
        # Armazenar a energia a cada iteração
        energy_history.append(current_energy)
        
        # Exibe a temperatura e energia a cada 1000 iterações
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}: energy = {current_energy}, temperature = {temperature}")
    
    return best_solution, best_energy, energy_history

# Função principal para rodar múltiplas execuções
def run_multiple_executions(clauses, num_variables, num_executions=1000):
    best_overall_solution = None
    best_overall_energy = float('inf')
    all_energy_histories = []  
    
    for exec_num in range(num_executions):

        print(f"Execution {exec_num + 1} started...")
        solution, energy, energy_history = simulated_annealing(clauses, num_variables)
        all_energy_histories.append(energy_history)
        
        if energy < best_overall_energy:
            best_overall_solution = solution
            best_overall_energy = energy
        
        print(f"Execution {exec_num + 1} completed with energy: {energy}\n")
    
    # Calcular a média da energia ao longo das execuções
    max_iterations = len(all_energy_histories[0])  # Assume-se que todas as execuções têm o mesmo número de iterações
    mean_energy_history = np.mean(all_energy_histories, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mean_energy_history, label='Média da Energia')
    plt.title('Gráfico de Convergência - Simulated Annealing - 250 clausulas')
    plt.xlabel('Iterações')
    plt.ylabel('Energia (Número de cláusulas não satisfeitas)')
    plt.legend()
    plt.show()
    
    return best_overall_solution, best_overall_energy

filename = '/Users/juliallorente/Documents/IA/atividade-3sat/uf20-01.cnf'  
num_variables, clauses = load_cnf_file(filename)

best_solution, best_energy = run_multiple_executions(clauses, num_variables)

print(f"solução ótima encontrada: {best_solution}")
print(f"Quantidade de cláusulas não satisfeitas: {best_energy}")
