import random as rd
import numpy as np
import matplotlib.pyplot as plt
rd.seed(12)
W = 0.5
c1 = 1.6
c2 = 2.4
n_iterations = 1000
n_items = 30
target_error = 1e-6
class Particle():
    def __init__(self):
        x = (-1) ** bool(rd.getrandbits(1)) * rd.random() * 1000
        y = (-1) ** bool(rd.getrandbits(1)) * rd.random() * 1000
        self.position = np.array([x, y])
        self.pBest_position = self.position
        self.pBest_value = float('inf')
        self.velocity = np.array([0,0])

    def update(self):
        self.position = self.position + self.velocity

class Space():
    def __init__(self, target, target_error, n_items):
        self.target = target
        self.target_error = target_error
        self.n_items = n_items
        self.items = []
        self.gBest_value = float('inf')
        self.gBest_position = np.array([rd.random() * 50, rd.random() * 50])
            
    def fitness(self, particle):
        x = particle.position[0]
        y = particle.position[1]
        f =  x**2 + y**2 + 1
        return f
    
    def set_pBest(self):
        for particle in self.items:
            fitness_candidate = self.fitness(particle)
            if(particle.pBest_value > fitness_candidate):
                particle.pBest_value = fitness_candidate
                particle.pBest_position = particle.position
                
    def set_gBest(self):
        for particle in self.items:
            best_fitness_candidate = self.fitness(particle)
            if(self.gBest_value > best_fitness_candidate):
                self.gBest_value = best_fitness_candidate
                self.gBest_position = particle.position
                
    def update_items(self):
        for particle in self.items:
            global W
            inertial = W * particle.velocity
            self_confidence = c1 * rd.random() * (particle.pBest_position - particle.position)
            swarm_confidence = c2 * rd.random() * (self.gBest_position - particle.position)
            new_velocity = inertial + self_confidence + swarm_confidence
            particle.velocity = new_velocity
            particle.update()
            
    def show_items(self, iteration):        
        print(iteration, 'iterations')
        print('BestPosition in this time:', self.gBest_position)
        print('BestValue in this time:', self.gBest_value)
        
        plt.ion()
        for particle in self.items:
            plt.plot(particle.position[0], particle.position[1], 'ro')
            plt.draw()
        plt.plot(self.gBest_position[0], self.gBest_position[1], 'bo')
        plt.title(f'PSO iteration: {iteration}')
        plt.show()
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    def end(self):
        for particle in self.items:
            plt.plot(particle.position[0], particle.position[1], 'ro')
        plt.plot(self.gBest_position[0], self.gBest_position[1], 'bo')
        plt.title(f'Best solution PSO')
        plt.show(block=True)

search_space = Space(1, target_error, n_items)
particle_vector = [Particle() for _ in range(search_space.n_items)]
search_space.items = particle_vector

iteration = 0
while(iteration < n_iterations):
    # set particle best & global best
    search_space.set_pBest()
    search_space.set_gBest()

    # visualization
    search_space.show_items(iteration)
    
    # check conditional
    if(abs(search_space.gBest_value - search_space.target) <= search_space.target_error):
        break
        
    search_space.update_items()
    iteration += 1
    
print("The best solution is: ", search_space.gBest_position, " in ", iteration, " iterations")
search_space.end()