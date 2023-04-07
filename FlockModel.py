import numpy as np
import matplotlib.pyplot as plt
import math
import imageio
import os
import csv
import bisect
import FlockPlot as FP

def periodic_dist(L,x,y):
    return np.remainder(x - y + L/2, L) - L/2

class Model:
    def __init__(self, dt = 1, density = 1, maxtime = 100, radius = 1, L = 10, noise = 0.05, phenotype = [0,1,0], volume = 0.1, angle = 2*np.pi, predators=1):
        self.dt, self.curr_time, self.maxtime, self.t = dt, 0, maxtime, [0]
        self.density, self.L = density, L
        self.num_prey, self.num_predators = int(L**2 * density), predators
        self.r, self.volume, self.cone = radius, volum, np.cos(angle/2) # sam fix, put as an input of agents
        self.agents = []

        for i in range(self.num_prey):

            # Random initial position and normalised velocity
            x = np.random.uniform(0,L,2)
            v = np.random.uniform(-1/2,1/2,2)
            v = v/np.linalg.norm(v)

            self.agents.append(Prey(pos = x, vel = v, parameters = phenotype))

        for i in range(self.num_predators):

            x = np.random.uniform(0,L,2)
            v = np.random.uniform(-1/2,1/2,2)
            v = v/np.linalg.norm(v)
            self.agents.append(Predator(pos = x, vel = v, parameters = phenotype))

    def run(self):
        while self.curr_time < self.maxtime:
            self.step()
            self.curr_time += self.dt
            self.t.append(self.curr_time)

    def step(self):
        positions = [x.pos[-1] for x in self.agents]
        velocities = [x.vel[-1] for x in self.agents]

        for a in self.agents:
            predator_pos = [x.pos[-1] for x in self.agents if x.type == "Predator"]
            a.update_pos(self.dt, positions,velocities,self.L,self.r,self.cone,predator_pos)
            a.pos[-1] = a.pos[-1] % self.L # add other BC later

    # PLOTTING
    def quiver_plot(self,i = -1, animate = False, name = None):
        FP.quiver_plot(i,self.L,self.agents, animate, name, self.density)

    def vel_fluc_plot(self,i=-1):
        FP.vel_fluc_plot(i,self.L,self.agents)

    def order_plot(self):
        FP.order_plot(self.agents,self.t)

    def corr_plot(self,i=-1,num_bins=20):
        FP.corr_plot(i,self.L,self.agents,num_bins)

    def animate(self,name='Gif'):
        FP.animate(self.agents, self.L,name)

    def sus_plot(self,num_bins=20):
        FP.sus_plot(self.L,self.agents,self.t,num_bins)

    # EXTRA FUNS
    def ord(self,i=-1):
        return FP.ord(self.agents,i)

    def corr(self,i=-1,num_bins=20):
        return FP.corr(i,self.L,self.agents,num_bins)

    def susceptibility(self,num_bins=20):
        return FP.susceptibility(self.L,self.agents,self.t,num_bins)


    def blender_csv(self):
        # Create CSV with positions of each agent
        positions = []
        for a in self.agents:
            curr_positions = a.pos
            curr_positions = [[x[0],x[1],0] for x in curr_positions] # Add zero z-value for now
            positions += curr_positions
            positions +=[["n","n","n"]]

        with open("blender_scripts/positions.csv",  "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(positions)


class Agent:
    def __init__(self, pos = np.array([0,0]), vel = [1,1], noise = 0.1, parameters = [0,1,0]):
        self.pos = [pos]
        self.vel = [vel] # do scalar velocity
        self.noise = noise
        self.parameters = parameters

class Predator(Agent):
    def __init__(self, pos = np.array([0,0]), vel = [1,1], noise = 0.1, parameters = [0,1,0]):
        super().__init__(pos, vel, noise, parameters)
        self.type = "Predator"

    def update_pos(self,dt,positions,velocities,L,r,cone,predator_pos):
        # Predator moves towards nearest prey
        positions = [p for p in positions if not np.array_equal(p,self.pos[-1])]
        nearest_pos = positions[np.argmin([np.linalg.norm(periodic_dist(L,x,self.pos[-1])) for x in positions])]
        new_vel = periodic_dist(L,nearest_pos,self.pos[-1])
        self.vel.append(new_vel/np.linalg.norm(new_vel))
        slow_down = 0.9
        self.pos.append(self.pos[-1] + dt*self.vel[-1]*slow_down)

class Prey(Agent):
    def __init__(self, pos = np.array([0,0]), vel = [1,1], noise = 0.1, parameters = [0,1,0]):
        super().__init__(pos, vel, noise, parameters)
        self.type = "Prey"

    def update_pos(self,dt,positions,velocities,L,r,cone,predator_pos):
        # across boundary conditions
        nearby_pos = []
        nearby_vel = []
        nearby_predators = []

        pred_disp = [periodic_dist(L,x,self.pos[-1]) for x in predator_pos]
        for pd in pred_disp:
            if np.linalg.norm(pd) < r: # NOT IN VISION CONE YET
                nearby_predators.append(pd)

        for i, x in enumerate(positions):

            vector = periodic_dist(L,x,self.pos[-1])


            if np.linalg.norm(vector) < r:

                dot = np.dot(self.vel[-1], vector)
                norms = np.linalg.norm(self.vel[-1]) * np.linalg.norm(vector)

                if not norms:
                    nearby_pos.append(np.array(vector))
                    nearby_vel.append(velocities[i])

                elif dot/norms > cone:
                    nearby_pos.append(np.array(vector))
                    nearby_vel.append(velocities[i])


        if nearby_predators:

            new_vel = - sum(nearby_predators) + np.random.normal(0, self.noise, 2)

        else:

            current = self.vel[-1]

            if len(nearby_vel) > 0:

                align = sum(nearby_vel)/len(nearby_vel)

                centre = sum(nearby_pos)/len(nearby_pos)

                new_vel = self.parameters[0]*current + self.parameters[1]*align + self.parameters[2]*centre + np.random.normal(0, self.noise, 2)

            else:
                new_vel = current

        if np.linalg.norm(new_vel) > 0:
            self.vel.append(new_vel/np.linalg.norm(new_vel))
        else:
            self.vel.append(new_vel)
        self.pos.append(self.pos[-1] + dt*self.vel[-1])
