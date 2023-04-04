import numpy as np
import matplotlib.pyplot as plt
import math
import imageio
import os
import csv

class Model:
    def __init__(self, dt = 1, density = 1, maxtime = 100, radius = 1, L = 10, noise = 0.05, phenotype = None, volume = 0.1, angle = 2*np.pi, predators=1):
        self.dt = dt
        self.density = density
        self.num_agents = int(L**2 * density)
        self.maxtime = maxtime
        self.r = radius # sam fix, put as an input of agents
        self.curr_time = 0
        self.L = L
        self.agents = []
        self.volume = volume
        self.cone = np.cos(angle/2)

        for i in range(self.num_agents):

            x = np.random.uniform(0,L,2)
            v = np.random.uniform(-1/2,1/2,2)
            v = v/np.linalg.norm(v)

            if phenotype == "1":
                p = {"Current":1 + self.dt, "Align" :self.dt, "Centre":0}

            elif phenotype == "2":
                p = {"Current":1 + self.dt, "Align" :0, "Centre":self.dt}

            elif phenotype == "3":
                p = {"Current":1 + self.dt, "Align" :self.dt, "Centre":self.dt}

            elif phenotype == "4":
                p = {"Current":1 + self.dt, "Align" :2 * self.dt, "Centre":self.dt}

            elif phenotype == "5":
                p = {"Current":1, "Align" :self.dt, "Centre":0}

            elif phenotype == "6":
                p = {"Current":1 , "Align" :0, "Centre":self.dt}

            elif phenotype == "7":
                p = {"Current":1, "Align" :self.dt, "Centre":self.dt}

            elif phenotype == "8":
                p = {"Current":1, "Align" :2 * self.dt, "Centre":self.dt}

            elif phenotype == "Viscek":
                p = {"Current":0, "Align" : 1, "Centre": 0}
            else:
                p = {"Current":1 + self.dt, "Align" : np.random.uniform(0,1), "Centre":  np.random.uniform(-1,1)}

            self.agents.append(Prey(pos = x, vel = v, parameters = p))

        self.agents.append(Predator(pos = x, vel = v, parameters = p))

    def run(self):
        while self.curr_time < self.maxtime:
            self.step()
            self.curr_time += self.dt

    def step(self):
        positions = [x.pos[-1] for x in self.agents]
        velocities = [x.vel[-1] for x in self.agents]

        for a in self.agents:
            a.update_pos(self.dt, positions,velocities,self.L,self.r,self.cone)
            a.pos[-1] = a.pos[-1] % self.L # add other BC later

    def quiver_plot(self,i = -1, animate = False, name = None):
        positions = [x.pos[i] for x in self.agents]
        velocities = [x.vel[i] for x in self.agents]
        xs = [x[0] for x in positions]
        ys = [y[1] for y in positions]
        us = [u[0] for u in velocities]
        vs = [v[1] for v in velocities]
        plt.figure()
        plt.quiver(xs,ys,us,vs)
        plt.xlim([0,self.L])
        plt.ylim([0,self.L])
        if name:
            string = name + r", $\rho$ =" + str(self.density) + ", R =" + str(self.r)
            plt.title(string)
        if animate:
            plt.savefig('data/' + str(i)+'.png')

            plt.close()
        else:
            plt.show()

    def animate(self, name = "Gif"):
        entries = os.listdir('data/')
        for filename in entries:
            os.remove('data/' + filename)
        for i in range(len(self.agents[0].pos)):
            self.quiver_plot(i, True, name)
        entries = os.listdir('data/')
        entries = [int(x[:-4]) for x in entries]
        entries.sort()
        with imageio.get_writer(name + ".gif", mode='I') as writer:
            for i, filename in enumerate(entries):
                if i == 0:
                    for j in range(4):
                        image = imageio.imread('data/' + str(filename) + '.png')
                        writer.append_data(image)
                image = imageio.imread('data/' + str(filename) + '.png')
                writer.append_data(image)

    def order(self):
        return

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
    def __init__(self, pos = np.array([0,0]), vel = [1,1], noise = 0.1, parameters = {"Current": 1, "Align" : 0, "Centre": 0}):
        self.pos = [pos]
        self.vel = [vel] # do scalar velocity
        self.noise = noise
        self.parameters = parameters

class Predator(Agent):
    def __init__(self, pos = np.array([0,0]), vel = [1,1], noise = 0.1, parameters = {"Current": 1, "Align" : 0, "Centre": 0}):
        super().__init__(pos, vel, noise, parameters)
        self.type = "Predator"

    def update_pos(self,dt,positions,velocities,L,r,cone):
        # Predator moves towards nearest prey
        positions = [p for p in positions if not np.array_equal(p,self.pos[-1])]
        nearest_pos = positions[np.argmin([np.linalg.norm(np.remainder(x - self.pos[-1] + L/2, L) - L/2) for x in positions])]
        new_vel = np.remainder(nearest_pos - self.pos[-1] + L/2, L) - L/2
        self.vel.append(new_vel/np.linalg.norm(new_vel))
        self.pos.append(self.pos[-1] + dt*self.vel[-1])

class Prey(Agent):
    def __init__(self, pos = np.array([0,0]), vel = [1,1], noise = 0.1, parameters = {"Current": 1, "Align" : 0, "Centre": 0}):
        super().__init__(pos, vel, noise, parameters)
        self.type = "Prey"

    def update_pos(self,dt,positions,velocities,L,r,cone):
        # across boundary conditions
        nearby_pos = []
        nearby_vel = []

        for i, x in enumerate(positions):

            vector = np.remainder(x - self.pos[-1] + L/2, L) - L/2

            if np.linalg.norm(vector) < r:

                dot = np.dot(self.vel[-1], vector)
                norms = np.linalg.norm(self.vel[-1]) * np.linalg.norm(vector)

                if not norms:
                    nearby_pos.append(np.array(vector))
                    nearby_vel.append(velocities[i])

                elif dot/norms > cone:
                        nearby_pos.append(np.array(vector))
                        nearby_vel.append(velocities[i])

        current = self.vel[-1]

        if len(nearby_vel) > 0:

            align = sum(nearby_vel)/len(nearby_vel)

            centre = sum(nearby_pos)/len(nearby_pos)

            new_vel = self.parameters["Current"] * current + self.parameters["Align"] * align + self.parameters["Centre"] * centre + np.random.normal(0, self.noise, 2)

        else:
            new_vel = current

        if np.linalg.norm(new_vel) > 0:
            self.vel.append(new_vel/np.linalg.norm(new_vel))
        else:
            self.vel.append(new_vel)
        self.pos.append(self.pos[-1] + dt*self.vel[-1])
