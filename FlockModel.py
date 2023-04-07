import numpy as np
import matplotlib.pyplot as plt
import math
import imageio
import os
import csv
import bisect

def periodic_dist(L,x,y):
    return np.remainder(x - y + L/2, L) - L/2

def do_quiver(positions, velocities,L, type = "Prey"):
    xs = [x[0] for x in positions]
    ys = [y[1] for y in positions]
    us = [u[0] for u in velocities]
    vs = [v[1] for v in velocities]
    if type == "Predator":
        plt.quiver(xs,ys,us,vs,color='r')
    else:
        plt.quiver(xs,ys,us,vs)
    plt.xlim([0,L])
    plt.ylim([0,L])

class Model:
    def __init__(self, dt = 1, density = 1, maxtime = 100, radius = 1, L = 10, noise = 0.05, phenotype = [0,1,0], volume = 0.1, angle = 2*np.pi, predators=1):
        self.dt = dt
        self.density = density
        self.num_agents = int(L**2 * density)
        self.maxtime = maxtime
        self.r = radius # sam fix, put as an input of agents
        self.curr_time = 0
        self.t = [self.curr_time]
        self.L = L
        self.agents = []
        self.volume = volume
        self.cone = np.cos(angle/2)

        for i in range(self.num_agents):

            x = np.random.uniform(0,L,2)
            v = np.random.uniform(-1/2,1/2,2)
            v = v/np.linalg.norm(v)

            self.agents.append(Prey(pos = x, vel = v, parameters = phenotype))

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

    def vel_fluc_plot(self,i=-1):
        positions = [x.pos[i] for x in self.agents if x.type != "Predator"]
        prey_agents_vel = [a.vel[i] for a in self.agents if a.type != "Predator"]
        avg_vel = np.mean(prey_agents_vel,axis=0)
        denom = math.sqrt(sum([np.linalg.norm(v-avg_vel)**2 for v in prey_agents_vel])/len(prey_agents_vel))
        dim_vel = [(v-avg_vel)/denom for v in prey_agents_vel]

        do_quiver(positions, dim_vel, self.L)
        plt.show()

    def quiver_plot(self,i = -1, animate = False, name = None):
        prey_positions = [x.pos[i] for x in self.agents if x.type != "Predator"]
        prey_velocities = [x.vel[i] for x in self.agents if x.type != "Predator"]
        predator_positions = [x.pos[i] for x in self.agents if x.type == "Predator"]
        predator_velocities = [x.vel[i] for x in self.agents if x.type == "Predator"]
        plt.figure()
        do_quiver(prey_positions, prey_velocities,self.L,"Prey")
        do_quiver(predator_positions, predator_velocities, self.L, "Predator")
        if name:
            string = name + r", $\rho$ =" + str(self.density) + ", R =" + str(self.r)
            plt.title(string)
        if animate:
            plt.savefig('data/' + str(i)+'.png')

            plt.close()
        else:
            plt.show()

    def order_plot(self):
        orders = [self.ord(i) for i in range(len(self.t))]
        plt.figure()
        plt.plot(self.t,orders)
        plt.xlabel("Time")
        plt.ylabel("Order")
        plt.show()

    def ord(self,i=-1):
        velocities = [x.vel[i] for x in self.agents if x.type != "Predator"]
        return np.linalg.norm(np.mean(velocities,axis=0))

    def susceptibility(self):
        chis = []
        for i in range(len(self.t)):
            bins,correlation = self.corr(i=i)
            j = 0; chi =0;
            while correlation[j] > 0:
                chi += correlation[j]
                j += 1
            chis.append(chi)
        return chis

    def plot_corr(self,i=-1):
        bins,correlation = self.corr(i=i)
        plt.figure()
        plt.plot(bins, correlation)
        plt.show()

    def corr(self,i=-1,num_bins=20):
        prey_agents = [a for a in self.agents if a.type != "Predator"]
        N = len(prey_agents)
        distances = np.zeros((N,N))
        for j in range(N):
            for k in range(j,N):
                distances[j][k] = np.linalg.norm(periodic_dist(self.L,prey_agents[j].pos[i],prey_agents[k].pos[i]))
                distances[k][j] = distances[j][k]

        max_r = np.max(distances)
        bins = np.linspace(0,max_r,num_bins+1)
        bin_matrix = np.zeros((N,N))
        for j in range(N):
            for k in range(j,N):
                bin_matrix[j][k] = int(np.searchsorted(bins,distances[j][k]))
                bin_matrix[k][j] = bin_matrix[j][k]

        correlation = np.zeros((num_bins+1))

        prey_agents_vel = [a.vel[i] for a in prey_agents]
        avg_vel = np.mean(prey_agents_vel,axis=0    )
        denom = math.sqrt(sum([np.linalg.norm(v-avg_vel)**2 for v in prey_agents_vel])/len(prey_agents_vel))
        dim_vel = [(v-avg_vel)/denom for v in prey_agents_vel]

        # loop through bins
        for b in range(num_bins+1):
            dot_prod = 0
            # dot product everything in bin
            indices = np.where(bin_matrix == b)
            if len(indices[0])>0:
                counter = 0
                for j,x in enumerate(indices[0]):
                    dot_prod += np.dot(dim_vel[x],dim_vel[indices[1][j]])
                    counter += 1

                dot_prod = dot_prod / counter
            else:
                print("No agents in bin")

            correlation[b] = dot_prod

        return bins, correlation

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
