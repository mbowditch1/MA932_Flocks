import numpy as np
import imageio
import math
import csv
from sklearn.cluster import DBSCAN
import os
import FlockPlot as FP
import matplotlib.pyplot as plt


def disp_finder(L, x, y, periodic=True):
    if periodic:
        return np.remainder(x - y + L/2, L) - L/2
    else:
        return x - y


def soft_boundary(x, y, L, r):
    vx = vy = 0
    if x < r:
        vx = math.cos((x*math.pi)/(2*r))
    elif x > L-r:
        vx = -1*math.cos(((L-x)*math.pi)/(2*r))
    if y < r:
        vy = math.cos((y*math.pi)/(2*r))
    elif y > L-r:
        vy = -1*math.cos(((L-y)*math.pi)/(2*r))
    if vx or vy:
        return np.array([vx,vy])
    else:
        return np.zeros(2)


class Cell:
    def __init__(self, i, j, r_hat): #why r_hat and not just r?
        self.ymin = i * r_hat
        self.ymax = (i+1) * r_hat
        self.xmin = j*r_hat
        self.xmax = (j+1)*r_hat
        self.agents = []

def cell_finder(pos, r_hat):
    return math.floor(pos[0]/r_hat), math.floor(pos[1]/r_hat) #int is the same as math.floor

class Model:
    def __init__(self, dt=0.25, prey_density=1, pred_density = 0.01, maxtime=60, prey_radius=1, pred_radius =1, L=15,
                 noise=0, prey_phenotype=[0, 1/2, 1, -5,0] ,pred_phenotype = [1,1,0,0,0,0], prey_angle=2*np.pi, pred_angle=7*np.pi/4,
                bc="periodic", ic = "random", br=0.1, exc_r = 0.4, fixed_vel = True, pred_max = 1.25, prey_max = 1):

        self.dt, self.curr_time, self.maxtime, self.t = dt, 0, maxtime, [0]
        self.L = L
        self.num_prey, self.num_preds = int(L**2 * prey_density), int(L**2 * pred_density)
        self.prey_r, self.pred_r, self.prey_cone, self.pred_cone = prey_radius, pred_radius, np.cos(prey_angle/2), np.cos(pred_angle/2)
        self.num_cells = math.floor(self.L/self.prey_r)
        self.pred_cells = math.ceil(self.pred_r/self.prey_r)
        self.r_hat = self.L/self.num_cells
        self.grid = [[Cell(i, j, self.r_hat) for i in range(self.num_cells)]
                     for j in range(self.num_cells)]
        self.agents = []
        self.exc_r =  exc_r
        self.br = L*br
        self.bc = bc
        self.periodic = False
        if bc == "periodic":
            self.periodic = True

        for i in range(self.num_prey):

            if ic == "school":
                x = np.random.normal(L/3, L/10, 2)%L
                v = np.array([0.0,0.5]) + np.random.normal(0,noise,2)
                if fixed_vel:
                    v = prey_max * v/np.linalg.norm(v)
                else:
                    norm = np.linalg.norm(v)
                    if norm > prey_max:
                        v = prey_max * v/np.linalg.norm(v)
            else:
                # Random initial position and normalised velocity
                x = np.random.uniform(0, L, 2)
                v = np.random.uniform(-1/2, 1/2, 2)
                if fixed_vel:
                    v = prey_max * v/np.linalg.norm(v)
                else:
                    norm = np.linalg.norm(v)
                    if norm > prey_max:
                        v = prey_max * v/np.linalg.norm(v)

            agent_i, agent_j = cell_finder(x, self.r_hat)

            # Add prey to cell
            self.grid[agent_i][agent_j].agents.append(
                    Prey(pos=x, vel=v, parameters = prey_phenotype, bc=self.bc, noise = noise, br=self.br, exc_r = self.exc_r, r = self.prey_r, cone = self.prey_cone, fixed_vel = fixed_vel, max_vel = prey_max))

        for i in range(self.num_preds):
            if ic == "school":
                x = np.random.normal(2*L/3, L/10, 2)%L
                v = np.array([-0.5,0]) + np.random.normal(0,noise,2)
                if fixed_vel:
                    v = pred_max * v/np.linalg.norm(v)
                else:
                    norm = np.linalg.norm(v)
                    if norm > pred_max:
                        v = pred_max * v/np.linalg.norm(v)
            else:
                x = np.random.uniform(0, L, 2)
                v = np.random.uniform(-1/2, 1/2, 2)
                if fixed_vel:
                    v = pred_max * v/np.linalg.norm(v)
                else:
                    norm = np.linalg.norm(v)
                    if norm > pred_max:
                        v = pred_max * v/np.linalg.norm(v)

            # Find cell that prey is contained within
            agent_i = math.floor(x[0]/self.r_hat)
            agent_j = math.floor(x[1]/self.r_hat)

            self.grid[agent_i][agent_j].agents.append(Pred(
                pos=x, vel=v, parameters = pred_phenotype, bc=self.bc, br = self.br, noise = noise, r = self.prey_r, cone = self.prey_cone,  fixed_vel = fixed_vel, max_vel = prey_max))

    def run(self):
        while self.curr_time < self.maxtime:
            self.step()
            self.curr_time += self.dt
            self.t.append(self.curr_time)

        for i in range(self.num_cells):
            for j in range(self.num_cells):
                self.agents += self.grid[i][j].agents

    def step(self):
        """
        Gives each agents all the prey and pred within the adjacent cells
        """
        index = len(self.t) - 1
        for i in range(self.num_cells):
            for j in range(self.num_cells):
                for a in self.grid[i][j].agents:
                    prey_pos = []
                    prey_vel = []
                    pred_pos = []
                    pred_vel = []
                    if a.type == "Prey":
                        for n in range(-1, 2):
                            for m in range(-1, 2):
                                for a_2 in self.grid[(i+n) % self.num_cells][(j+m) % self.num_cells].agents:  # does this work for non-periodic BCs?
                                    if a != a_2:
                                        if a_2.type == "Prey":
                                            prey_pos.append(a_2.pos[index])
                                            prey_vel.append(a_2.vel[index])
                                        elif a_2.type == "Predator":
                                            pred_pos.append(a_2.pos[index])
                                            pred_vel.append(a_2.vel[index])
                    if a.type == "Predator":
                        for n in range(-self.pred_cells,self.pred_cells+1):
                            for m in range(-self.pred_cells,self.pred_cells+1):
                                for a_2 in self.grid[(i+n) % self.num_cells][(j+m) % self.num_cells].agents:  # does this work for non-periodic BCs?
                                    if a != a_2:
                                        if a_2.type == "Prey":
                                            prey_pos.append(a_2.pos[index])
                                            prey_vel.append(a_2.vel[index])
                                        elif a_2.type == "Predator":
                                            pred_pos.append(a_2.pos[index])
                                            pred_vel.append(a_2.vel[index])

                    a.update_pos(self.dt, prey_pos, prey_vel, pred_pos, pred_vel, self.L)

                    a.pos[-1] = a.pos[-1] % self.L  # add other BC later

        for i in range(self.num_cells):
            for j in range(self.num_cells):
                for k, a in enumerate(self.grid[i][j].agents):
                    if not (a.pos[-1][0] > self.grid[i][j].xmin and
                            a.pos[-1][0] < self.grid[i][j].xmax and
                            a.pos[-1][1] < self.grid[i][j].ymax and
                            a.pos[-1][1] > self.grid[i][j].ymin):
                        curr_a = a

                        # Delete agent from current cell
                        self.grid[i][j].agents.pop(k)
                        new_i, new_j = cell_finder(curr_a.pos[-1], self.r_hat)
                        self.grid[new_i][new_j].agents.append(curr_a)

    def quiver_plot(self, i=-1, animate=False, title=None, ax=None):
        FP.quiver_plot(i, self.L, self.agents, animate, title, ax)

    def spatial_distribution_plot(self,i=-1, frames = 1, save = False, ax = None , title = "Clustering Plot"):
        FP.spatial_distribution_average(i, self.L, self.agents, frames = frames, ax = ax, periodic = self.periodic, save = save, title=title)

    def vel_fluc_plot(self, i=-1, ax=None):
        FP.vel_fluc_plot(i, self.L, self.agents, ax)

    def order_plot(self, save=False, title="order_plot", ax=None):
        FP.order_plot(self.agents, self.t, save=save, title=title, ax=ax)

    def corr_plot(self, i=-1, num_bins=20, ax = None):
        FP.corr_plot(i, self.L, self.agents, num_bins, ax)

    def animate(self, title='Gif'):
        FP.animate(self.agents, self.L, title)

    def sus_plot(self, num_bins=20, ax=None):
        FP.sus_plot(self.L, self.agents, self.t, num_bins, ax)

    def groups_plot(self, eps=0.3, min_samples=5, ax=None):
        num_groups = np.zeros(len(self.t))
        for i in range(len(num_groups)):
            db = self.clustering(i=i, eps=eps, min_samples=min_samples)[0]
            labels = db.labels_
            num_groups[i] = len(set(labels)) - (1 if -1 in labels else 0)
        ax = ax or plt.gca()
        ax.plot(self.t,num_groups)

    def cluster_plot(self, i=-1, eps=0.3, min_samples=5, animate=False):
        db, prey_pos, prey_vel, pred_pos, pred_vel = self.clustering(i=i, eps=eps, min_samples=min_samples)
        FP.cluster_plot(db, prey_pos, prey_vel, pred_pos, pred_vel, i=i, animate=animate, L=self.L)

    def animate_cluster_plot(self, eps=0.3, min_samples=5, title="cluster_gif"):
        entries = os.listdir('data/')
        for filename in entries:
            os.remove('data/' + filename)
        for i in range(len(self.agents[0].pos)):
            self.cluster_plot(i=i, eps=eps, min_samples=min_samples, animate=True)
        entries = os.listdir('data/')
        entries = [int(x[:-4]) for x in entries]
        entries.sort()
        with imageio.get_writer(title + ".gif", mode='I') as writer:
            for i, filename in enumerate(entries):
                if i == 0:
                    for j in range(4):
                        image = imageio.imread('data/' + str(filename) + '.png')
                        writer.append_data(image)
                image = imageio.imread('data/' + str(filename) + '.png')
                writer.append_data(image)

    # EXTRA FUNS
    def ord(self, i=-1):
        return FP.ord(self.agents, i)

    def corr(self, i=-1, num_bins=20):
        return FP.corr(i, self.L, self.agents, num_bins)

    def susceptibility(self, num_bins=20):
        return FP.susceptibility(self.L, self.agents, self.t, num_bins)

    def clustering(self, i=-1, eps=0.3, min_samples=5):
        # Use DBSCAN algorithm to assign points to groups
        prey_pos = []
        prey_vel = []
        pred_pos = []
        pred_vel = []
        for a in self.agents:
            if a.type == "Prey":
                prey_pos.append(a.pos[i])
                prey_vel.append(a.vel[i])
            else:
                pred_pos.append(a.pos[i])
                pred_vel.append(a.vel[i])

        db = DBSCAN(eps=1, min_samples=5).fit(prey_pos)

        return db, prey_pos, prey_vel, pred_pos, pred_vel

    def blender_csv(self):
        # Create CSV with positions of each agent
        positions = []
        for a in self.agents:
            curr_positions = a.pos
            curr_positions = [[x[0], x[1], 0] for x in curr_positions]
            positions += curr_positions
            positions += [["n", "n", "n"]]

        with open("blender_scripts/positions.csv",  "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(positions)


class Agent:
    def __init__(self, pos=np.array([0, 0]), vel=[1, 1],
                 noise=0.1, parameters=[0, 1, 0], bc="periodic", br=1, r = 1, cone = -1, fixed_vel = True, max_vel = 1):
        self.fixed_vel = fixed_vel
        self.pos = [pos]
        self.vel = [vel]  # do scalar velocity
        self.noise = noise
        self.parameters = parameters
        self.bc = bc
        self.br = br
        self.cone =cone
        self.r = r
        self.periodic = False
        if self.bc == "periodic":
            self.periodic = True
        self.max_vel = max_vel
        self.fixed_vel = fixed_vel


class Pred(Agent):
    def __init__(self, pos=np.array([0, 0]), vel=[1, 1],
                 noise=0.1, parameters=[1, 2, 0, 1, -1/2, 1/2], bc="periodic", br=1, r = 1, cone = -1, fixed_vel = True, max_vel = 1.2):
        super().__init__(pos, vel, noise, parameters, bc, br, r, cone, fixed_vel, max_vel)
        self.type = "Predator"

    def update_pos(self, dt, prey_pos, prey_vel, pred_pos, pred_vel, L):
        directions = np.zeros((2,7))

        directions[:,0] = self.vel[-1]

        #find the other prey in our visions cone
        preyVC_pos = []
        preyVC_vel = []

        for i, x in enumerate(prey_pos):
            vector = disp_finder(L, x, self.pos[-1], self.periodic)
            d = np.linalg.norm(vector)

            if d < self.r:
                dot = np.dot(self.vel[-1], vector)
                norms = np.linalg.norm(self.vel[-1]) * np.linalg.norm(vector)

                if dot >= self.cone * norms:
                    preyVC_pos.append(vector)
                    preyVC_vel.append(prey_vel[i])

        # if there is other prey, calculate the directions to the centre and the alignment
        if len(preyVC_pos) > 0:
            directions[:,1] = sum(preyVC_pos)/ (len(preyVC_pos) * self.r)
            directions[:,2]= sum(preyVC_vel)/ len(preyVC_vel)
            directions[:,3] = preyVC_pos[np.argmin([np.linalg.norm(disp_finder(L, x, self.pos[-1], self.periodic)) for x in preyVC_pos])]

        #find the pred in our visions clen(preyVC_pos) >one

        if self.parameters[4] or self.parameters[5]:

            predVC_pos = []
            predVC_vel = []

            for i, x in enumerate(pred_pos):
                vector = disp_finder(L, x, self.pos[-1], self.periodic)
                d = np.linalg.norm(vector)
                if d < self.r:
                    dot = np.dot(self.vel[-1], vector)
                    norms = np.linalg.norm(self.vel[-1]) * np.linalg.norm(vector)

                    if dot >= self.cone * norms:
                        predVC_pos.append(vector)
                        predVC_vel.append(pred_vel[i])

            # if there is predators, calculate the directions to the centre and the alignment
            if len(predVC_pos) > 0:
                directions[:,4] = sum(predVC_pos)/ len(predVC_pos)
                directions[:,5]= sum(predVC_vel)/ len(predVC_vel)

        # get the boundary condition forces

        if self.bc == "soft":
            # Get distance to boundary
            directions[:,6] = soft_boundary(self.pos[-1][0], self.pos[-1][1], L, self.br)

        new_vel = np.zeros(2)
        for i, value in enumerate(self.parameters):
            new_vel += value * directions[:,i]

        new_vel += np.linalg.norm(self.parameters, ord=1) * directions[:,-1]

        if not np.linalg.norm(new_vel):
            new_vel = self.vel[-1]

        if not self.fixed_vel:
            norm = np.linalg.norm(new_vel)
            if norm > self.max_vel:
                new_vel = self.max_vel* new_vel/np.linalg.norm(new_vel)
            new_vel += np.random.normal(0,self.noise,2)
            norm = np.linalg.norm(new_vel)
            if norm >self.max_vel:
                new_vel = self.max_vel * new_vel/np.linalg.norm(new_vel)
        else:
            new_vel = self.max_vel * new_vel/np.linalg.norm(new_vel)
            new_vel += np.random.normal(0,self.noise,2)
            new_vel = self.max_vel* new_vel/np.linalg.norm(new_vel)


        self.vel.append(new_vel)

        self.pos.append(self.pos[-1] + dt*self.vel[-1])


class Prey(Agent):
    def __init__(self, pos=np.array([0, 0]), vel=[1, 1],
                 noise=0.1, parameters=np.array([1,-1,3,-5,-5,0]), bc="periodic", br=0.1, r = 1, cone = -1, exc_r = 0.4, fixed_vel = True, max_vel = 1):
        super().__init__(pos, vel, noise, parameters, bc, br, r, cone, fixed_vel, max_vel)
        self.type = "Prey"
        self.exc_r = exc_r

    def update_pos(self, dt, prey_pos, prey_vel, pred_pos, pred_vel, L):
        directions = np.zeros((2,6))
        directions[:,0] = self.vel[-1]

        #find the other prey in our visions cone
        preyVC_pos = []
        preyVC_vel = []

        for i, x in enumerate(prey_pos):
            vector = disp_finder(L, x, self.pos[-1], self.periodic)
            d = np.linalg.norm(vector)

            if d < self.r:
                dot = np.dot(self.vel[-1], vector)
                norms = np.linalg.norm(self.vel[-1]) * np.linalg.norm(vector)

                if dot >= self.cone * norms:
                    preyVC_pos.append(vector*(-1*np.power(np.log(d/self.r)/np.log(self.exc_r/self.r),10) + 1))
                    preyVC_vel.append(prey_vel[i])

        # if there is other prey, calculate the directions to the centre and the alignment
        if len(preyVC_pos) > 0:
            directions[:,1] = sum(preyVC_pos)/(len(preyVC_pos)*self.r)
            directions[:,2] = sum(preyVC_vel)/ len(preyVC_vel)

        #find the pred in our visions cone

        if self.parameters[3] or self.parameters[4]:

            predVC_pos = []
            predVC_vel = []

            for i, x in enumerate(pred_pos):
                vector = disp_finder(L, x, self.pos[-1], self.periodic)
                d = np.linalg.norm(vector)
                if d < self.r:
                    dot = np.dot(self.vel[-1], vector)
                    norms = np.linalg.norm(self.vel[-1]) * np.linalg.norm(vector)
                    if dot >= self.cone * norms:
                        predVC_pos.append(vector)
                        predVC_vel.append(pred_vel[i])

            # if there is predators, calculate the directions to the centre and the alignment
            if len(predVC_pos) > 0:
                directions[:,3] = sum(predVC_pos)/ len(predVC_pos)
                directions[:,4]= sum(predVC_vel)/ len(predVC_vel)


        # get the boundary condition forces

        if self.bc == "soft":
            directions[:,5] = soft_boundary(self.pos[-1][0], self.pos[-1][1], L, self.br)

        new_vel = np.zeros(2)
        for i, value in enumerate(self.parameters):
            new_vel += value * directions[:,i]

        new_vel += np.linalg.norm(self.parameters, ord=1) * directions[:,-1]

        if not np.linalg.norm(new_vel):
            new_vel = self.vel[-1]

        if not self.fixed_vel:
            norm = np.linalg.norm(new_vel)
            if norm > self.max_vel:
                new_vel = self.max_vel * new_vel/np.linalg.norm(new_vel)
            new_vel += np.random.normal(0,self.noise,2)
            norm = np.linalg.norm(new_vel)
            if norm > self.max_vel:
                new_vel = self.max_vel * new_vel/np.linalg.norm(new_vel)
        else:
            new_vel = self.max_vel * new_vel/np.linalg.norm(new_vel)
            new_vel += np.random.normal(0,self.noise,2)
            new_vel = self.max_vel * new_vel/np.linalg.norm(new_vel)

        self.vel.append(new_vel)

        self.pos.append(self.pos[-1] + dt*self.vel[-1])
