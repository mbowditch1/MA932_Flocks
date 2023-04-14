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


def soft_boundary(x, L, r):
    if x < r:
        return math.cos((x*math.pi)/(2*r))
    elif x > L-r:
        return -1*math.cos(((L-x)*math.pi)/(2*r))
    else:
        return 0


class Cell:
    def __init__(self, i, j, r_hat):
        self.ymin = i * r_hat
        self.ymax = (i+1) * r_hat
        self.xmin = j*r_hat
        self.xmax = (j+1)*r_hat
        self.agents = []


def cell_finder(pos, r_hat):
    return math.floor(pos[0]/r_hat), math.floor(pos[1]/r_hat)


class Model:
    def __init__(self, dt=0.25, density=1, maxtime=50, radius=1, L=10,
                 noise=0.05, phenotype=[0, 1, 0], angle=2*np.pi,
                 predators=0, bc="periodic", exc_r=0.05, br=0.1):
        self.dt, self.curr_time, self.maxtime, self.t = dt, 0, maxtime, [0]
        self.density, self.L = density, L
        self.num_prey, self.num_predators = int(L**2 * density), predators
        self.r, self.cone = radius, np.cos(angle/2)
        self.num_cells = math.floor(self.L/self.r)
        self.r_hat = self.L/self.num_cells
        self.grid = [[Cell(i, j, self.r_hat) for i in range(self.num_cells)]
                     for j in range(self.num_cells)]
        self.agents = []
        self.exc_r = exc_r

        self.br = L*br
        self.bc = bc
        self.periodic = False
        if bc == "periodic":
            self.periodic = True

        for i in range(self.num_prey):
            # Random initial position and normalised velocity
            x = np.random.uniform(0, L, 2)
            v = np.random.uniform(-1/2, 1/2, 2)
            v = v/np.linalg.norm(v)
            agent_i, agent_j = cell_finder(x, self.r_hat)

            # Add prey to cell
            self.grid[agent_i][agent_j].agents.append(
                    Prey(pos=x, vel=v, parameters=phenotype, bc=self.bc, br=self.br))

        for i in range(self.num_predators):

            x = np.random.uniform(0, L, 2)
            v = np.random.uniform(-1/2, 1/2, 2)
            v = v/np.linalg.norm(v)

            # Find cell that prey is contained within
            agent_i = math.floor(x[0]/self.r_hat)
            agent_j = math.floor(x[1]/self.r_hat)

            self.grid[agent_i][agent_j].agents.append(Predator(
                pos=x, vel=v, parameters=phenotype, bc=self.bc))

    def run(self):
        while self.curr_time < self.maxtime:
            self.step()
            self.curr_time += self.dt
            self.t.append(self.curr_time)

        for i in range(self.num_cells):
            for j in range(self.num_cells):
                self.agents += self.grid[i][j].agents

    def step(self):
        index = len(self.t) - 1
        for i in range(self.num_cells):
            for j in range(self.num_cells):
                for a in self.grid[i][j].agents:
                    positions = []
                    velocities = []
                    predator_pos = []
                    for n in range(-1, 2):
                        for m in range(-1, 2):
                            for a_2 in self.grid[(i+n) % self.num_cells][(j+m) % self.num_cells].agents:
                                if a_2.type == "Prey":
                                    positions.append(a_2.pos[index])
                                    velocities.append(a_2.vel[index])
                                elif a_2.type == "Predator":
                                    predator_pos.append(a_2.pos[index])

                    a.update_pos(self.dt, positions, velocities, self.L,
                                 self.r, self.cone, predator_pos, self.exc_r)
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

    def quiver_plot(self, i=-1, animate=False, name=None, ax=None):
        FP.quiver_plot(i, self.L, self.agents, animate, name, self.density, ax)

    def vel_fluc_plot(self, i=-1, ax=None):
        FP.vel_fluc_plot(i, self.L, self.agents, ax)

    def order_plot(self, save=False, title="order_plot", ax=None):
        FP.order_plot(self.agents, self.t, save=save, title=title, ax=ax)

    def corr_plot(self, i=-1, num_bins=20, ax = None):
        FP.corr_plot(i, self.L, self.agents, num_bins, ax)

    def animate(self, name='Gif'):
        FP.animate(self.agents, self.L, name)

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
        db, positions, velocities, pred_pos, pred_vel = self.clustering(i=i, eps=eps, min_samples=min_samples)
        FP.cluster_plot(db, positions=positions, velocities=velocities, pred_pos=pred_pos, pred_vel=pred_vel, i=i, animate=animate, L=self.L)

    def animate_cluster_plot(self, eps=0.3, min_samples=5, name="cluster_gif"):
        entries = os.listdir('data/')
        for filename in entries:
            os.remove('data/' + filename)
        for i in range(len(self.agents[0].pos)):
            self.cluster_plot(i=i, eps=eps, min_samples=min_samples, animate=True)
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

    # EXTRA FUNS
    def ord(self, i=-1):
        return FP.ord(self.agents, i)

    def corr(self, i=-1, num_bins=20):
        return FP.corr(i, self.L, self.agents, num_bins)

    def susceptibility(self, num_bins=20):
        return FP.susceptibility(self.L, self.agents, self.t, num_bins)

    def clustering(self, i=-1, eps=0.3, min_samples=5):
        # Use DBSCAN algorithm to assign points to groups
        positions = []
        velocities = []
        pred_pos = []
        pred_vel = []
        for a in self.agents:
            if a.type == "Prey":
                positions.append(a.pos[i])
                velocities.append(a.vel[i])
            else:
                pred_pos.append(a.pos[i])
                pred_vel.append(a.vel[i])

        db = DBSCAN(eps=1, min_samples=5).fit(positions)

        return db, np.array(positions), np.array(velocities), np.array(pred_pos), np.array(pred_vel)

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
                 noise=0.1, parameters=[0, 1, 0], bc="periodic", br=0.1):
        self.pos = [pos]
        self.vel = [vel]  # do scalar velocity
        self.noise = noise
        self.parameters = parameters
        self.bc = bc
        self.br = br
        self.periodic = False
        if self.bc == "periodic":
            self.periodic = True


class Predator(Agent):
    def __init__(self, pos=np.array([0, 0]), vel=[1, 1],
                 noise=0.1, parameters=[0, 1, 0], bc="periodic", br=1):
        super().__init__(pos, vel, noise, parameters, bc, br)
        self.type = "Predator"

    def update_pos(self, dt, positions, velocities, L, r, cone, predator_pos, exc_r):
        # Predator moves towards nearest prey
        positions = [p for p in positions if not
                     np.array_equal(p, self.pos[-1])]

        # Velocity boundary contribution
        bc_vel = np.zeros(2)
        if self.bc == "soft":
            # Get distance to boundary
            bc_vel[0] = soft_boundary(self.pos[-1][0], L, self.br)
            bc_vel[1] = soft_boundary(self.pos[-1][1], L, self.br)

        if len(positions) > 0:
            nearest_pos = positions[np.argmin([np.linalg.norm(
                disp_finder(L, x, self.pos[-1], self.periodic)) for x in positions])]
            new_vel = disp_finder(L, nearest_pos, self.pos[-1], self.periodic)
        else:
            new_vel = self.vel[-1]

        new_vel = new_vel+(bc_vel)
        self.vel.append(new_vel/np.linalg.norm(new_vel))
        slow_down = 0.9
        self.pos.append(self.pos[-1] + dt*self.vel[-1]*slow_down)


class Prey(Agent):
    def __init__(self, pos=np.array([0, 0]), vel=[1, 1],
                 noise=0.1, parameters=[0, 1, 0], bc="periodic", br=0.1):
        super().__init__(pos, vel, noise, parameters, bc, br=br)
        self.type = "Prey"

    def update_pos(self, dt, positions, velocities, L, r, cone, predator_pos, exc_r):
        # across boundary conditions
        nearby_pos = []
        nearby_vel = []
        nearby_predators = []

        pred_disp = [disp_finder(L, x, self.pos[-1], self.periodic) for x in predator_pos]
        for pd in pred_disp:
            if np.linalg.norm(pd) < r:
                nearby_predators.append(pd)

        # Velocity boundary contribution
        bc_vel = np.zeros(2)
        if self.bc == "soft":
            # Get distance to boundary
            bc_vel[0] = soft_boundary(self.pos[-1][0], L, self.br)
            bc_vel[1] = soft_boundary(self.pos[-1][1], L, self.br)

        collide_pos = []
        # Find nearby prey in vision cone
        for i, x in enumerate(positions):
            vector = disp_finder(L, x, self.pos[-1], self.periodic)
            d = np.linalg.norm(vector)
            if d < r:
                dot = np.dot(self.vel[-1], vector)
                norms = np.linalg.norm(self.vel[-1]) * np.linalg.norm(vector)

                if not norms:
                    nearby_pos.append(np.array(vector))
                    nearby_vel.append(velocities[i])

                elif dot/norms > cone:
                    nearby_pos.append(np.array(vector))
                    nearby_vel.append(velocities[i])

                if d < exc_r:
                    collide_pos.append(np.array(vector))

        pred_vel = np.zeros(2)
        if nearby_predators:
            pred_vel = -sum(nearby_predators)+np.random.normal(0, self.noise, 2)
            pred_vel = pred_vel/np.linalg.norm(pred_vel)

        exclusion = np.zeros(2)
        current = self.vel[-1]
        if len(nearby_vel) > 0:
            align = sum(nearby_vel)/len(nearby_vel)
            centre = sum(nearby_pos)/len(nearby_pos)
            new_vel = self.parameters[0]*current
            new_vel += np.random.normal(0, self.noise, 2)
            new_vel += self.parameters[1]*align+self.parameters[2]*centre
            if len(collide_pos) > 1:
                exclusion = -sum(collide_pos)/len(collide_pos)
                exclusion = exclusion/np.linalg.norm(exclusion)
        else:
            new_vel = current

        new_vel = new_vel/np.linalg.norm(new_vel)
        new_vel = new_vel + pred_vel + bc_vel + exclusion
        if np.linalg.norm(new_vel) > 0:
            self.vel.append(new_vel/np.linalg.norm(new_vel))
        else:
            self.vel.append(new_vel)
        self.pos.append(self.pos[-1] + dt*self.vel[-1])
