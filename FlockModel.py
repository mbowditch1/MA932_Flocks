import numpy as np
import math
import csv
import FlockPlot as FP


def periodic_dist(L, x, y):
    return np.remainder(x - y + L/2, L) - L/2


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
    def __init__(self, dt=1, density=1, maxtime=100, radius=1, L=10,
                 noise=0.05, phenotype=[0, 1, 0], volume=0.1, angle=2*np.pi,
                 predators=1):
        self.dt, self.curr_time, self.maxtime, self.t = dt, 0, maxtime, [0]
        self.density, self.L = density, L
        self.num_prey, self.num_predators = int(L**2 * density), predators
        self.r, self.volume, self.cone = radius, volume, np.cos(angle/2)
        self.num_cells = math.floor(self.L/self.r)
        self.r_hat = self.L/self.num_cells
        self.grid = [[Cell(i, j, self.r_hat) for i in range(self.num_cells)]
                     for j in range(self.num_cells)]
        self.agents = []

        for i in range(self.num_prey):
            # Random initial position and normalised velocity
            x = np.random.uniform(0, L, 2)
            v = np.random.uniform(-1/2, 1/2, 2)
            v = v/np.linalg.norm(v)
            agent_i, agent_j = cell_finder(x, self.r_hat)

            # Add prey to cell
            self.grid[agent_i][agent_j].agents.append(
                    Prey(pos=x, vel=v, parameters=phenotype))

        for i in range(self.num_predators):

            x = np.random.uniform(0, L, 2)
            v = np.random.uniform(-1/2, 1/2, 2)
            v = v/np.linalg.norm(v)

            # Find cell that prey is contained within
            agent_i = math.floor(x[0]/self.r_hat)
            agent_j = math.floor(x[1]/self.r_hat)

            self.grid[agent_i][agent_j].agents.append(Predator(
                pos=x, vel=v, parameters=phenotype))

    def run(self):
        while self.curr_time < self.maxtime:
            self.step()
            self.curr_time += self.dt
            self.t.append(self.curr_time)

        for i in range(self.num_cells):
            for j in range(self.num_cells):
                self.agents += self.grid[i][j].agents

    def step(self):
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
                                    positions.append(a_2.pos[-1])
                                    velocities.append(a_2.vel[-1])
                                elif a_2.type == "Predator":
                                    predator_pos.append(a_2.pos[-1])

                    a.update_pos(self.dt, positions, velocities, self.L,
                                 self.r, self.cone, predator_pos)
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

    # PLOTTING
    def quiver_plot(self, i=-1, animate=False, name=None):
        FP.quiver_plot(i, self.L, self.agents, animate, name, self.density)

    def vel_fluc_plot(self, i=-1):
        FP.vel_fluc_plot(i, self.L, self.agents)

    def order_plot(self):
        FP.order_plot(self.agents, self.t)

    def corr_plot(self, i=-1, num_bins=20):
        FP.corr_plot(i, self.L, self.agents, num_bins)

    def animate(self, name='Gif'):
        FP.animate(self.agents, self.L, name)

    def sus_plot(self, num_bins=20):
        FP.sus_plot(self.L, self.agents, self.t, num_bins)

    # EXTRA FUNS
    def ord(self, i=-1):
        return FP.ord(self.agents, i)

    def corr(self, i=-1, num_bins=20):
        return FP.corr(i, self.L, self.agents, num_bins)

    def susceptibility(self, num_bins=20):
        return FP.susceptibility(self.L, self.agents, self.t, num_bins)

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
                 noise=0.1, parameters=[0, 1, 0]):
        self.pos = [pos]
        self.vel = [vel]  # do scalar velocity
        self.noise = noise
        self.parameters = parameters


class Predator(Agent):
    def __init__(self, pos=np.array([0, 0]), vel=[1, 1],
                 noise=0.1, parameters=[0, 1, 0]):
        super().__init__(pos, vel, noise, parameters)
        self.type = "Predator"

    def update_pos(self, dt, positions, velocities, L, r, cone, predator_pos):
        # Predator moves towards nearest prey
        positions = [p for p in positions if not
                     np.array_equal(p, self.pos[-1])]
        if len(positions) > 0:
            nearest_pos = positions[np.argmin([np.linalg.norm(
                periodic_dist(L, x, self.pos[-1])) for x in positions])]
            new_vel = periodic_dist(L, nearest_pos, self.pos[-1])
            self.vel.append(new_vel/np.linalg.norm(new_vel))
        else:
            self.vel.append(self.vel[-1])

        slow_down = 0.9
        self.pos.append(self.pos[-1] + dt*self.vel[-1]*slow_down)


class Prey(Agent):
    def __init__(self, pos=np.array([0, 0]), vel=[1, 1],
                 noise=0.1, parameters=[0, 1, 0]):
        super().__init__(pos, vel, noise, parameters)
        self.type = "Prey"

    def update_pos(self, dt, positions, velocities, L, r, cone, predator_pos):
        # across boundary conditions
        nearby_pos = []
        nearby_vel = []
        nearby_predators = []

        pred_disp = [periodic_dist(L, x, self.pos[-1]) for x in predator_pos]
        for pd in pred_disp:
            if np.linalg.norm(pd) < r:
                nearby_predators.append(pd)

        for i, x in enumerate(positions):
            vector = periodic_dist(L, x, self.pos[-1])

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
            new_vel = -sum(nearby_predators)+np.random.normal(0, self.noise, 2)
        else:
            current = self.vel[-1]
            if len(nearby_vel) > 0:
                align = sum(nearby_vel)/len(nearby_vel)
                centre = sum(nearby_pos)/len(nearby_pos)
                new_vel = self.parameters[0]*current
                new_vel += np.random.normal(0, self.noise, 2)
                new_vel += self.parameters[1]*align+self.parameters[2]*centre
            else:
                new_vel = current

        if np.linalg.norm(new_vel) > 0:
            self.vel.append(new_vel/np.linalg.norm(new_vel))
        else:
            self.vel.append(new_vel)
        self.pos.append(self.pos[-1] + dt*self.vel[-1])
