import numpy as np
import matplotlib.pyplot as plt
import math
import imageio
import os


# A file to handle the bulk of the plotting functions
def periodic_dist(L, x, y):
    return np.remainder(x - y + L/2, L) - L/2


# Needed to make plots with quivers
def do_quiver(i, L, agents=None, positions=None, velocities=None, ax=None):
    if (positions is not None and velocities is None) or (positions is None and velocities is not None):
        print('Please provide both positions and velocities for quiver plot')
    elif positions is not None and velocities is not None:
        xs, ys = [x[0] for x in positions], [y[1] for y in positions]
        us, vs = [u[0] for u in velocities], [v[1] for v in velocities]
        cols = ['k' for x in xs]
    else:
        xs, ys = [x.pos[i][0] for x in agents], [x.pos[i][1] for x in agents]
        us, vs = [x.vel[i][0] for x in agents], [x.vel[i][1] for x in agents]
        cols = ['r' if x.type == "Predator" else "k" for x in agents]

    ax = ax or plt.gca()
    ax.quiver(xs, ys, us, vs, color=cols, scale = L)
    ax.set_xlim([0, L])
    ax.set_ylim([0, L])


# Call a quiver plot of agents
def quiver_plot(i, L, agents, save=False, title=None, ax=None):
    if ax:
        do_quiver(i, L, agents, ax=ax)
        return
    plt.figure(figsize=(12, 12))
    do_quiver(i, L, agents)
    if title:
        plt.title(title)
    if save:
        plt.savefig('data/' + str(i)+'.png')
        plt.close()
    else:
        plt.show()


def group_plot(db, prey_pos, prey_vel, pred_pos, pred_vel, i, L, animate=False):
    labels = db.labels_
    # Number of groups in labels, ignoring noise if present.
    n_groups_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    plt.figure(figsize=(24, 16))
    ax = plt.gca()
    ax.set_facecolor('k')
    colors = [plt.cm.cool(each) for each in np.linspace(0, 1, len(unique_labels))]
    colors.append([1, 1, 1, 1])
    prey_colors = [colors[z] for z in labels]
    for z in range(len(pred_vel)):
        prey_colors.append([1, 0, 0, 1])
    positions = np.vstack((prey_pos, pred_pos))
    velocities = np.vstack((prey_vel, pred_vel))
    plt.quiver(
        positions[:, 0],
        positions[:, 1],
        velocities[:, 0],
        velocities[:, 1],
        color=prey_colors,
        headaxislength = 0.1
    )

    plt.title(f"Estimated number of groups: {n_groups_}")
    plt.xlim((0, L))
    plt.ylim((0, L))
    if animate:
        plt.savefig('data/' + str(i) + '.png')
        plt.close()
    else:
        plt.show()


# Animating quiver_plot
def animate(agents, L, title="Gif"):
    entries = os.listdir('data/')
    for filename in entries:
        os.remove('data/' + filename)
    for i in range(len(agents[0].pos)):
        quiver_plot(i, L, agents, True, title)
    entries = os.listdir('data/')
    entries = [int(x[:-4]) for x in entries]
    entries.sort()
    with imageio.get_writer('gifs/' + title + '.gif', mode='I') as writer:
        for i, filename in enumerate(entries):
            if i == 0:
                for j in range(4):
                    image = imageio.imread('data/' + str(filename) + '.png')
                    writer.append_data(image)
            image = imageio.imread('data/' + str(filename) + '.png')
            writer.append_data(image)


# Velocity fluctuation plot
def vel_fluc_plot(i, L, agents, ax = None):
    positions = [x.pos[i] for x in agents if x.type != "Predator"]  # Fast
    prey_agents_vel = [a.vel[i] for a in agents if a.type != "Predator"]
    avg_vel = np.mean(prey_agents_vel, axis=0)
    denom = math.sqrt(sum([np.linalg.norm(v-avg_vel)**2 for v in prey_agents_vel])/len(prey_agents_vel))
    dim_vel = [(v-avg_vel)/denom for v in prey_agents_vel]
    if ax:
        do_quiver(i, L, agents, ax=ax)
        return
    plt.figure()
    do_quiver(i, L, None, positions, dim_vel)
    plt.show()

def spatial_distribution(i, L, agents, periodic=True, n=20):
    rs = np.linspace(0,L,n)
    K = np.zeros(n)
    prey = [x for x in agents if x.type != "Predator"]  # Fast
    N = len(prey)
    for j in range(N):
        for k in range(j, N):
            if j != k:
                d = np.linalg.norm(disp_finder(L, prey[j].pos[i], prey[k].pos[i]), periodic)
                bools = [d < r for r in rs]
                index = sum(bools)
                K[-index:] += 2
    K = K * L**2 / (N * (N-1) * np.pi)
    K = np.power(K, 1/2) - rs
    return K

def spatial_distribution_average(i, L, agents,  frames = 1, periodic=True, n=20, prey_radius = 1):
    rs = np.linspace(0,L,n)
    K = np.zeros(n)
    for j in range(frames):
        K += spatial_distribution(i-j, L, agents, n=n)
    K = K/frames
    flock_size = rs[np.argmax(K)]
    exclusion_radius = rs[np.argmin(K[:math.ceil(prey_radius/(L/n))])]
    return rs, K, flock_size, exclusion_radius

def clustering(i, L, agents, frames=1, periodic=True, n=20, ax = None, save = False, title = "Clustering Plot", prey_radius = 1):
    rs, K, flock_size, exclusion_radius  = spatial_distribution_average(i,L,agents,frames,periodic,n, prey_radius)
    ax = ax or plt.gca()
    ax.plot(rs, K)
    ax.set_xticks = [rs[np.argmax(K)]]
    ax.set_yticks = [0, np.argmax(K)]
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\langle \hat{L}(r)\rangle$")
    ax.set_ylim(1.05*min(K), -1.05*min(K))
    ax.set_title(title)
    if save:
        ax.figure.savefig('figures/' + title + ".png")
        plt.close()
    else:
        plt.show()
    return flock_size, exclusion_radius

def disp_finder(L, x, y, periodic=True):
    if periodic:
        return np.remainder(x - y + L/2, L) - L/2
    else:
        return x - y

# Need for order plot
def ord(agents, i=-1):
    velocities = [x.vel[i]/np.linalg.norm(x.vel[i]) for x in agents if x.type != "Predator"]
    return np.linalg.norm(np.mean(velocities, axis=0))

# Order plot
def order_plot(agents, ts, save=False, title="Order Plot", ax = None):
    orders = [ord(agents, i) for i in range(len(ts))]
    if ax:
        ax.plot(ts,orders)
        return

    plt.figure()
    plt.figure(figsize=(24, 16))
    plt.plot(ts, orders)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Order")
    plt.ylim(0,1)
    if save:
        plt.savefig('figures/' + title + ".png")
        plt.close()
    else:
        plt.show()

def vel(agents, i=-1):
    velocities = [np.linalg.norm(x.vel[i]) for x in agents if x.type != "Predator"]
    return np.mean(velocities, axis=0)

# Order plot
def vel_plot(agents, ts, save=False, title="Velocity Plot", ax = None):
    velocities = [vel(agents, i) for i in range(len(ts))]
    if ax:
        ax.plot(ts,velocities)
        return



    plt.figure()
    plt.figure(figsize=(24, 16))
    plt.plot(ts, velocities)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Average Velocity")
    plt.ylim(0,1)
    if save:
        plt.savefig('figures/' + title + ".png")
        plt.close()
    else:
        plt.show()


# To corr_plot
def corr(i, L, agents, num_bins=20):
    prey_agents = [a for a in agents if a.type != "Predator"]
    N = len(prey_agents)
    distances = np.zeros((N, N))
    for j in range(N):
        for k in range(j, N):
            distances[j][k] = np.linalg.norm(periodic_dist(L, prey_agents[j].pos[i], prey_agents[k].pos[i]))
            distances[k][j] = distances[j][k]

    max_r = np.max(distances)
    bins = np.linspace(0, max_r, num_bins+1)
    bin_matrix = np.zeros((N, N))
    for j in range(N):
        for k in range(j, N):
            bin_matrix[j][k] = int(np.searchsorted(bins, distances[j][k]))
            bin_matrix[k][j] = bin_matrix[j][k]

    correlation = np.zeros((num_bins+1))

    prey_agents_vel = [a.vel[i] for a in prey_agents]
    avg_vel = np.mean(prey_agents_vel, axis=0)
    denom = math.sqrt(sum([np.linalg.norm(v-avg_vel)**2 for v in prey_agents_vel])/len(prey_agents_vel))
    dim_vel = [(v-avg_vel)/denom for v in prey_agents_vel]

    # loop through bins
    for b in range(num_bins+1):
        dot_prod = 0
        # dot product everything in bin
        indices = np.where(bin_matrix == b)
        if len(indices[0]) > 0:
            counter = 0
            for j, x in enumerate(indices[0]):
                dot_prod += np.dot(dim_vel[x], dim_vel[indices[1][j]])
                counter += 1

            dot_prod = dot_prod / counter

        correlation[b] = dot_prod

    return bins, correlation


# Corr plot
def corr_plot(i, L, agents, num_bins, ax = None):
    bins, correlation = corr(i, L, agents, num_bins)
    if ax:
        ax.plot(bins, correlation)
        return
    plt.figure()
    plt.plot(bins, correlation)
    plt.show()

# Chi stuff
def susceptibility(L, agents, ts, num_bins=20):
    chis = []
    for i in range(len(ts)):
        bins, correlation = corr(i, L, agents, num_bins)
        j = 0
        chi = 0
        while correlation[j] > 0:
            chi += correlation[j]
            j += 1
        chis.append(chi)
    return chis


def sus_plot(L, agents, ts, num_bins=20,ax=None):
    chis = susceptibility(L, agents, ts, num_bins=20)
    if ax:
        ax.plot(ts, chis)
        return
    plt.figure()
    plt.plot(ts, chis)
    plt.xlabel('Time')
    plt.ylabel('Chi')
    plt.show()
