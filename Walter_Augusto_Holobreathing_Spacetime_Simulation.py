import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
L = 100  # Size of grid
T = 300  # Number of time steps
epsilon = 0.08  # Breathing nonlinearity strength
dt = 0.1  # Time step
dx = 1.0  # Space step
c = 1.0  # "Speed" of breathing

# Initialize fields
phi = np.random.uniform(-0.2, 0.2, (L, L))  # Breathing field
phi_new = np.copy(phi)
phi_old = np.copy(phi)

# Create figure
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(phi, cmap='plasma', vmin=-1, vmax=1)
plt.axis('off')

def update(frame):
    global phi, phi_old, phi_new

    laplacian = (
        np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) +
        np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) - 4 * phi
    ) / dx**2

    # Breathing field evolution (based on our breathing equation)
    acceleration = c**2 * laplacian - np.sin(phi) * (1 + epsilon * phi**2)

    phi_new = 2 * phi - phi_old + acceleration * dt**2
    phi_old = np.copy(phi)
    phi = np.copy(phi_new)

    im.set_array(phi)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=T, interval=50, blit=True)
ani.save('Walter_Augusto_Holobreathing_Spacetime.gif', writer='pillow')

plt.close()
