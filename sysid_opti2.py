from casadi import *
import numpy as np
import logger
import matplotlib.pyplot as plt

data_log = logger.from_json('log.json', 't')
N = len(data_log.times)
samp_freq = N / (data_log.end_time - data_log.start_time)
step_per_sample = 4
dt = 1/samp_freq/step_per_sample
ts = np.linspace(data_log.start_time, data_log.end_time, N)

measured_x = np.zeros((4,N))
measured_u = np.zeros((2,N))

t = data_log.start_time
for i in range(N):
    log = data_log.sample(t)
    x = np.array([log['q1'], log['q2'], log['q1d'], log['q2d']])
    u = np.array([log['u1'], log['u2']])

    measured_x[:,i] = x
    measured_u[:,i] = u

    t = t + 1/samp_freq

x0 = measured_x[:,0]

#p = [m1, m2, r1, r2, l1, I1, I2, B11, B22, Kb11, Kb22]
#     0   1   2   3   4   5   6   7    8    9     10
def dynamics(state, input, p):
    A = p[0]*p[2]**2 + p[1]*p[4]**2 + p[5]
    B = p[1]*p[3]**2 + p[6]
    h = p[1]*p[4]*p[3]
    c2 = cos(state[1])
    den = h**2*c2**2 - A*B

    Minv = MX(2,2)
    Minv[0,0] = -B
    Minv[0,1] = B + h*c2
    Minv[1,0] = B + h*c2
    Minv[1,1] = -(A + B + 2*h*c2)
    Minv = Minv/den

    Kg = MX(Sparsity.upper(2))
    Kg[0,0] = p[0]*p[2] + p[1]*p[4]
    Kg[0,1] = p[1]*p[3]
    Kg[1,1] = p[1]*p[3]
    Kg = Kg * 9.806

    B = MX(Sparsity.diag(2))
    B[0,0] = p[7]
    B[1,1] = p[8]

    Kb = MX(Sparsity.diag(2))
    Kb[0,0] = p[9]
    Kb[1,1] = p[10]

    C = MX(2,2)
    C[0,0] = -state[3]
    C[0,1] = -(state[2]+state[3])
    C[1,0] = state[2]
    C = C*h*sin(state[1])

    cosq = cos(vertcat(state[0], state[0]+state[1]))

    dstate = MX(4,1)
    dstate[:2] = state[2:]
    dstate[2:] = Minv @ (B @ input - Kb @ state[2:] - C @ state[2:] - Kg @ cosq)
    return dstate

def rk4(f, x, u, dt):
    k1 = f(x,u)
    k2 = f(x + k1*dt/2.0, u)
    k3 = f(x + k2*dt/2.0, u)
    k4 = f(x + k3*dt, u)

    return x + dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)

def step(state, input, parameters):
    x = state
    for i in range(step_per_sample):
        x = rk4(lambda x,u: dynamics(x, u, parameters), x, input, dt)
    
    return x

def identify(initial_guess = None):
    if initial_guess is None:
        initial_guess = np.ones(11)
    opti = Opti()

    #p = [m1, m2, r1, r2, l1, I1, I2, B11, B22, Kb11, Kb22]
    #     0   1   2   3   4   5   6   7    8    9     10
    p = opti.variable(11)
    opti.subject_to(p[:] > 0.0)
    opti.set_initial(p, initial_guess)

    opti.subject_to(p[1] == initial_guess[1])
    opti.subject_to(p[4] == initial_guess[4])

    X = opti.variable(4, N)
    opti.set_initial(X, measured_x)
    U = opti.parameter(2, N)
    opti.set_value(U, measured_u)
    X_M = opti.parameter(4, N)
    opti.set_value(X_M, measured_x)

    print("Initial state...")
    for k in range(4):
        opti.subject_to(X[k,0] == x0[k])

    print("Intermediate state...")
    for i in range(0,N-1):
        opti.subject_to(X[:,i+1] == step(X[:,i], U[:,i], p))

    print("Creating cost...")
    J = MX(0)
    for i in range(N):
        J = J + (X[0,i] - X_M[0,i])**2 + (X[1,i] - X_M[1,i])**2
    opti.minimize(J)

    opti.solver('ipopt')
    opti.callback(lambda i: print(opti.debug.value(p)))
    #opti.callback(lambda _: plot_sim(opti.debug.value(X), opti.debug.value(X_M)))

    print("Solving...")
    sol = opti.solve()

    print(sol.value(p))
    return (sol, X, X_M)

def plot_sim(X, X_M):
    plt.clf()
    plt.plot(ts, X[0,:])
    plt.plot(ts, X_M[0,:])
    plt.plot(ts, measured_x[0,:])

def get_theoretical():
    # Length of segments
    l1 = 46.25 * .0254
    l2 = 41.80 * .0254

    # Mass of segments
    m1 = 9.34 * .4536
    m2 = 9.77 * .4536

    # Distance from pivot to CG for each segment
    r1 = 21.64 * .0254
    r2 = 26.70 * .0254

    # Moment of inertia about CG for each segment
    I1 = 2957.05 * .0254*.0254 * .4536
    I2 = 2824.70 * .0254*.0254 * .4536

    # Gearing of each segment
    G1 = 140.
    G2 = 90.

    # Number of motors in each gearbox
    N1 = 1
    N2 = 2

    # Gravity
    g = 9.81

    stall_torque = 3.36
    free_speed = 5880.0 * 2.0*np.pi/60.0
    stall_current = 166

    Rm = 12.0/stall_current

    Kv = free_speed / 12.0
    Kt = stall_torque / stall_current

    m11 = m1*r1**2 + m2*(l1**2+r1**2) + I1 + I2
    m12 = m2*r2**2 + I2
    m21 = m12
    m22 = m12
    h = m2*l1*r2
    Kg11 = m1*r1+m2*l1*g
    Kg12 = m2*r2*g
    Kg22 = m2*r2*g
    B11 = G1*N1*Kt/Rm
    B22 = G2*N2*Kt/Rm
    Kb11 = G1*G1*N1*Kt/Kv/Rm
    Kb22 = G2*G2*N2*Kt/Kv/Rm

    return np.array([m1, m2, r1, r2, l1, I1, I2, B11, B22, Kb11, Kb22])

def main():
    p_guess = get_theoretical()
    print(p_guess)
    plt.figure()
    (sol, X, X_M) = identify(p_guess)
    print(sol.value(X))
    plt.clf()
    plt.plot(ts, measured_x[0,:])
    plt.plot(ts, measured_x[1,:])
    plt.plot(ts, sol.value(X)[0,:])
    plt.plot(ts, sol.value(X)[1,:])
    plt.show()

if __name__ == "__main__":
    main()

#[1.74144939e+04 1.26986698e+04 9.49915451e+03 4.81151813e-03
# 6.20799743e-02 1.00233054e+05 7.11229579e+04 1.01806379e+05
# 9.19764101e+04 1.74638868e+05 2.85319077e+05 3.34559277e+05]