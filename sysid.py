from casadi import *
import logger

data_log = logger.from_json('traj.json', 't')
N = len(data_log.times)
samp_freq = N / (data_log.end_time - data_log.start_time)
step_per_sample = 10
dt = 1/samp_freq/step_per_sample

q = MX.sym('q', 2)
qd = MX.sym('qd', 2)

u = MX.sym('u', 2)

M = MX.sym('M', 2,2)
h = MX.sym('h')
Kg = MX.sym('Kg', Sparsity.upper(2))
B = MX.sym('B', Sparsity.diag(2))
Kb = MX.sym('Kb', Sparsity.diag(2))

params = vertcat(M.reshape((4,1)), h, Kg[0,0], Kg[0,1], Kg[1,1], B[0,0], B[1,1], Kb[0,0], Kb[1,1])
params = vertcat(vec(M), h, Kg[0,0], Kg[0,1], Kg[1,1], vec(B), vec(Kb))

H = vertcat(horzcat(2*h,h), horzcat(h,0))
C = h*sin(q[1])*vertcat(horzcat(-qd[1], -(qd[0]+qd[1])), horzcat(qd[0], 0))
cosq = cos(vertcat(q[0], q[0]+q[1]))

basic_torque = B @ u
back_emf = Kb @ qd
coriolis = C @ qd
gravity = Kg @ cosq

rhs = vertcat(qd, solve(M + H*cos(q[1]), basic_torque - back_emf - coriolis - gravity))

states = vertcat(q, qd)

print(params)

ode = Function('ode', [states, u, params], [rhs])



print(ode(np.array([0,0,0,0]), np.array([5,5]), np.array([1,0,0,1,0,0,0,1,1,0,0])))

k1 = ode(states, u, params)
k2 = ode(states + dt/2.0*k1, u, params)
k3 = ode(states + dt/2.0*k2, u, params)
k4 = ode(states + dt*k3, u, params)

end_states = states + dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)

one_step = Function('one_step', [states, u, params], [end_states])

X = states

for i in range(step_per_sample):
    X = one_step(X, u, params)

one_sample = Function('one_sample', [states, u, params], [X])

all_samples = one_sample.mapaccum("all_samples", N)

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

    # K3*Voltage - K4*velocity = motor torque
    K3 = np.array([[N1*G1, 0], [0, N2*G2]])*Kt/Rm
    K4 = np.array([[G1*G1*N1, 0], [0, G2*G2*N2]])*Kt/Kv/Rm

    M = DM([[m1*r1**2 + m2*(l1**2+r1**2) + I1 + I2, m2*r2**2 + I2], [m2*r2**2 + I2, m2*r2**2 + I2]])
    h = DM(m2*l1*r2)
    Kg = DM([[m1*r1+m2*l1, m2*r2], [0, m2*r2]]*g)
    B = DM([[G1*N1*Kt/Rm, 0], [0, G2*N2*Kt/Rm]])