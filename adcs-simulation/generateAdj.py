import numpy as np
import pandas as pd

fields = [
    "q_actual_0", "q_actual_1", "q_actual_2", "q_actual_3",     # 0-3
    "w_actual_x", "w_actual_y", "w_actual_z",                 # 4-6
    "w_rxwls_0", "w_rxwls_1", "w_rxwls_2",                   # 7-9
    "w_estimated_x", "w_estimated_y", "w_estimated_z",        # 10-12
    "w_desired_x", "w_desired_y", "w_desired_z",              # 13-15
    "attitude_err_x", "attitude_err_y", "attitude_err_z",     # 16-18
    "attitude_rate_err_x", "attitude_rate_err_y", "attitude_rate_err_z",  # 19-21
    "M_ctrl_x", "M_ctrl_y", "M_ctrl_z",                      # 22-24
    "M_applied_x", "M_applied_y", "M_applied_z",              # 25-27
    "M_perturb_x", "M_perturb_y", "M_perturb_z",              # 28-30
    "pos_x", "pos_y", "pos_z",                               # 31-33
    "vel_x", "vel_y", "vel_z",                               # 34-36
    "DCM_estimated_00", "DCM_estimated_01", "DCM_estimated_02",  # 37-39
    "DCM_estimated_10", "DCM_estimated_11", "DCM_estimated_12",  # 40-42
    "DCM_estimated_20", "DCM_estimated_21", "DCM_estimated_22",  # 43-45
    "DCM_desired_00", "DCM_desired_01", "DCM_desired_02",     # 46-48
    "DCM_desired_10", "DCM_desired_11", "DCM_desired_12",     # 49-51
    "DCM_desired_20", "DCM_desired_21", "DCM_desired_22",     # 52-54
]

N = 55
A = np.zeros((N, N), dtype=int)

groups = {
    "q_actual":          list(range(0, 4)),
    "w_actual":          list(range(4, 7)),
    "w_rxwls":           list(range(7, 10)),
    "w_estimated":       list(range(10, 13)),
    "w_desired":         list(range(13, 16)),
    "attitude_err":      list(range(16, 19)),
    "attitude_rate_err": list(range(19, 22)),
    "M_ctrl":            list(range(22, 25)),
    "M_applied":         list(range(25, 28)),
    "M_perturb":         list(range(28, 31)),
    "pos":               list(range(31, 34)),
    "vel":               list(range(34, 37)),
    "DCM_estimated":     list(range(37, 46)),
    "DCM_desired":       list(range(46, 55)),
}

def connect(g1, g2):
    for i in groups[g1]:
        for j in groups[g2]:
            A[i, j] = 1
            A[j, i] = 1

def connect_internal(g):
    idx = groups[g]
    for i in idx:
        for j in idx:
            if i != j:
                A[i, j] = 1

# Within-group coupling
for g in ["w_actual", "w_rxwls", "w_estimated", "w_desired",
          "attitude_err", "attitude_rate_err",
          "M_ctrl", "M_applied", "M_perturb",
          "pos", "vel"]:
    connect_internal(g)

connect_internal("DCM_estimated")
connect_internal("DCM_desired")

# Between-group coupling
# 1. Attitude kinematics: q_dot = f(q, w)
connect("q_actual", "w_actual")

# 2. Rotational dynamics: J*w_dot = M_applied + M_perturb - w×(Jw)
connect("w_actual", "M_applied")
connect("w_actual", "M_perturb")

# 3. Angular momentum exchange between reaction wheels and spacecraft body
connect("w_actual", "w_rxwls")
connect("w_rxwls", "M_applied")

# 4. Gyroscope measurement model: w_estimated = w_actual + bias + noise
connect("w_actual", "w_estimated")

# 5. TRIAD attitude estimation
connect("DCM_estimated", "w_estimated")
connect("DCM_estimated", "q_actual")

# 6. Attitude error: err = f(DCM_estimated, DCM_desired)
connect("attitude_err", "DCM_estimated")
connect("attitude_err", "DCM_desired")

# 7. Angular velocity error: rate_err = f(w_estimated, w_desired, attitude_err)
connect("attitude_rate_err", "w_estimated")
connect("attitude_rate_err", "w_desired")
connect("attitude_rate_err", "attitude_err")

# 8. PD control law: M_ctrl = J(-Kd*rate_err - Kp*err)
connect("M_ctrl", "attitude_err")
connect("M_ctrl", "attitude_rate_err")

# 9. Actuator response: M_applied ≈ M_ctrl + noise
connect("M_applied", "M_ctrl")

# 10. Disturbance torque dependence on orbital position
connect("M_perturb", "pos")

# 11. Orbital kinematics: pos_dot = vel
connect("pos", "vel")

# 12. Internal consistency of the desired state
connect("DCM_desired", "w_desired")

print(f"Adjacency matrix shape: {A.shape}")
print(f"Number of undirected edges: {A.sum() // 2}")
print(f"Sparsity: {1 - A.sum() / (N*N):.3f}")
print(np.array(A))
np.save("adcs_adjacency_matrix.npy", A)
# df = pd.DataFrame(A, index=fields, columns=fields)
# df.to_csv("adcs_adjacency_matrix.csv")
# print("Saved adcs_adjacency_matrix.npy and adcs_adjacency_matrix.csv")