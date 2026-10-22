# -*- coding: utf-8 -*-
"""Project module for attitude determination and control system.

This module is simply the control script that utilizes the simulation engine
and plots the results.

Dataset Generation
------------------
Call ``generate_dataset()`` (or run this file directly) to produce three CSV
files in the current working directory:

    adcs_train.csv        -- Normal operation, no anomalies (training set)
    adcs_test.csv         -- Five injected anomaly types (test set)
    adcs_test_labels.csv  -- Per-timestep labels for the test set

Anomaly types injected into the test set
-----------------------------------------
  Type 1  Gyro Bias Spike         t=[ 800, 860)s   duration=60s
          A sudden additive offset on the estimated angular velocity,
          simulating a radiation-induced IMU latch-up.  The spacecraft
          state is unchanged; only the controller's perception of ω is
          corrupted, leading to a wrong control torque.

  Type 2  Reaction Wheel Saturation  t=[1500,1545)s  duration=45s
          The commanded control torque is forced to zero, simulating a
          reaction-wheel over-temperature protection trip or driver
          failure.  Attitude error grows unimpeded.

  Type 3  Attitude Jump (Impact)   t=[2200,2205)s  duration=5s
          An impulsive torque is added to the perturbation torque,
          simulating a micro-debris strike that instantaneously imparts
          angular momentum.

  Type 4  Magnetic Torque Spike    t=[3000,3120)s  duration=120s
          The perturbation torque is scaled up by ×15, simulating an
          elevated geomagnetic field during a solar storm.

  Type 5  Control Torque Reversal  t=[4200,4230)s  duration=30s
          M_ctrl is negated before reaching the actuators, simulating a
          sign-flip bug in the control law.  Creates positive feedback
          and attitude divergence.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spacecraft import Spacecraft
from actuators import Actuators
from sensors import Gyros, Magnetometer, EarthHorizonSensor
from controller import PDController
from math_utils import (quaternion_multiply, t1_matrix, t2_matrix, t3_matrix,
                        dcm_to_quaternion, quaternion_to_dcm, normalize, cross)
from simulation import simulate_adcs


# ---------------------------------------------------------------------------
# Anomaly schedule for the test set
# ---------------------------------------------------------------------------
# Keys   : anomaly start time (seconds)
# Values : config dict consumed by simulation.py
#
# Physical parameter choices
#   gyro_bias_spike : 0.05 rad/s ~ 170×  the LN-200S 1σ ARW noise at Δt=1s,
#                     large enough to flip attitude-rate error sign.
#   impact_torque   : 5×10⁻⁴ N·m ~ 5× max gravity-gradient torque at 400 km.
#   mag_scale       : 15 — consistent with severe geomagnetic storm (Kp ≥ 8).
# ---------------------------------------------------------------------------
ANOMALY_SCHEDULE = {
    800:  {
        'type': 1,
        'duration': 60,
        'gyro_bias_spike': np.array([0.05, -0.025, 0.04]),   # rad/s
    },
    1500: {
        'type': 2,
        'duration': 45,
    },
    2200: {
        'type': 3,
        'duration': 5,
        'impact_torque': np.array([5e-4, 2.5e-4, -1.5e-4]),  # N·m
    },
    3000: {
        'type': 4,
        'duration': 120,
        'mag_scale': 15.0,
    },
    4200: {
        'type': 5,
        'duration': 30,
    },
}

ANOMALY_TYPE_NAMES = {
    0: 'normal',
    1: 'gyro_bias_spike',
    2: 'reaction_wheel_saturation',
    3: 'attitude_jump_impact',
    4: 'magnetic_torque_spike',
    5: 'control_torque_reversal',
}


def _build_spacecraft(J, controller, gyros, magnetometer,
                      earth_horizon_sensor, actuators,
                      q_0, w_0, r_0, v_0):
    """Helper: construct a fresh Spacecraft with the given sub-systems."""
    return Spacecraft(
        J=J,
        controller=controller,
        gyros=gyros,
        magnetometer=magnetometer,
        earth_horizon_sensor=earth_horizon_sensor,
        actuators=actuators,
        q=q_0,
        w=w_0,
        r=r_0,
        v=v_0)


def _build_actuators(perfect=False):
    if perfect:
        return Actuators(
            rxwl_mass=226e-3,
            rxwl_radius=0.5 * 65e-3,
            rxwl_max_torque=np.inf,
            rxwl_max_momentum=np.inf,
            noise_factor=0.0)
    return Actuators(
        rxwl_mass=226e-3,
        rxwl_radius=0.5 * 65e-3,
        rxwl_max_torque=20e-3,
        rxwl_max_momentum=0.18,
        noise_factor=0.03)


def _results_to_dataframe(results):
    """Flatten the simulate_adcs results dict into a tidy DataFrame.

    The DCM arrays (shape N×3×3) are flattened to 9 scalar columns each,
    named DCM_estimated_ij and DCM_desired_ij (i,j ∈ {0,1,2}).
    """
    N = len(results["times"])

    data = {
        'time':              results["times"],
        # actual quaternion
        'q_actual_0':        results["q_actual"][:, 0],
        'q_actual_1':        results["q_actual"][:, 1],
        'q_actual_2':        results["q_actual"][:, 2],
        'q_actual_3':        results["q_actual"][:, 3],
        # actual angular velocity (rad/s)
        'w_actual_x':        results["w_actual"][:, 0],
        'w_actual_y':        results["w_actual"][:, 1],
        'w_actual_z':        results["w_actual"][:, 2],
        # reaction-wheel angular velocity (rad/s)
        'w_rxwls_0':         results["w_rxwls"][:, 0],
        'w_rxwls_1':         results["w_rxwls"][:, 1],
        'w_rxwls_2':         results["w_rxwls"][:, 2],
        # estimated angular velocity (rad/s)
        'w_estimated_x':     results["w_estimated"][:, 0],
        'w_estimated_y':     results["w_estimated"][:, 1],
        'w_estimated_z':     results["w_estimated"][:, 2],
        # desired angular velocity (rad/s)
        'w_desired_x':       results["w_desired"][:, 0],
        'w_desired_y':       results["w_desired"][:, 1],
        'w_desired_z':       results["w_desired"][:, 2],
        # attitude error (rad, small-angle)
        'attitude_err_x':    results["attitude_err"][:, 0],
        'attitude_err_y':    results["attitude_err"][:, 1],
        'attitude_err_z':    results["attitude_err"][:, 2],
        # attitude rate error (rad/s)
        'attitude_rate_err_x': results["attitude_rate_err"][:, 0],
        'attitude_rate_err_y': results["attitude_rate_err"][:, 1],
        'attitude_rate_err_z': results["attitude_rate_err"][:, 2],
        # commanded control torque (N·m)
        'M_ctrl_x':          results["M_ctrl"][:, 0],
        'M_ctrl_y':          results["M_ctrl"][:, 1],
        'M_ctrl_z':          results["M_ctrl"][:, 2],
        # actually applied torque (N·m)
        'M_applied_x':       results["M_applied"][:, 0],
        'M_applied_y':       results["M_applied"][:, 1],
        'M_applied_z':       results["M_applied"][:, 2],
        # perturbation torque (N·m)
        'M_perturb_x':       results["M_perturb"][:, 0],
        'M_perturb_y':       results["M_perturb"][:, 1],
        'M_perturb_z':       results["M_perturb"][:, 2],
        # inertial position (m)
        'pos_x':             results["positions"][:, 0],
        'pos_y':             results["positions"][:, 1],
        'pos_z':             results["positions"][:, 2],
        # inertial velocity (m/s)
        'vel_x':             results["velocities"][:, 0],
        'vel_y':             results["velocities"][:, 1],
        'vel_z':             results["velocities"][:, 2],
    }

    # flatten DCM_estimated  (N×3×3) → 9 columns
    for i in range(3):
        for j in range(3):
            data[f'DCM_estimated_{i}{j}'] = results["DCM_estimated"][:, i, j]

    # flatten DCM_desired  (N×3×3) → 9 columns
    for i in range(3):
        for j in range(3):
            data[f'DCM_desired_{i}{j}'] = results["DCM_desired"][:, i, j]

    return pd.DataFrame(data)


def _make_labels(times, anomaly_schedule):
    """Build per-timestep label arrays aligned with *times*.

    Returns
    -------
    label : ndarray of int, shape (N,)
        0 = normal, 1 = anomalous
    anomaly_type : ndarray of int, shape (N,)
        0 = normal, 1–5 = anomaly type number
    """
    label       = np.zeros(len(times), dtype=int)
    anomaly_type = np.zeros(len(times), dtype=int)

    for t_start, config in anomaly_schedule.items():
        duration = config.get('duration', 1)
        mask = (times >= t_start) & (times < t_start + duration)
        label[mask]        = 1
        anomaly_type[mask] = config['type']

    return label, anomaly_type


def generate_dataset(stop_time=6000, delta_t=1,
                     train_csv='adcs_train.csv',
                     test_csv='adcs_test.csv',
                     labels_csv='adcs_test_labels.csv',
                     verbose=False):
    """Run two simulations and save training set, test set, and labels as CSV.

    The training set uses the same spacecraft / sensor / actuator parameters
    as the original project.py simulation.  The test set is identical except
    that the anomaly schedule defined in ANOMALY_SCHEDULE is active.

    Args:
        stop_time (float): simulation duration in seconds (default 6000).
        delta_t (float): logging interval in seconds (default 1).
        train_csv (str): output path for the training CSV.
        test_csv (str): output path for the test CSV.
        labels_csv (str): output path for the test-set labels CSV.
        verbose (bool): passed to simulate_adcs.

    Returns:
        tuple: (df_train, df_test, df_labels) — the three DataFrames.
    """

    # ------------------------------------------------------------------
    # Shared spacecraft parameters (identical to original project.py)
    # ------------------------------------------------------------------
    sc_mass = 8
    sc_dim = [226.3e-3, 100.0e-3, 366.0e-3]
    J = 1 / 12 * sc_mass * np.diag([
        sc_dim[1]**2 + sc_dim[2]**2,
        sc_dim[0]**2 + sc_dim[2]**2,
        sc_dim[0]**2 + sc_dim[1]**2
    ])

    controller = PDController(
        k_d=np.diag([.01, .01, .01]),
        k_p=np.diag([.1,  .1,  .1]))

    gyros                   = Gyros(bias_stability=1, angular_random_walk=0.07)
    magnetometer            = Magnetometer(resolution=10e-9)
    earth_horizon_sensor    = EarthHorizonSensor(accuracy=0.25)
    actuators               = _build_actuators(perfect=False)

    mu_earth     = 3.986004418e14
    R_e          = 6.3781e6
    orbit_radius = R_e + 400e3
    orbit_w      = np.sqrt(mu_earth / orbit_radius**3)
    period       = 2 * np.pi / orbit_w

    def position_velocity_func(t):
        r = orbit_radius / np.sqrt(2) * np.array([
            -np.cos(orbit_w * t),
             np.sqrt(2) * np.sin(orbit_w * t),
             np.cos(orbit_w * t),
        ])
        v = orbit_w * orbit_radius / np.sqrt(2) * np.array([
             np.sin(orbit_w * t),
             np.sqrt(2) * np.cos(orbit_w * t),
            -np.sin(orbit_w * t),
        ])
        return r, v

    r_0, v_0 = position_velocity_func(0)

    b_x = -normalize(r_0)
    b_y = normalize(v_0)
    b_z = cross(b_x, b_y)

    dcm_0_nominal = np.stack([b_x, b_y, b_z])
    q_0_nominal   = dcm_to_quaternion(dcm_0_nominal)

    w_nominal_i = 2 * np.pi / period * normalize(cross(r_0, v_0))
    w_nominal   = np.matmul(dcm_0_nominal, w_nominal_i)

    q_0 = quaternion_multiply(
        np.array([0, np.sin(2 * np.pi / 180 / 2), 0,
                  np.cos(2 * np.pi / 180 / 2)]),
        q_0_nominal)
    w_0 = w_nominal + np.array([0.005, 0, 0])

    def perturb_func(satellite):
        return (satellite.approximate_gravity_gradient_torque()
                + satellite.approximate_magnetic_field_torque())

    def desired_state_func(t):
        if w_nominal[0] != 0:
            dcm_nominal = np.matmul(t1_matrix(w_nominal[0] * t), dcm_0_nominal)
        elif w_nominal[1] != 0:
            dcm_nominal = np.matmul(t2_matrix(w_nominal[1] * t), dcm_0_nominal)
        elif w_nominal[2] != 0:
            dcm_nominal = np.matmul(t3_matrix(w_nominal[2] * t), dcm_0_nominal)
        return dcm_nominal, w_nominal

    # ------------------------------------------------------------------
    # 1. Training set — no anomalies
    # ------------------------------------------------------------------
    print("Generating training set (no anomalies)...")
    satellite_train = _build_spacecraft(
        J, controller, gyros, magnetometer,
        earth_horizon_sensor, actuators,
        q_0, w_0, r_0, v_0)

    results_train = simulate_adcs(
        satellite=satellite_train,
        nominal_state_func=desired_state_func,
        perturbations_func=perturb_func,
        position_velocity_func=position_velocity_func,
        start_time=0,
        delta_t=delta_t,
        stop_time=stop_time,
        verbose=verbose,
        anomaly_schedule=None)   # <-- no anomalies

    df_train = _results_to_dataframe(results_train)
    df_train['label']             = 0
    df_train['anomaly_type']      = 0
    df_train['anomaly_type_name'] = 'normal'

    df_train.to_csv(train_csv, index=False)
    print(f"  Saved {len(df_train)} rows -> {train_csv}")

    # ------------------------------------------------------------------
    # 2. Test set — five injected anomalies
    # ------------------------------------------------------------------
    print("Generating test set (with anomalies)...")

    # Re-instantiate all sub-systems with fresh state so the test run is
    # independent of the training run (gyro bias, wheel velocities, etc.).
    gyros_test                = Gyros(bias_stability=1, angular_random_walk=0.07)
    magnetometer_test         = Magnetometer(resolution=10e-9)
    earth_horizon_sensor_test = EarthHorizonSensor(accuracy=0.25)
    actuators_test            = _build_actuators(perfect=False)

    satellite_test = _build_spacecraft(
        J, controller,
        gyros_test, magnetometer_test, earth_horizon_sensor_test,
        actuators_test,
        q_0, w_0, r_0, v_0)

    results_test = simulate_adcs(
        satellite=satellite_test,
        nominal_state_func=desired_state_func,
        perturbations_func=perturb_func,
        position_velocity_func=position_velocity_func,
        start_time=0,
        delta_t=delta_t,
        stop_time=stop_time,
        verbose=verbose,
        anomaly_schedule=ANOMALY_SCHEDULE)   # <-- anomalies active

    df_test = _results_to_dataframe(results_test)

    # attach ground-truth labels
    label, anomaly_type = _make_labels(
        df_test['time'].values, ANOMALY_SCHEDULE)
    df_test['label']             = label
    df_test['anomaly_type']      = anomaly_type
    df_test['anomaly_type_name'] = [ANOMALY_TYPE_NAMES[a] for a in anomaly_type]

    df_test.to_csv(test_csv, index=False)
    print(f"  Saved {len(df_test)} rows -> {test_csv}")
    print(f"  Anomaly breakdown (label=1): "
          f"{label.sum()} / {len(label)} timesteps "
          f"({100 * label.mean():.1f} %)")
    for atype, name in ANOMALY_TYPE_NAMES.items():
        if atype == 0:
            continue
        n = (anomaly_type == atype).sum()
        if n > 0:
            print(f"    type {atype} ({name}): {n} timesteps")

    # ------------------------------------------------------------------
    # 3. Standalone labels file (time | label | anomaly_type | name)
    # ------------------------------------------------------------------
    df_labels = df_test[['time', 'label',
                          'anomaly_type', 'anomaly_type_name']].copy()
    df_labels.to_csv(labels_csv, index=False)
    print(f"  Saved labels -> {labels_csv}")

    return df_train, df_test, df_labels


def main():
    # Define 6U CubeSat mass, dimensions, drag coefficient
    sc_mass = 8
    sc_dim = [226.3e-3, 100.0e-3, 366.0e-3]
    J = 1 / 12 * sc_mass * np.diag([
        sc_dim[1]**2 + sc_dim[2]**2, sc_dim[0]**2 + sc_dim[2]**2,
        sc_dim[0]**2 + sc_dim[1]**2
    ])
    sc_dipole = np.array([0, 0.018, 0])

    # Define two `PDController` objects—one to represent no control and one
    # to represent PD control with the specified gains
    no_controller = PDController(
        k_d=np.diag([0, 0, 0]), k_p=np.diag([0, 0, 0]))
    controller = PDController(
        k_d=np.diag([.01, .01, .01]), k_p=np.diag([.1, .1, .1]))

    # Northrop Grumman LN-200S Gyros
    gyros = Gyros(bias_stability=1, angular_random_walk=0.07)
    perfect_gyros = Gyros(bias_stability=0, angular_random_walk=0)

    # NewSpace Systems Magnetometer
    magnetometer = Magnetometer(resolution=10e-9)
    perfect_magnetometer = Magnetometer(resolution=0)

    # Adcole Maryland Aerospace MAI-SES Static Earth Sensor
    earth_horizon_sensor = EarthHorizonSensor(accuracy=0.25)
    perfect_earth_horizon_sensor = EarthHorizonSensor(accuracy=0)

    # Sinclair Interplanetary 60 mNm-sec RXWLs
    actuators = Actuators(
        rxwl_mass=226e-3,
        rxwl_radius=0.5 * 65e-3,
        rxwl_max_torque=20e-3,
        rxwl_max_momentum=0.18,
        noise_factor=0.03)
    perfect_actuators = Actuators(
        rxwl_mass=226e-3,
        rxwl_radius=0.5 * 65e-3,
        rxwl_max_torque=np.inf,
        rxwl_max_momentum=np.inf,
        noise_factor=0.0)

    # define some orbital parameters
    mu_earth = 3.986004418e14
    R_e = 6.3781e6
    orbit_radius = R_e + 400e3
    orbit_w = np.sqrt(mu_earth / orbit_radius**3)
    period = 2 * np.pi / orbit_w

    # define a function that returns the inertial position and velocity of the
    # spacecraft (in m & m/s) at any given time
    def position_velocity_func(t):
        r = orbit_radius / np.sqrt(2) * np.array([
            -np.cos(orbit_w * t),
            np.sqrt(2) * np.sin(orbit_w * t),
            np.cos(orbit_w * t),
        ])
        v = orbit_w * orbit_radius / np.sqrt(2) * np.array([
            np.sin(orbit_w * t),
            np.sqrt(2) * np.cos(orbit_w * t),
            -np.sin(orbit_w * t),
        ])
        return r, v

    # compute the initial inertial position and velocity
    r_0, v_0 = position_velocity_func(0)

    # define the body axes in relation to where we want them to be:
    # x = Earth-pointing
    # y = pointing along the velocity vector
    # z = normal to the orbital plane
    b_x = -normalize(r_0)
    b_y = normalize(v_0)
    b_z = cross(b_x, b_y)

    # construct the nominal DCM from inertial to body (at time 0) from the body
    # axes and compute the equivalent quaternion
    dcm_0_nominal = np.stack([b_x, b_y, b_z])
    q_0_nominal = dcm_to_quaternion(dcm_0_nominal)

    # compute the nominal angular velocity required to achieve the reference
    # attitude; first in inertial coordinates then body
    w_nominal_i = 2 * np.pi / period * normalize(cross(r_0, v_0))
    w_nominal = np.matmul(dcm_0_nominal, w_nominal_i)

    # provide some initial offset in both the attitude and angular velocity
    q_0 = quaternion_multiply(
        np.array(
            [0, np.sin(2 * np.pi / 180 / 2), 0,
             np.cos(2 * np.pi / 180 / 2)]), q_0_nominal)
    w_0 = w_nominal + np.array([0.005, 0, 0])

    # define a function that will model perturbations
    def perturb_func(satellite):
        return (satellite.approximate_gravity_gradient_torque() +
                satellite.approximate_magnetic_field_torque())

    # define a function that returns the desired state at any given point in
    # time (the initial state and a subsequent rotation about the body x, y, or
    # z axis depending upon which nominal angular velocity term is nonzero)
    def desired_state_func(t):
        if w_nominal[0] != 0:
            dcm_nominal = np.matmul(t1_matrix(w_nominal[0] * t), dcm_0_nominal)
        elif w_nominal[1] != 0:
            dcm_nominal = np.matmul(t2_matrix(w_nominal[1] * t), dcm_0_nominal)
        elif w_nominal[2] != 0:
            dcm_nominal = np.matmul(t3_matrix(w_nominal[2] * t), dcm_0_nominal)
        return dcm_nominal, w_nominal

    satellite_noise = Spacecraft(
        J=J,
        controller=controller,
        gyros=gyros,
        magnetometer=magnetometer,
        earth_horizon_sensor=earth_horizon_sensor,
        actuators=actuators,
        q=q_0,
        w=w_0,
        r=r_0,
        v=v_0)

    simulate(
        satellite=satellite_noise,
        nominal_state_func=desired_state_func,
        perturbations_func=perturb_func,
        position_velocity_func=position_velocity_func,
        stop_time=6000,
        tag=r"(Actual Estimation \& Control)")

    # ------------------------------------------------------------------
    # Uncomment the line below to also generate the anomaly dataset files
    generate_dataset()
    # ------------------------------------------------------------------


def simulate(satellite,
             nominal_state_func,
             perturbations_func,
             position_velocity_func,
             stop_time=6000,
             tag=""):

    # carry out the actual simulation and gather the results
    results = simulate_adcs(
        satellite=satellite,
        nominal_state_func=nominal_state_func,
        perturbations_func=perturbations_func,
        position_velocity_func=position_velocity_func,
        start_time=0,
        delta_t=1,
        stop_time=stop_time,
        verbose=True)

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    plt.rc("figure", dpi=120)
    plt.rc("savefig", dpi=120)

    print(results)


if __name__ == "__main__":
    # Running the original simulation (unchanged behaviour):
    #   python project.py
    #
    # Generating the anomaly-detection dataset instead:
    #   python project.py --dataset
    import sys
    if '--dataset' in sys.argv:
        generate_dataset()
    else:
        main()
