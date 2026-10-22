# -*- coding: utf-8 -*-
"""Simulation module for attitude determination and control system.

This module forms the core of the simulation engine and utilizes the classes
and functions written elsewhere to model the system and perform numerical
integration.
"""

import numpy as np
from scipy.integrate import ode, RK45, odeint
from kinematics import quaternion_derivative
from dynamics import angular_velocity_derivative
from math_utils import normalize
from errors import calculate_attitude_error, calculate_attitude_rate_error


# ---------------------------------------------------------------------------
# Anomaly type reference (injected via anomaly_schedule)
# ---------------------------------------------------------------------------
# Each entry in anomaly_schedule is keyed by t_start (float, seconds) and
# maps to a dict with the following fields:
#
#   type (int):
#     1 - Gyro Bias Spike
#         Sudden additive offset on the estimated angular velocity, simulating
#         a radiation-induced IMU latch-up or gyro bias jump.
#         Physical effect: corrupted w_estimated -> wrong attitude error ->
#         wrong control torque.
#         Required key: 'gyro_bias_spike' (ndarray, rad/s, shape (3,))
#
#     2 - Reaction Wheel Saturation
#         Forces the commanded control torque to zero for the anomaly window,
#         simulating a wheel over-temperature protection trip or driver failure.
#         Physical effect: attitude error grows unimpeded; wheels stop reacting.
#         No extra keys required.
#
#     3 - Attitude Jump (Impact / Debris Strike)
#         Adds a one-shot impulsive torque to M_perturb for the anomaly window,
#         simulating a micro-debris impact that instantaneously imparts angular
#         momentum to the spacecraft.
#         Required key: 'impact_torque' (ndarray, N·m, shape (3,))
#
#     4 - Magnetic Torque Spike (Geomagnetic Storm)
#         Scales M_perturb by a factor > 1 for the anomaly window, simulating
#         an elevated magnetic field during a geomagnetic storm or intense solar
#         particle event.
#         Required key: 'mag_scale' (float, dimensionless, e.g. 15.0)
#
#     5 - Control Torque Reversal (Controller Software Fault)
#         Negates the sign of M_ctrl before passing it to the actuators,
#         simulating a sign-flip bug in the control law (e.g. wrong DCM
#         convention, parameter table corruption).
#         Physical effect: positive feedback -> attitude divergence.
#         No extra keys required.
#
#   duration (float): length of anomaly window in seconds
#
# Example:
#   anomaly_schedule = {
#       800:  {'type': 1, 'duration': 60,
#              'gyro_bias_spike': np.array([0.05, -0.025, 0.04])},
#       1500: {'type': 2, 'duration': 45},
#       2200: {'type': 3, 'duration': 5,
#              'impact_torque': np.array([5e-4, 2.5e-4, -1.5e-4])},
#       3000: {'type': 4, 'duration': 120, 'mag_scale': 15.0},
#       4200: {'type': 5, 'duration': 30},
#   }
# ---------------------------------------------------------------------------


def _get_active_anomaly(t, anomaly_schedule):
    """Return the anomaly config active at time t, or None."""
    if anomaly_schedule is None:
        return None
    for t_start, config in anomaly_schedule.items():
        if t_start <= t < t_start + config.get('duration', 1):
            return config
    return None


def simulate_adcs(satellite,
                  nominal_state_func,
                  perturbations_func,
                  position_velocity_func,
                  start_time=0,
                  delta_t=1,
                  stop_time=6000,
                  verbose=False,
                  anomaly_schedule=None):
    """Simulates an attitude determination and control system over a period of time

    Args:
        satellite (Spacecraft): the Spacecraft object that represents the
            satellite being modeled
        nominal_state_func (function): the function that should compute the
            nominal attitude (in DCM form) and angular velocity; its header
            must be (t)
        perturbations_func (function): the function that should compute the
            perturbation torques (N * m); its header must be (satellite)
        position_velocity_func (function): the function that should compute
            the position and velocity; its header must be (t)
        start_time (float, optional): Defaults to 0. The start time of the
            simulation in seconds
        delta_t (float, optional): Defaults to 1. The time between user-defined
            integrator steps (not the internal/adaptive integrator steps) in
            seconds
        stop_time (float, optional): Defaults to 6000. The end time of the
            simulation in seconds
        verbose (bool, optional). Defaults to False. Whether or not to print
            integrator output to the console while running.
        anomaly_schedule (dict, optional): Defaults to None (no anomalies).
            Keys are anomaly start times (float, seconds); values are anomaly
            config dicts. See module-level docstring for full specification.

    Returns:
        dict: a dictionary of simulation results. Each value is an NxM numpy
            ndarray where N is the number of time steps taken and M is the
            size of the data being represented at that time (M=1 for time,
            3 for angular velocity, 4 for quaternion, etc.)
            Contains:
                - times (numpy ndarray): the times of all associated data
                - q_actual (numpy ndarray): actual quaternion
                - w_actual (numpy ndarray): actual angular velocity
                - w_rxwls (numpy ndarray): angular velocity of the reaction
                    wheels
                - DCM_estimated (numpy ndarray): estimated DCM
                - w_estimated (numpy ndarray): estimated angular velocity
                - DCM_desired (numpy ndarray): desired DCM
                - w_desired (numpy ndarray): desired angular velocity
                - attitude_err (numpy ndarray): attitude error
                - attitude_rate_err (numpy ndarray): attitude rate error
                - M_ctrl (numpy ndarray): control torque
                - M_applied (numpy ndarray): applied control torque
                - w_dot_rxwls (numpy ndarray): angular acceleration of
                    reaction wheels
                - M_perturb (numpy ndarray): sum of perturbation torques
                - positions (numpy ndarray): inertial positions
                - velocities (numpy ndarray): inertial velocities

    """
    try:
        init_state = [*satellite.q, *satellite.w, *satellite.actuators.w_rxwls]
    except AttributeError:
        init_state = [*satellite.q, *satellite.w, 0, 0, 0]

    solver = ode(derivatives_func)
    solver.set_integrator(
        'lsoda',
        rtol=(1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-6, 1e-6,
              1e-6),
        atol=(1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-8, 1e-8,
              1e-8),
        nsteps=10000)
    solver.set_initial_value(y=init_state, t=start_time)
    solver.set_f_params(satellite, nominal_state_func, perturbations_func,
                        position_velocity_func, delta_t, anomaly_schedule)

    length = int((stop_time - 0) / delta_t + 1)
    times = np.empty((length,))
    q_actual = np.empty((length, 4))
    w_actual = np.empty((length, 3))
    w_rxwls = np.empty((length, 3))
    DCM_estimated = np.empty((length, 3, 3))
    w_estimated = np.empty((length, 3))
    DCM_desired = np.empty((length, 3, 3))
    w_desired = np.empty((length, 3))
    attitude_err = np.empty((length, 3))
    attitude_rate_err = np.empty((length, 3))
    M_ctrl = np.empty((length, 3))
    M_applied = np.empty((length, 3))
    w_dot_rxwls = np.empty((length, 3))
    M_perturb = np.empty((length, 3))
    positions = np.empty((length, 3))
    velocities = np.empty((length, 3))
    i = 0

    if verbose:
        print("Starting propagation at time: {}".format(start_time))

    while solver.successful() and solver.t <= stop_time:
        if verbose:
            print("Time: {}\nState: {}\n".format(solver.t, solver.y))

        # this section currently duplicates calculations for logging purposes
        t = solver.t
        q = normalize(solver.y[0:4])
        w = solver.y[4:7]
        r, v = position_velocity_func(t)
        satellite.q = q
        satellite.w = w
        satellite.r = r
        satellite.v = v

        # look up whether an anomaly is active at this logged timestep
        active_anomaly = _get_active_anomaly(t, anomaly_schedule)

        _, _, log = simulate_estimation_and_control(
            t, satellite, nominal_state_func, delta_t,
            log=True, active_anomaly=active_anomaly)

        times[i] = t
        q_actual[i] = q
        w_actual[i] = w
        w_rxwls[i] = solver.y[7:10]
        DCM_estimated[i] = log["DCM_estimated"]
        w_estimated[i] = log["w_estimated"]
        DCM_desired[i] = log["DCM_desired"]
        w_desired[i] = log["w_desired"]
        attitude_err[i] = log["attitude_err"]
        attitude_rate_err[i] = log["attitude_rate_err"]
        M_ctrl[i] = log["M_ctrl"]
        M_applied[i] = log["M_applied"]
        w_dot_rxwls[i] = log["w_dot_rxwls"]
        M_perturb[i] = perturbations_func(satellite)

        # Type 3 & 4: apply anomaly to the logged perturbation torque as well,
        # so the saved M_perturb column reflects what was actually fed to the
        # dynamics (consistent with derivatives_func below).
        if active_anomaly is not None:
            if active_anomaly['type'] == 3:
                M_perturb[i] = M_perturb[i] + active_anomaly['impact_torque']
            elif active_anomaly['type'] == 4:
                M_perturb[i] = M_perturb[i] * active_anomaly['mag_scale']

        positions[i] = r
        velocities[i] = v
        i += 1

        # continue integrating
        solver.integrate(solver.t + delta_t)

    results = {}
    results["times"] = times
    results["q_actual"] = q_actual
    results["w_actual"] = w_actual
    results["w_rxwls"] = w_rxwls
    results["DCM_estimated"] = DCM_estimated
    results["w_estimated"] = w_estimated
    results["DCM_desired"] = DCM_desired
    results["w_desired"] = w_desired
    results["attitude_err"] = attitude_err
    results["attitude_rate_err"] = attitude_rate_err
    results["M_ctrl"] = M_ctrl
    results["M_applied"] = M_applied
    results["M_perturb"] = M_perturb
    results["positions"] = positions
    results["velocities"] = velocities
    return results


def derivatives_func(t, x, satellite, nominal_state_func, perturbations_func,
                     position_velocity_func, delta_t, anomaly_schedule=None):
    """Computes the derivative of the spacecraft state

    Args:
        t (float): the time (in seconds)
        x (numpy ndarray): the state (10x1) where the elements are:
            [0, 1, 2, 3]: the quaternion describing the spacecraft attitude
            [4, 5, 6]: the angular velocity of the spacecraft
            [7, 8, 9]: the angular velocities of the reaction wheels
        satellite (Spacecraft): the Spacecraft object that represents the
            satellite being modeled
        nominal_state_func (function): the function that should compute the
            nominal attitude (in DCM form) and angular velocity; its header
            must be (t)
        perturbations_func (function): the function that should compute the
            perturbation torques (N * m); its header must be (satellite)
        position_velocity_func (function): the function that should compute
            the position and velocity; its header must be (t)
        delta_t (float): the time between user-defined integrator steps
                (not the internal/adaptive integrator steps) in seconds
        anomaly_schedule (dict, optional): see simulate_adcs docstring.

    Returns:
        numpy ndarray: the derivative of the state (10x1) with respect to time
    """
    r, v = position_velocity_func(t)
    satellite.q = normalize(x[0:4])
    satellite.w = x[4:7]
    satellite.r = r
    satellite.v = v
    # only set if the satellite has actuators
    try:
        satellite.actuators.w_rxwls = x[7:10]
    except AttributeError:
        pass

    active_anomaly = _get_active_anomaly(t, anomaly_schedule)

    M_applied, w_dot_rxwls, _ = simulate_estimation_and_control(
        t, satellite, nominal_state_func, delta_t,
        active_anomaly=active_anomaly)

    # calculate the perturbing torques on the satellite
    M_perturb = perturbations_func(satellite)

    # Type 3: debris impact — add impulsive torque to perturbation
    if active_anomaly is not None and active_anomaly['type'] == 3:
        M_perturb = M_perturb + active_anomaly['impact_torque']

    # Type 4: geomagnetic storm — scale up perturbation torque
    if active_anomaly is not None and active_anomaly['type'] == 4:
        M_perturb = M_perturb * active_anomaly['mag_scale']

    dx = np.empty((10,))
    dx[0:4] = quaternion_derivative(satellite.q, satellite.w)
    dx[4:7] = angular_velocity_derivative(satellite.J, satellite.w,
                                          [M_applied, M_perturb])
    dx[7:10] = w_dot_rxwls
    return dx


def simulate_estimation_and_control(t,
                                    satellite,
                                    nominal_state_func,
                                    delta_t,
                                    log=False,
                                    active_anomaly=None):
    """Simulates attitude estimation and control for derivatives calculation

    Args:
        t (float): the time (in seconds)
        satellite (Spacecraft): the Spacecraft object that represents the
            satellite being modeled
        nominal_state_func (function): the function that should compute the
            nominal attitude (in DCM form) and angular velocity; its header
            must be (t)
        delta_t (float): the time between user-defined integrator steps
                (not the internal/adaptive integrator steps) in seconds
        log (bool): whether to return a detailed log dict
        active_anomaly (dict or None): anomaly config for the current timestep,
            as returned by _get_active_anomaly().  None means no anomaly.

    Returns:
        numpy ndarray: the control moment (3x1) as actually applied on
                the reaction wheels (the input control torque with some
                Gaussian noise applied) (N * m)
        numpy ndarray: the angular acceleration of the 3 reaction wheels
            applied to achieve the applied torque (rad/s^2)
        dict: a dictionary containing results logged for this simulation step;
            Contains:
                - DCM_estimated (numpy ndarray): estimated DCM
                - w_estimated (numpy ndarray): estimated angular velocity
                - DCM_desired (numpy ndarray): desired DCM
                - w_desired (numpy ndarray): desired angular velocity
                - attitude_err (numpy ndarray): attitude error
                - attitude_rate_err (numpy ndarray): attitude rate error
                - M_ctrl (numpy ndarray): control torque
                - M_applied (numpy ndarray): applied control torque
                - w_dot_rxwls (numpy ndarray): angular acceleration of
                    reaction wheels
    """
    # get an attitude and angular velocity estimate from the sensors
    w_estimated = satellite.estimate_angular_velocity(t, delta_t)
    DCM_estimated = satellite.estimate_attitude(t, delta_t)

    # -----------------------------------------------------------------------
    # Type 1: Gyro Bias Spike
    # Add a sudden offset to the estimated angular velocity only.
    # The true spacecraft state is not affected; only the controller's
    # perception of w is corrupted, producing incorrect attitude rate error
    # and therefore an incorrect control torque.
    # -----------------------------------------------------------------------
    if active_anomaly is not None and active_anomaly['type'] == 1:
        w_estimated = w_estimated + active_anomaly['gyro_bias_spike']

    # compute the desired attitude and angular velocity
    DCM_desired, w_desired = nominal_state_func(t)

    # calculate the errors between your desired and estimated state
    attitude_err = calculate_attitude_error(DCM_desired, DCM_estimated)
    attitude_rate_err = calculate_attitude_rate_error(w_desired, w_estimated,
                                                      attitude_err)

    # determine the control torques necessary to change state
    M_ctrl = satellite.calculate_control_torques(attitude_err,
                                                 attitude_rate_err)

    # -----------------------------------------------------------------------
    # Type 5: Control Torque Reversal
    # Negate M_ctrl before sending it to the actuators, simulating a
    # controller software fault (e.g. wrong sign convention in the error
    # calculation or a corrupted gain table).  This creates positive feedback:
    # the actuators push the spacecraft further away from the nominal attitude.
    # -----------------------------------------------------------------------
    if active_anomaly is not None and active_anomaly['type'] == 5:
        M_ctrl = -M_ctrl

    # -----------------------------------------------------------------------
    # Type 2: Reaction Wheel Saturation / Failure
    # Zero out M_ctrl so that the actuators receive a zero torque command,
    # simulating a reaction wheel over-temperature protection trip or a
    # driver circuit failure.  The wheels still spin (momentum is conserved)
    # but can no longer accelerate; attitude error grows freely.
    # -----------------------------------------------------------------------
    if active_anomaly is not None and active_anomaly['type'] == 2:
        M_ctrl = np.zeros(3)

    # use actuators to apply the control torques
    M_applied, w_dot_rxwls = satellite.apply_control_torques(
        M_ctrl, t, delta_t)

    if log:
        logged_results = {
            "DCM_estimated": DCM_estimated,
            "w_estimated": w_estimated,
            "DCM_desired": DCM_desired,
            "w_desired": w_desired,
            "attitude_err": attitude_err,
            "attitude_rate_err": attitude_rate_err,
            "M_ctrl": M_ctrl,
            "M_applied": M_applied,
            "w_dot_rxwls": w_dot_rxwls
        }
    else:
        logged_results = None

    return M_applied, w_dot_rxwls, logged_results
