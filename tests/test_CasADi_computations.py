# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import logging

import casadi as cs
import gym_ignition_models
import idyntree.swig as idyntree
import numpy as np
import pytest

from adam.casadi.computations import KinDynComputations
from adam.geometry import utils

model_path = gym_ignition_models.get_model_file("iCubGazeboV2_5")

joints_name_list = [
    "torso_pitch",
    "torso_roll",
    "torso_yaw",
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_shoulder_yaw",
    "l_elbow",
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow",
    "l_hip_pitch",
    "l_hip_roll",
    "l_hip_yaw",
    "l_knee",
    "l_ankle_pitch",
    "l_ankle_roll",
    "r_hip_pitch",
    "r_hip_roll",
    "r_hip_yaw",
    "r_knee",
    "r_ankle_pitch",
    "r_ankle_roll",
]


def SX2DM(x):
    return cs.DM(x)


def H_from_Pos_RPY_idyn(xyz, rpy):
    T = idyntree.Transform.Identity()
    R = idyntree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
    p = idyntree.Position()
    [p.setVal(i, xyz[i]) for i in range(3)]
    T.setRotation(R)
    T.setPosition(p)
    return T


logging.basicConfig(level=logging.DEBUG)
logging.debug("Showing the robot tree.")

root_link = "root_link"
comp = KinDynComputations(model_path, joints_name_list, root_link)
robot_iDyn = idyntree.ModelLoader()
robot_iDyn.loadReducedModelFromFile(model_path, joints_name_list)

kinDyn = idyntree.KinDynComputations()
kinDyn.loadRobotModel(robot_iDyn.model())
kinDyn.setFloatingBase(root_link)
kinDyn.setFrameVelocityRepresentation(idyntree.MIXED_REPRESENTATION)
n_dofs = len(joints_name_list)

# base pose quantities
xyz = (np.random.rand(3) - 0.5) * 5
rpy = (np.random.rand(3) - 0.5) * 5
base_vel = (np.random.rand(6) - 0.5) * 5
# joints quantitites
joints_val = (np.random.rand(n_dofs) - 0.5) * 5
joints_dot_val = (np.random.rand(n_dofs) - 0.5) * 5

# set iDynTree kinDyn
H_b_idyn = H_from_Pos_RPY_idyn(xyz, rpy)
vb = idyntree.Twist()
[vb.setVal(i, base_vel[i]) for i in range(6)]

s = idyntree.VectorDynSize(n_dofs)
[s.setVal(i, joints_val[i]) for i in range(n_dofs)]
s_dot = idyntree.VectorDynSize(n_dofs)
[s_dot.setVal(i, joints_dot_val[i]) for i in range(n_dofs)]

g = idyntree.Vector3()
g.zero()
g.setVal(2, -9.80665)
kinDyn.setRobotState(H_b_idyn, s, vb, s_dot, g)
# set ADAM
H_b = utils.H_from_Pos_RPY(xyz, rpy)
vb_ = base_vel
s_ = joints_val
s_dot_ = joints_dot_val


def test_mass_matrix():
    M = comp.mass_matrix_fun()
    mass_mx = idyntree.MatrixDynSize()
    kinDyn.getFreeFloatingMassMatrix(mass_mx)
    mass_mxNumpy = mass_mx.toNumPy()
    mass_test = SX2DM(M(H_b, s_))
    assert mass_test - mass_mxNumpy == pytest.approx(0.0, abs=1e-5)


def test_CMM():
    Jcm = comp.centroidal_momentum_matrix_fun()
    cmm_idyntree = idyntree.MatrixDynSize()
    kinDyn.getCentroidalTotalMomentumJacobian(cmm_idyntree)
    cmm_idyntreeNumpy = cmm_idyntree.toNumPy()
    Jcm_test = SX2DM(Jcm(H_b, s_))
    assert Jcm_test - cmm_idyntreeNumpy == pytest.approx(0.0, abs=1e-5)


def test_CoM_pos():
    com_f = comp.CoM_position_fun()
    CoM_cs = SX2DM(com_f(H_b, s_))
    CoM_iDynTree = kinDyn.getCenterOfMassPosition().toNumPy()
    assert CoM_cs - CoM_iDynTree == pytest.approx(0.0, abs=1e-5)


def test_total_mass():
    assert comp.get_total_mass() - robot_iDyn.model().getTotalMass() == pytest.approx(
        0.0, abs=1e-5
    )


def test_jacobian():
    J_tot = comp.jacobian_fun("l_sole")
    iDyntreeJ_ = idyntree.MatrixDynSize(6, n_dofs + 6)
    kinDyn.getFrameFreeFloatingJacobian("l_sole", iDyntreeJ_)
    iDynNumpyJ_ = iDyntreeJ_.toNumPy()
    J_test = SX2DM(J_tot(H_b, s_))
    assert iDynNumpyJ_ - J_test == pytest.approx(0.0, abs=1e-5)


def test_jacobian_non_actuated():
    J_tot = comp.jacobian_fun("head")
    iDyntreeJ_ = idyntree.MatrixDynSize(6, n_dofs + 6)
    kinDyn.getFrameFreeFloatingJacobian("head", iDyntreeJ_)
    iDynNumpyJ_ = iDyntreeJ_.toNumPy()
    J_test = SX2DM(J_tot(H_b, s_))
    assert iDynNumpyJ_ - J_test == pytest.approx(0.0, abs=1e-5)


def test_fk():
    H_idyntree = kinDyn.getWorldTransform("l_sole")
    p_idy2np = H_idyntree.getPosition().toNumPy()
    R_idy2np = H_idyntree.getRotation().toNumPy()
    T = comp.forward_kinematics_fun("l_sole")
    H_test = SX2DM(T(H_b, s_))
    assert R_idy2np - H_test[:3, :3] == pytest.approx(0.0, abs=1e-5)
    assert p_idy2np - H_test[:3, 3] == pytest.approx(0.0, abs=1e-5)


def test_fk_non_actuated():
    H_idyntree = kinDyn.getWorldTransform("head")
    p_idy2np = H_idyntree.getPosition().toNumPy()
    R_idy2np = H_idyntree.getRotation().toNumPy()
    T = comp.forward_kinematics_fun("head")
    H_test = SX2DM(T(H_b, s_))
    assert R_idy2np - H_test[:3, :3] == pytest.approx(0.0, abs=1e-5)
    assert p_idy2np - H_test[:3, 3] == pytest.approx(0.0, abs=1e-5)


def test_bias_force():
    h_iDyn = idyntree.FreeFloatingGeneralizedTorques(kinDyn.model())
    assert kinDyn.generalizedBiasForces(h_iDyn)
    h_iDyn_np = np.concatenate(
        (h_iDyn.baseWrench().toNumPy(), h_iDyn.jointTorques().toNumPy())
    )
    h = comp.bias_force_fun()
    h_test = SX2DM(h(H_b, s_, vb_, s_dot_))
    assert h_iDyn_np - h_test == pytest.approx(0.0, abs=1e-4)


def test_coriolis_term():
    g0 = idyntree.Vector3()
    g0.zero()
    kinDyn.setRobotState(H_b_idyn, s, vb, s_dot, g0)
    C_iDyn = idyntree.FreeFloatingGeneralizedTorques(kinDyn.model())
    assert kinDyn.generalizedBiasForces(C_iDyn)
    C_iDyn_np = np.concatenate(
        (C_iDyn.baseWrench().toNumPy(), C_iDyn.jointTorques().toNumPy())
    )
    C = comp.coriolis_term_fun()
    C_test = SX2DM(C(H_b, s_, vb_, s_dot_))
    assert C_iDyn_np - C_test == pytest.approx(0.0, abs=1e-4)


def test_gravity_term():
    vb0 = idyntree.Twist()
    s_dot0 = idyntree.VectorDynSize(n_dofs)
    kinDyn.setRobotState(H_b_idyn, s, vb0, s_dot0, g)
    G_iDyn = idyntree.FreeFloatingGeneralizedTorques(kinDyn.model())
    assert kinDyn.generalizedBiasForces(G_iDyn)
    G_iDyn_np = np.concatenate(
        (G_iDyn.baseWrench().toNumPy(), G_iDyn.jointTorques().toNumPy())
    )
    G = comp.gravity_term_fun()
    G_test = SX2DM(G(H_b, s_))
    assert G_iDyn_np - G_test == pytest.approx(0.0, abs=1e-4)
