# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import numpy as np
import torch

from adam.core.rbd_algorithms import RBDAlgorithms
from adam.pytorch.spatial_math_pytorch import SpatialMathPytorch


class KinDynComputations(RBDAlgorithms, SpatialMathPytorch):
    """This is a small class that retrieves robot quantities using Pytorch
    in mixed representation, for Floating Base systems - as humanoid robots.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list,
        root_link: str = "root_link",
        gravity: np.array = torch.FloatTensor([0, 0, -9.80665, 0, 0, 0]),
    ) -> None:
        """
        Args:
            urdfstring (str): path of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): the first link. Defaults to 'root_link'.
        """
        super().__init__(
            urdfstring=urdfstring,
            joints_name_list=joints_name_list,
            root_link=root_link,
            gravity=gravity,
        )

    def mass_matrix(self, base_transform, s):
        """Returns the Mass Matrix functions computed the CRBA

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            s (torch.tensor): The joints position

        Returns:
            M (torch.tensor): Mass Matrix
        """
        [M, _] = super().crba(base_transform, s)
        return M

    def centroidal_momentum_matrix(self, base_transform, s):
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            s (torch.tensor): The joints position

        Returns:
            Jcc (torch.tensor): Centroidal Momentum matrix
        """
        [_, Jcm] = super().crba(base_transform, s)
        return Jcm

    def forward_kinematics(self, frame, base_transform, s):
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (torch.tensor): The homogenous transform from base to world frame
            s (torch.tensor): The joints position

        Returns:
            T_fk (torch.tensor): The fk represented as Homogenous transformation matrix
        """
        return super().forward_kinematics(
            frame, torch.FloatTensor(base_transform), torch.FloatTensor(s)
        )

    def jacobian(self, frame, base_transform, joint_positions):
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position

        Returns:
            J_tot (torch.tensor): The Jacobian relative to the frame
        """
        chain = self.robot_desc.get_chain(self.root_link, frame)
        T_fk = self.eye(4)
        T_fk = T_fk @ base_transform
        J = self.zeros(6, self.NDoF)
        T_ee = self.forward_kinematics(frame, base_transform, joint_positions)
        P_ee = T_ee[:3, 3]
        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.type == "fixed":
                    xyz = joint.origin.xyz
                    rpy = joint.origin.rpy
                    joint_frame = self.H_from_Pos_RPY(xyz, rpy)
                    T_fk = T_fk @ joint_frame
                if joint.type == "revolute" or joint.type == "continuous":
                    if joint.idx is not None:
                        q_ = joint_positions[joint.idx]
                    else:
                        q_ = 0.0
                    T_joint = self.H_revolute_joint(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis,
                        q_,
                    )
                    T_fk = T_fk @ T_joint
                    p_prev = P_ee - T_fk[:3, 3]
                    z_prev = T_fk[:3, :3] @ torch.tensor(joint.axis)
                    if joint.idx is not None:
                        stack = np.hstack(((self.skew(z_prev) @ p_prev), z_prev))
                        J[:, joint.idx] = torch.tensor(stack)

        # Adding the floating base part of the Jacobian, in Mixed representation
        J_tot = self.zeros(6, self.NDoF + 6)
        J_tot[:3, :3] = self.eye(3)
        J_tot[:3, 3:6] = -self.skew((P_ee - base_transform[:3, 3]))
        J_tot[:3, 6:] = J[:3, :]
        J_tot[3:, 3:6] = self.eye(3)
        J_tot[3:, 6:] = J[3:, :]
        return J_tot

    def relative_jacobian(self, frame, joint_positions):
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            joint_positions (torch.tensor): The joints position

        Returns:
            J (torch.tensor): The Jacobian between the root and the frame
        """
        chain = self.robot_desc.get_chain(self.root_link, frame)
        base_transform = self.eye(4)
        T_fk = self.eye(4)
        T_fk = T_fk @ base_transform
        J = self.zeros(6, self.NDoF)
        T_ee = self.forward_kinematics(frame, base_transform, joint_positions)
        P_ee = T_ee[:3, 3]
        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.type == "fixed":
                    xyz = joint.origin.xyz
                    rpy = joint.origin.rpy
                    joint_frame = self.H_from_Pos_RPY(xyz, rpy)
                    T_fk = T_fk @ joint_frame
                if joint.type == "revolute" or joint.type == "continuous":
                    if joint.idx is not None:
                        q_ = joint_positions[joint.idx]
                    else:
                        q_ = 0.0
                    T_joint = self.H_revolute_joint(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis,
                        q_,
                    )
                    T_fk = T_fk @ T_joint
                    p_prev = P_ee - T_fk[:3, 3]
                    z_prev = T_fk[:3, :3] @ torch.tensor(joint.axis)
                    if joint.idx is not None:
                        stack = np.hstack(((self.skew(z_prev) @ p_prev), z_prev))
                        J[:, joint.idx] = torch.tensor(stack)
        return J

    def CoM_position(self, base_transform, joint_positions):
        """Returns the CoM positon

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position

        Returns:
            com (torch.tensor): The CoM position
        """
        return super().CoM_position(base_transform, joint_positions)

    def bias_force(self, base_transform, s, base_velocity, joint_velocities):
        """Returns the bias force of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            s (torch.tensor): The joints position
            base_velocity (torch.tensor): The base velocity in mixed representation
            joint_velocities (torch.tensor): The joints velocity

        Returns:
            h (torch.tensor): the bias force
        """
        h = super().rnea(
            base_transform, s, base_velocity.reshape(6, 1), joint_velocities, self.g
        )
        return h[:, 0]

    def coriolis_term(
        self, base_transform, joint_positions, base_velocity, joint_velocities
    ):
        """Returns the coriolis term of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position
            base_velocity (torch.tensor): The base velocity in mixed representation
            joint_velocities (torch.tensor): The joints velocity

        Returns:
            C (torch.tensor): the Coriolis term
        """
        # set in the bias force computation the gravity term to zero
        C = super().rnea(
            base_transform,
            joint_positions,
            base_velocity.reshape(6, 1),
            joint_velocities,
            torch.zeros(6),
        )
        return C[:, 0]

    def gravity_term(self, base_transform, base_positions):
        """Returns the gravity term of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            base_positions (torch.tensor): The joints position

        Returns:
            G (torch.tensor): the gravity term
        """
        G = super().rnea(
            base_transform,
            base_positions,
            torch.zeros(6).reshape(6, 1),
            torch.zeros(self.NDoF),
            torch.FloatTensor(self.g),
        )
        return G[:, 0]
