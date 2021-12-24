# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from adam.core.rbd_algorithms import RBDAlgorithms
from adam.jax.spatial_math_jax import SpatialMathJax


class KinDynComputations(RBDAlgorithms, SpatialMathJax):
    """This is a small class that retrieves robot quantities using Jax
    in mixed representation, for Floating Base systems - as humanoid robots.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list,
        root_link: str = "root_link",
        gravity: np.array = jnp.array([0, 0, -9.80665, 0, 0, 0]),
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

    def mass_matrix(self, base_transform: jnp.array, joint_positions: jnp.array):
        """Returns the Mass Matrix functions computed the CRBA

        Args:
            base_transform (jnp.ndarray): The homogenous transform from base to world frame
            joint_positions (jnp.ndarray): The joints position

        Returns:
            M (jnp.ndarray): Mass Matrix
        """
        [M, _] = self.crba(base_transform, joint_positions)
        return M.array

    def centroidal_momentum_matrix(
        self, base_transform: jnp.array, joint_positions: jnp.array
    ):
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Args:
            base_transform (jnp.ndarray): The homogenous transform from base to world frame
            joint_positions (jnp.ndarray): The joints position

        Returns:
            Jcc (jnp.ndarray): Centroidal Momentum matrix
        """
        [_, Jcm] = self.crba(base_transform, joint_positions)
        return Jcm.array

    def relative_jacobian(self, frame: str, joint_positions: jnp.array):
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            joint_positions (jnp.ndarray): The joints position

        Returns:
            J (jnp.ndarray): The Jacobian between the root and the frame
        """
        return super().relative_jacobian(frame, joint_positions)

    def CoM_position(self, base_transform: jnp.array, joint_positions: jnp.array):
        """Returns the CoM positon

        Args:
            base_transform (jnp.ndarray): The homogenous transform from base to world frame
            joint_positions (jnp.ndarray): The joints position

        Returns:
            com (jnp.ndarray): The CoM position
        """
        return super().CoM_position(base_transform, joint_positions)

    def forward_kinematics(
        self, frame: str, base_transform: jnp.array, joint_positions: jnp.array
    ):
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (jnp.ndarray): The homogenous transform from base to world frame
            joint_positions (jnp.ndarray): The joints position

        Returns:
            T_fk (jnp.ndarray): The fk represented as Homogenous transformation matrix
        """
        return super().forward_kinematics(frame, base_transform, joint_positions).array

    def forward_kinematics_fun(self, frame):
        fk_frame = lambda T, joint_positions: self.forward_kinematics(
            frame, T, joint_positions
        )
        return fk_frame

    def jacobian(
        self, frame: str, base_transform: jnp.array, joint_positions: jnp.array
    ):
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            base_transform (jnp.ndarray): The homogenous transform from base to world frame
            s (jnp.ndarray): The joints position
            frame (str): The frame to which the jacobian will be computed

        Returns:
            J (jnp.ndarray): The Jacobian between the root and the frame
        """
        return super().jacobian(frame, base_transform, joint_positions).array

    def rnea(
        self,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        base_velocity: jnp.array,
        joint_velocities: jnp.array,
        g: jnp.array,
    ) -> jnp.array:
        """Implementation of reduced Recursive Newton-Euler algorithm
        (no acceleration and external forces). For now used to compute the bias force term

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position
            base_velocity (jnp.array): The base velocity in mixed representation
            joint_velocities (jnp.array): The joints velocity
            g (jnp.array): The 6D gravity acceleration

        Returns:
            tau (jnp.array): generalized force variables
        """
        return super().rnea(
            base_transform, joint_positions, base_velocity, joint_velocities, g
        )

    def bias_force(
        self,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        base_velocity: jnp.array,
        s_dot: jnp.array,
    ) -> jnp.array:
        """Returns the bias force of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (jnp.ndarray): The homogenous transform from base to world frame
            joint_positions (jnp.ndarray): The joints position
            base_velocity (jnp.ndarray): The base velocity in mixed representation
            s_dot (jnp.ndarray): The joints velocity

        Returns:
            h (jnp.ndarray): the bias force
        """
        h = self.rnea(
            base_transform, joint_positions, base_velocity.reshape(6, 1), s_dot, self.g
        )
        return h.array[:, 0]

    def coriolis_term(
        self,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        base_velocity: jnp.array,
        s_dot: jnp.array,
    ) -> jnp.array:
        """Returns the coriolis term of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (jnp.ndarray): The homogenous transform from base to world frame
            joint_positions (jnp.ndarray): The joints position
            base_velocity (jnp.ndarray): The base velocity in mixed representation
            s_dot (jnp.ndarray): The joints velocity

        Returns:
            C (jnp.ndarray): the Coriolis term
        """
        # set in the bias force computation the gravity term to zero
        C = self.rnea(
            base_transform,
            joint_positions,
            base_velocity.reshape(6, 1),
            s_dot,
            np.zeros(6),
        )
        return C.array[:, 0]

    def gravity_term(
        self, base_transform: jnp.array, joint_positions: jnp.array
    ) -> jnp.array:
        """Returns the gravity term of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (jnp.ndarray): The homogenous transform from base to world frame
            joint_positions (jnp.ndarray): The joints position

        Returns:
            G (jnp.ndarray): the gravity term
        """
        G = self.rnea(
            base_transform,
            joint_positions,
            np.zeros(6).reshape(6, 1),
            np.zeros(self.NDoF),
            self.g,
        )
        return G.array[:, 0]

    def CoM_position(
        self, base_transform: jnp.array, joint_positions: jnp.array
    ) -> jnp.array:
        """Returns the CoM positon

        Args:
            base_transform (jnp.ndarray): The homogenous transform from base to world frame
            joint_positions (jnp.ndarray): The joints position

        Returns:
            com (jnp.ndarray): The CoM position
        """
        return super().CoM_position(base_transform, joint_positions).array
