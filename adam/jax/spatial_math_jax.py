# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import dataclasses
from typing import List

import casadi as cs
import jax.numpy as jnp
import numpy as np
from icecream import ic

from adam.core.spatial_math import SpatialMathAbstract


@dataclasses.dataclass
class JaxArray:
    array: jnp.array

    def __setitem__(self, idx, value):
        if type(self) is type(value):
            try:
                self.array = self.array.at[idx].set(value.array)
            except:
                if value.shape == (1, 1):
                    value.array = value.array.reshape()
                elif 1 in value.shape:
                    value.array = value.array.reshape(-1)
                self.array = self.array.at[idx].set(value.array)
        elif isinstance(value, List) or len(value.shape) == 2:
            self.array = self.array.at[idx].set(value)
        else:
            value = value.reshape(-1, 1)
            self.array = self.array.at[idx].set(value)

    def __getitem__(self, idx):
        return JaxArray(self.array[idx])

    @property
    def shape(self):
        return self.array.shape

    def reshape(self, *args):
        return self.array.reshape(*args)

    @property
    def T(self):
        return JaxArray(self.array.T)

    def __matmul__(self, other):
        if type(self) is type(other):
            return JaxArray(self.array @ other.array)
        else:
            return JaxArray(self.array @ jnp.array(other))

    def __rmatmul__(self, other):
        if type(self) is type(other):
            return JaxArray(other.array @ self.array)
        else:
            return JaxArray(jnp.array(other) @ self.array)

    def __mul__(self, other):
        if type(self) is type(other):
            return JaxArray(self.array * other.array)
        else:
            return JaxArray(self.array * other)

    def __truediv__(self, other):
        if type(self) is type(other):
            return JaxArray(self.array / other.array)
        else:
            return JaxArray(self.array / other)

    def __add__(self, other):
        if type(self) is type(other):
            return JaxArray(self.array + other.array)
        else:
            return JaxArray(self.array + other)

    def __sub__(self, other):
        if type(self) is type(other):
            return JaxArray(self.array - other.array)
        else:
            return JaxArray(self.array - other)

    def __rsub__(self, other):
        if type(self) is type(other):
            return JaxArray(other.array - self.array)
        else:
            return JaxArray(other - self.array)

    def __call__(self):
        return self.array

    def __neg__(self):
        return JaxArray(-self.array)


class SpatialMathJax(SpatialMathAbstract):
    @classmethod
    def R_from_axis_angle(cls, axis, q):
        cq, sq = jnp.cos(q), jnp.sin(q)
        return (
            cq * (jnp.eye(3) - jnp.outer(np.array(axis), np.array(axis)))
            + sq * cls.skew(axis)
            + jnp.outer(np.array(axis), np.array(axis))
        )

    @classmethod
    def Rx(cls, q):
        R = jnp.eye(3)
        cq, sq = jnp.cos(q), jnp.sin(q)
        R = R.at[1, 1].set(cq)
        R = R.at[1, 2].set(-sq)
        R = R.at[2, 1].set(sq)
        R = R.at[2, 2].set(cq)
        return R

    @classmethod
    def Ry(cls, q):
        R = jnp.eye(3)
        cq, sq = jnp.cos(q), jnp.sin(q)
        R = R.at[0, 0].set(cq)
        R = R.at[0, 2].set(sq)
        R = R.at[2, 0].set(-sq)
        R = R.at[2, 2].set(cq)
        return R

    @classmethod
    def Rz(cls, q):
        R = jnp.eye(3)
        cq, sq = jnp.cos(q), jnp.sin(q)
        R = R.at[0, 0].set(cq)
        R = R.at[0, 1].set(-sq)
        R = R.at[1, 0].set(sq)
        R = R.at[1, 1].set(cq)
        return R

    @classmethod
    def H_revolute_joint(cls, xyz, rpy, axis, q):
        T = JaxArray(jnp.eye(4))
        R = cls.R_from_RPY(rpy) @ cls.R_from_axis_angle(axis, q)
        T[:3, :3] = R
        T[:3, 3] = xyz
        # T = T.at[:3, :3].set(R)
        # T = T.at[:3, 3].set(xyz)
        return T

    @classmethod
    def H_from_Pos_RPY(cls, xyz, rpy):
        T = jnp.eye(4)
        T = T.at[:3, :3].set(cls.R_from_RPY(rpy))
        T = T.at[:3, 3].set(xyz)
        return JaxArray(T)

    @classmethod
    def R_from_RPY(cls, rpy):
        return cls.Rz(rpy[2]) @ cls.Ry(rpy[1]) @ cls.Rx(rpy[0])

    @classmethod
    def X_revolute_joint(cls, xyz, rpy, axis, q):
        T = cls.H_revolute_joint(xyz, rpy, axis, q)
        R = T.array[:3, :3].T
        p = -T.array[:3, :3].T @ T.array[:3, 3]
        return cls.spatial_transform(R, p)

    @classmethod
    def X_fixed_joint(cls, xyz, rpy):
        T = cls.H_from_Pos_RPY(xyz, rpy)
        R = T[:3, :3].T
        p = -T[:3, :3].T @ T[:3, 3]
        return cls.spatial_transform(R, p)

    @classmethod
    def spatial_transform(cls, R, p):
        X = cls.zeros(6, 6)
        X[:3, :3] = R
        X[3:, 3:] = R
        if isinstance(R, JaxArray):
            R = R.array
        X[:3, 3:] = cls.skew(p) @ R

        # X = X.at[:3, :3].set(R)
        # X = X.at[3:, 3:].set(R)
        # X = X.at[:3, 3:].set(cls.skew(p) @ R)
        return X

    @classmethod
    def spatial_inertia(cls, I, mass, c, rpy):
        # Returns the 6x6 inertia matrix expressed at the origin of the link (with rotation)"""
        IO = jnp.zeros([6, 6])
        Sc = cls.skew(c)
        R = cls.R_from_RPY(rpy)
        inertia_matrix = np.array(
            [[I.ixx, I.ixy, I.ixz], [I.ixy, I.iyy, I.iyz], [I.ixz, I.iyz, I.izz]]
        )
        IO = IO.at[:3, :3].set(jnp.eye(3) * mass)
        IO = IO.at[3:, 3:].set(R @ inertia_matrix @ R.T + mass * Sc @ Sc.T)
        IO = IO.at[3:, :3].set(mass * Sc)
        IO = IO.at[:3, 3:].set(mass * Sc.T)
        IO = IO.at[:3, :3].set(np.eye(3) * mass)
        return JaxArray(IO)

    @classmethod
    def spatial_skew(cls, v):
        X = cls.zeros(6, 6)
        X[:3, :3] = cls.skew(v[3:])
        X[:3, 3:] = cls.skew(v[:3])
        X[3:, 3:] = cls.skew(v[3:])
        # X = X.at[:3, :3].set(cls.skew(v[3:]))
        # X = X.at[:3, 3:].set(cls.skew(v[:3]))
        # X = X.at[3:, 3:].set(cls.skew(v[3:]))
        return X

    @classmethod
    def spatial_skew_star(cls, v):
        return -cls.spatial_skew(v).T

    # @classmethod
    # def skew(cls, x):
    #     if isinstance(x, JaxArray):
    #         x = x.array
    #     return jnp.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

    @staticmethod
    def zeros(*x):
        return JaxArray(jnp.zeros(x))

    @staticmethod
    def vertcat(*x):
        if isinstance(x[0], JaxArray):
            v = jnp.vstack([x[i].array for i in range(len(x))]).reshape(-1, 1)
        else:
            v = jnp.vstack([x[i] for i in range(len(x))]).reshape(-1, 1)
        # This check is needed since vercat is used for two types of data structure in RBDAlgo class.
        # CasADi handles the cases smootly, with NumPy I need to handle the two cases.
        # It should be improved
        # if v.shape[1] > 1:
        #     v = jnp.concatenate(x)
        return JaxArray(v)

    @staticmethod
    def eye(x):
        return JaxArray(jnp.eye(x))

    @staticmethod
    def skew(x):
        if not isinstance(x, JaxArray):
            return -jnp.cross(jnp.array(x), jnp.eye(3), axisa=0, axisb=0)
        x = x.array
        return JaxArray(-jnp.cross(jnp.array(x), jnp.eye(3), axisa=0, axisb=0))

    @staticmethod
    def array(*x):
        return JaxArray(jnp.array(x))


if __name__ == "__main__":
    from icecream import ic

    a = JaxArray(jnp.zeros([3, 2]))
    ic(a)
    a[1, 1] = JaxArray(jnp.array(1).reshape(1, 1))
    ic(a)
