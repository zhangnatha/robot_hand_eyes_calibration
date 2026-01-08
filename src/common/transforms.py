import numpy as np
from .rpy import RPY, RPYType, ReferenceType

class Pose3D:
    def __init__(self, x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.rx = rx
        self.ry = ry
        self.rz = rz

    @staticmethod
    def from_transform(T):
        """从 4x4 变换矩阵创建 Pose3D"""
        x = T[0, 3]
        y = T[1, 3]
        z = T[2, 3]
        
        R = T[:3, :3]
        rpy = RPY.from_rotation_matrix(R, RPYType.XYZ, ReferenceType.EXTRINSIC)
        
        return Pose3D(x, y, z, rpy.rx, rpy.ry, rpy.rz)

    def to_transform_matrix(self):
        """转换为 4x4 齐次变换矩阵"""
        T = np.eye(4)
        T[0, 3] = self.x
        T[1, 3] = self.y
        T[2, 3] = self.z
        
        rpy = RPY(self.rx, self.ry, self.rz, RPYType.XYZ, ReferenceType.EXTRINSIC)
        R = rpy.to_rotation_matrix()
        T[:3, :3] = R
        return T

class Transform3D:
    def __init__(self, matrix=None):
        if matrix is None:
            self.matrix = np.eye(4)
        else:
            self.matrix = matrix

    def translate(self, x, y, z):
        T = np.eye(4)
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z
        self.matrix = self.matrix @ T

    def rotate(self, rpy):
        R = rpy.to_rotation_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        self.matrix = self.matrix @ T
