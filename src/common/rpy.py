import numpy as np
from enum import Enum, auto

class RPYType(Enum):
    XYZ = 0
    XZY = 1
    YXZ = 2
    YZX = 3
    ZXY = 4
    ZYX = 5
    XYX = 6
    XZX = 7
    YXY = 8
    YZY = 9
    ZXZ = 10
    ZYZ = 11
    ROT_VECTOR = 12 

class ReferenceType(Enum):
    INTRINSIC = 0
    EXTRINSIC = 1

class RPY:
    def __init__(self, rx=0.0, ry=0.0, rz=0.0, rpy_type=RPYType.XYZ, ref_type=ReferenceType.EXTRINSIC):
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.rpy_type = rpy_type
        self.ref_type = ref_type

    def to_rotation_matrix(self):
        # 内部将角度转换为弧度进行三角计算
        
        if self.rpy_type == RPYType.ROT_VECTOR:
            vec = np.array([self.rx, self.ry, self.rz])
            angle = np.linalg.norm(vec)
            if angle < 1e-10:
                return np.eye(3)
            # 罗德里格斯公式
            axis = vec / angle
            K = np.array([[0, -axis[2], axis[1]], 
                          [axis[2], 0, -axis[0]], 
                          [-axis[1], axis[0], 0]])
            I = np.eye(3)
            R = I + np.sin(angle)*K + (1-np.cos(angle)) * (K @ K)
            return R

        # 转换为弧度
        r1 = np.deg2rad(self.rx)
        r2 = np.deg2rad(self.ry)
        r3 = np.deg2rad(self.rz)
        
        c1, s1 = np.cos(r1), np.sin(r1)
        c2, s2 = np.cos(r2), np.sin(r2)
        c3, s3 = np.cos(r3), np.sin(r3)
        
        # 基本旋转矩阵辅助函数
        def RotX(c, s):
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        def RotY(c, s):
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        def RotZ(c, s):
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        Rm1, Rm2, Rm3 = None, None, None

        if self.rpy_type == RPYType.XYZ:
            Rm1, Rm2, Rm3 = RotX(c1, s1), RotY(c2, s2), RotZ(c3, s3)
        elif self.rpy_type == RPYType.XZY:
            Rm1, Rm2, Rm3 = RotX(c1, s1), RotZ(c2, s2), RotY(c3, s3)
        elif self.rpy_type == RPYType.YXZ:
            Rm1, Rm2, Rm3 = RotY(c1, s1), RotX(c2, s2), RotZ(c3, s3)
        elif self.rpy_type == RPYType.YZX:
            Rm1, Rm2, Rm3 = RotY(c1, s1), RotZ(c2, s2), RotX(c3, s3)
        elif self.rpy_type == RPYType.ZXY:
            Rm1, Rm2, Rm3 = RotZ(c1, s1), RotX(c2, s2), RotY(c3, s3)
        elif self.rpy_type == RPYType.ZYX:
            Rm1, Rm2, Rm3 = RotZ(c1, s1), RotY(c2, s2), RotX(c3, s3)
        
        if self.ref_type == ReferenceType.EXTRINSIC:
             # m = r3 * r2 * r1
             if Rm1 is not None:
                return Rm3 @ Rm2 @ Rm1
        else:
             # m = r1 * r2 * r3
             if Rm1 is not None:
                return Rm1 @ Rm2 @ Rm3
                
        return np.eye(3)

    @staticmethod
    def from_rotation_matrix(R, rpy_type=RPYType.XYZ, ref_type=ReferenceType.EXTRINSIC):
        eps = 1e-9
        r1, r2, r3 = 0, 0, 0
        
        if rpy_type == RPYType.ROT_VECTOR:
            # 使用手动罗德里格斯逆变换
            # 简单的 trace 方法
            tr = np.trace(R)
            arg = (tr - 1) / 2.0
            arg = np.clip(arg, -1, 1)
            angle = np.arccos(arg)
            if angle < 1e-10:
                return RPY(0,0,0, rpy_type, ref_type)
            
            fac = 1.0 / (2 * np.sin(angle))
            rx = (R[2,1] - R[1,2]) * fac * angle
            ry = (R[0,2] - R[2,0]) * fac * angle
            rz = (R[1,0] - R[0,1]) * fac * angle
            return RPY(rx, ry, rz, rpy_type, ref_type)
            
        if rpy_type == RPYType.XYZ or rpy_type == RPYType.ZYX:
             is_xyz_ext = (rpy_type == RPYType.XYZ and ref_type == ReferenceType.EXTRINSIC)
             is_zyx_int = (rpy_type == RPYType.ZYX and ref_type == ReferenceType.INTRINSIC)
             
             if is_xyz_ext or is_zyx_int:
                 # r2 = atan2(-r(2, 0), sqrt(r(0, 0)^2 + r(1, 0)^2))
                 sy = np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
                 r2 = np.arctan2(-R[2,0], sy)
                 
                 # 处理奇异点
                 if r2 < np.pi/2 + eps and r2 > np.pi/2 - eps:
                     r2 = np.pi/2
                     r1 = np.arctan2(R[0,1], R[1,1]) 
                     r3 = 0
                 elif r2 > -np.pi/2 - eps and r2 < -np.pi/2 + eps:
                     r2 = -np.pi/2
                     r1 = -np.arctan2(R[0,1], R[1,1])
                     r3 = 0
                 else:
                     c2 = np.cos(r2)
                     r3 = np.arctan2(R[1,0]/c2, R[0,0]/c2)
                     r1 = np.arctan2(R[2,1]/c2, R[2,2]/c2)
                     
                 # 转换为度数
                 return RPY(np.rad2deg(r1), np.rad2deg(r2), np.rad2deg(r3), rpy_type, ref_type)

        # 默认返回
        return RPY(0,0,0, rpy_type, ref_type)
