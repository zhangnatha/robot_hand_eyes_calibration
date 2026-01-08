import numpy as np
import math
from enum import Enum
from ..common.rpy import RPY, RPYType, ReferenceType
from ..common.transforms import Pose3D, Transform3D
import cv2

class CalibrationType3D(Enum):
    ETH = 0 # 眼在手上 (Eye-to-Hand)
    EIH = 1 # 眼在手 (Eye-in-Hand)

class RobotCameraCalibrator3D:
    def __init__(self):
        self.rpy_type = RPYType.XYZ
        self.ref_type = ReferenceType.EXTRINSIC
        self.calibration_error = []
        self.eye_to_fixed = np.eye(4) # 结果矩阵
        self.tcp = np.zeros(3)

    def set_rpy_type(self, rpy_type, ref_type=ReferenceType.EXTRINSIC):
        self.rpy_type = rpy_type
        self.ref_type = ref_type

    def calibrate(self, marks, robot_poses, calib_type, initial_guess=None, initial_tcp=None):
        """
        核心标定函数
        marks: 标记点列表 (List of arrays/lists [x,y,z]) - 在相机坐标系下
        robot_poses: 机器人位姿列表 (List of Pose3D)
        calib_type: ETH 或 EIH
        """
        n = len(marks)
        if n != len(robot_poses):
            print("Error: Marks and Poses count mismatch.")
            return -1.0
        
        if n < 3:
            print("Error: Need at least 3 points.")
            return -1.0

        # 准备数据
        # R_base_end, t_base_end
        Rs = []
        ts = []
        
        # P_cam (marker in cam)
        P_cams = []

        for i in range(n):
            # 将 Robot Pose 转换为 R, t
            # Pose3D to matrix
            pose = robot_poses[i]
            rpy = RPY(pose.rx, pose.ry, pose.rz, self.rpy_type, self.ref_type)
            R = rpy.to_rotation_matrix()
            t = np.array([pose.x, pose.y, pose.z])
            
            Rs.append(R)
            ts.append(t)
            P_cams.append(np.array(marks[i]))

        # 初始化估计
        # 这里为了简化，使用简单的 Ax=B 求解或直接进入迭代优化（因为 Python scipy 优化器很强）
        
        # 我们使用一个简单的迭代优化器来最小化重投影误差
        # 变量: X (Eye-to-Fixed, 6 params), Y (TCP/Tool offset if needed?)
        # 3D 标定通常解 X (Camera relative to Flange or Base)
        
        # 对于 ETH (Eye-to-Hand): Camera 也是 Fixed near Base
        # Robot moving, Camera sees Marker on Flange
        # T_base_cam * P_cam = T_base_flange * P_flange_marker
        # 我们不知道 P_flange_marker (TCP offset)
        # 所以我们需要同时解 Extrinsic (T_base_cam) 和 TCP offset
        
        # 对于 EIH (Eye-in-Hand): Camera on Flange
        # Robot moving, Camera sees Marker fixed in World
        # T_base_flange * T_flange_cam * P_cam = P_world
        # 我们不知道 P_world
        # 所以同样需要解 Extrinsic (T_flange_cam) 和 P_world
        
        # 这是一个经典的 AX=XB 问题变体 (Hand-Eye), 但这里是 Point-based
        # 我们使用非线性最小二乘法 (Levenberg-Marquardt)
        
        from scipy.optimize import least_squares
        
        def transform(rvec, tvec, pts):
            # Apply R, t to pts
            # pts: (N, 3)
            # rvec: (3,)
            # tvec: (3,)
            R, _ = cv2.Rodrigues(rvec)
            return (R @ pts.T).T + tvec
            
        def residuals(params):
            # params: [rx, ry, rz, tx, ty, tz,  mx, my, mz]
            # 前 6 个是 Extrinsic (Camera pose), 后 3 个是 Unknown Marker/TCP offset
            
            # Unpack Extrinsic
            e_rvec = params[:3]
            e_tvec = params[3:6]
            
            # Unpack Unknown Point (TCP or Base Marker)
            u_pt = params[6:9]
            
            errs = []
            
            if calib_type == CalibrationType3D.ETH:
                # Eye-to-Hand
                # T_cam_base * P_cam = T_flange_base^-1 * P_tcp (WRONG chain)
                # Correct: P_base = T_base_flange * P_tcp
                #          P_base = T_base_cam * P_cam
                # => T_base_flange * P_tcp = T_base_cam * P_cam
                # 我们想求 T_base_cam (或者 T_cam_base)
                # 设 Params 为 T_base_cam
                
                # P_base_est1 = T_base_flange[i] * u_pt (u_pt is P_tcp)
                # P_base_est2 = Transform(e_rvec, e_tvec, P_cam[i])
                
                # 误差 = P_base_est1 - P_base_est2
                
                # Precompute T_base_flange applied to u_pt
                # Rs[i], ts[i] is T_base_flange
                
                for i in range(n):
                    p_rob = Rs[i] @ u_pt + ts[i] # Robot end carrying marker
                    
                    R_cal, _ = cv2.Rodrigues(e_rvec)
                    p_cam_transformed = R_cal @ P_cams[i] + e_tvec
                    
                    diff = p_rob - p_cam_transformed
                    errs.extend(diff)
                    
            else: 
                # Eye-in-Hand
                # Camera on Flange.
                # P_world = T_base_flange * T_flange_cam * P_cam
                # u_pt is P_world
                # Params is T_flange_cam
                
                for i in range(n):
                    R_cal, _ = cv2.Rodrigues(e_rvec)
                    p_cam_in_flange = R_cal @ P_cams[i] + e_tvec
                    
                    p_cam_in_world = Rs[i] @ p_cam_in_flange + ts[i]
                    
                    diff = p_cam_in_world - u_pt
                    errs.extend(diff)
            
            return np.array(errs)

        # 初始猜测
        x0 = np.zeros(9)
        if initial_guess is not None:
             # Extract rvec, tvec from initial matrix
             rvec, _ = cv2.Rodrigues(initial_guess[:3,:3])
             x0[:3] = rvec.flatten()
             x0[3:6] = initial_guess[:3,3]
        else:
             x0[5] = 500 # Guess Z=500mm
             
        if initial_tcp is not None:
             x0[6:9] = initial_tcp
             
        res = least_squares(residuals, x0, method='lm')
        
        # 解析结果
        opt_params = res.x
        e_rvec = opt_params[:3]
        e_tvec = opt_params[3:6]
        u_pt = opt_params[6:9]
        
        R_res, _ = cv2.Rodrigues(e_rvec)
        T_res = np.eye(4)
        T_res[:3, :3] = R_res
        T_res[:3, 3] = e_tvec
        
        self.eye_to_fixed = T_res
        self.tcp = u_pt
        
        # 计算最终误差
        final_residuals = residuals(opt_params)
        # residuals 是一维数组 [dx0, dy0, dz0, dx1 ...]
        # 重组为 (N, 3)
        reshaped_res = final_residuals.reshape(-1, 3)
        self.calibration_error = np.linalg.norm(reshaped_res, axis=1).tolist()
        
        rmse = np.sqrt(np.mean(np.square(final_residuals)))
        return rmse

    def get_result(self):
        return self.eye_to_fixed, self.tcp

    def getCalibrationError(self):
        return self.calibration_error
