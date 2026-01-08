import cv2
import numpy as np
import math
from enum import Enum
from ..common.transforms import Pose3D

class CameraInstallType(Enum):
    SameToTCPZ = 0 # 相机 Z 轴与 TCP Z 轴方向相同
    Opposite = 1   # 相机 Z 轴与 TCP Z 轴方向相反

class CalibrationType2D(Enum):
    ETH = 0 # 眼在手上 (Eye-to-Hand)
    EIH = 1 # 眼在手 (Eye-in-Hand)

class TwelvePointCalibrator:
    def __init__(self):
        self.image_mark_points = []
        self.robot_poses = []
        self.registration_point = np.zeros(3) # 拍照点
        
        self.hand_eye_type = CalibrationType2D.EIH
        self.camera_install_type = CameraInstallType.SameToTCPZ
        
        self.transform_pose = np.eye(3) # 仿射变换矩阵 (2x3 in 3x3 container)
        self.rotation_center = np.zeros(2) # 旋转中心 (x, y)
        self.fitting_radius = 0.0
        self.errors = []
        
        self._raw_tool_center = np.zeros(2) # 图像坐标系下的旋转中心

    def set_image_mark_points(self, points):
        self.image_mark_points = points

    def set_robot_poses(self, poses):
        self.robot_poses = poses

    def set_registration_point(self, pt):
        self.registration_point = np.array(pt, dtype=np.float64)

    def set_hand_eye_type(self, t):
        self.hand_eye_type = t
        
    def set_camera_install_type(self, t):
        self.camera_install_type = t
        
    def run_calibration(self):
        if len(self.image_mark_points) != 12 or len(self.robot_poses) != 12:
             print("Please provide exactly 12 points and poses.")
             return False
        
        # 0. 准备数据
        # Poses 0-8 (9 points) used for Affine
        # Poses 9-11 (3 points) used for Rotation Center
        
        image_9points = []
        robot_9points = []
        angles = [0.0] * 12

        # 处理 EIH 的特殊顺序
        # 重排逻辑！
        
        idx_map = []
        if self.hand_eye_type == CalibrationType2D.EIH:
            # EIH order: 0, 5, 6, 7, 8, 1, 2, 3, 4
            idx_map = [0, 5, 6, 7, 8, 1, 2, 3, 4]
        else:
            if self.camera_install_type == CameraInstallType.SameToTCPZ:
                idx_map = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            else:
                idx_map = [0, 1, 8, 7, 6, 5, 4, 3, 2]
                
        for i in range(9):
            real_idx = idx_map[i]
            p = self.image_mark_points[real_idx]
            image_9points.append([p[0], p[1]])
            angles[i] = p[2] if len(p) > 2 else 0.0
            
            r = self.robot_poses[i]
            robot_9points.append([r[0], r[1]])
            
        img_pts_np = np.array(image_9points, dtype=np.float32)
        rob_pts_np = np.array(robot_9points, dtype=np.float32)
        
        # 1. 计算仿射变换 (只用前 9 个点)
        M, _ = cv2.estimateAffine2D(img_pts_np, rob_pts_np)
        if M is None:
            return False
        
        self.transform_pose = np.vstack([M, [0, 0, 1]])
        
        # 2. 计算旋转中心
        # 使用索引 9 到 end 的点 (即 9, 10, 11)
        rotate_p = []
        for i in range(9, 12):
            p = self.image_mark_points[i]
            rotate_p.append([p[0], p[1]])
            
        rotate_p = np.array(rotate_p, dtype=np.float64)
        
        fit_error, rot_center_img, fit_radius = self._compute_rotation_center(rotate_p)
        
        self.rotation_center = np.zeros(2)
        # real_center = rotate_center - registration_point
        self.rotation_center[0] = rot_center_img[0] - self.registration_point[0]
        self.rotation_center[1] = rot_center_img[1] - self.registration_point[1]
        
        self._raw_tool_center = rot_center_img # Save raw center for verify
        self.fitting_radius = fit_radius
        
        # 3. 计算误差
        self.errors = []
        
        # 前 9 个点的误差 (Affined + Rotated logic)
        for i in range(9):
            img_in = Pose3D(image_9points[i][0], image_9points[i][1], 0, 0, 0, angles[i])

            rob_in = Pose3D(self.robot_poses[i][0], self.robot_poses[i][1], 0, 0, 0, self.robot_poses[i][2])
            
            res_rob = self._transform_image_to_robot_pose(
                img_in, 
                Pose3D(self.registration_point[0], self.registration_point[1], 0, 0, 0, self.registration_point[2]),
                self.rotation_center,
                self.transform_pose
            )
            
            dist = math.sqrt((rob_in.x - res_rob.x)**2 + (rob_in.y - res_rob.y)**2)
            self.errors.append(dist)
            
        # 后 3 个点的误差 (Circle Fit Error)
        self.errors.extend(fit_error)
        
        return True

    def _compute_rotation_center(self, image_xy):
        # image_xy: (N, 2)
        n = len(image_xy)
        x = image_xy[:, 0]
        y = image_xy[:, 1]
        
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        Xi = x - mean_x
        Yi = y - mean_y
        Zi = Xi**2 + Yi**2
        
        Mxy = np.mean(Xi * Yi)
        Mxx = np.mean(Xi * Xi)
        Myy = np.mean(Yi * Yi)
        Mxz = np.mean(Xi * Zi)
        Myz = np.mean(Yi * Zi)
        Mzz = np.mean(Zi * Zi)
        
        Mz = Mxx + Myy
        Cov_xy = Mxx * Myy - Mxy * Mxy
        Var_z = Mzz - Mz * Mz
        
        A3 = 4.0 * Mz
        A2 = -3.0 * Mz * Mz - Mzz
        A1 = Var_z * Mz + 4.0 * Cov_xy * Mz - Mxz * Mxz - Myz * Myz
        A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var_z * Cov_xy
        
        A22 = A2 + A2
        A33 = A3 + A3 + A3
        
        # 牛顿迭代法
        iter_max = 99
        x_val = 0.0
        y_val = A0
        
        for i in range(iter_max):
            Dy = A1 + x_val * (A22 + A33 * x_val)
            if Dy == 0:
                 break
                 
            x_new = x_val - y_val / Dy
            
            if x_new == x_val or not np.isfinite(x_new):
                break
                
            y_new = A0 + x_new * (A1 + x_new * (A2 + x_new * A3))
            
            if abs(y_new) >= abs(y_val):
                break
                
            x_val = x_new
            y_val = y_new
            
        DET = x_val * x_val - x_val * Mz + Cov_xy
        
        if DET == 0:
            return [], np.array([0,0]), 0
            
        Xcenter = (Mxz * (Myy - x_val) - Myz * Mxy) / DET / 2.0
        Ycenter = (Myz * (Mxx - x_val) - Mxz * Mxy) / DET / 2.0
        
        final_x = Xcenter + mean_x
        final_y = Ycenter + mean_y
        
        radius = math.sqrt(Xcenter**2 + Ycenter**2 + Mz)
        
        fit_errors = []
        for i in range(n):
            dx = image_xy[i][0] - final_x
            dy = image_xy[i][1] - final_y
            dev = abs(math.sqrt(dx*dx + dy*dy) - radius)
            fit_errors.append(dev)
            
        return fit_errors, np.array([final_x, final_y]), radius

    def _rotate_point(self, pic_pix, angle, tool_center, register_point):
        # pic_pix: array [x, y]
        # tool_center: array [x, y] (Rotation Center relative to Reg)
        # register_point: Pose3D (x, y, rz)
        
        realtoolcx = register_point.x + tool_center[0]
        realtoolcy = register_point.y + tool_center[1]
        
        ang_rad = angle / 180.0 * math.pi
        c = math.cos(ang_rad)
        s = math.sin(ang_rad)
        
        dx = pic_pix[0] - realtoolcx
        dy = pic_pix[1] - realtoolcy
        
        out_x = realtoolcx + dx * c - dy * s
        out_y = realtoolcy + dy * c + dx * s
        
        return np.array([out_x, out_y])

    def _transform_image_to_robot_pose(self, image_xytheta, register_point, tool_center, trans_pose):
        # 1. Rz = Reg.angle - Image.angle
        Rz = register_point.rz - image_xytheta.rz
        
        # 2. Rotate Point
        regp = np.array([register_point.x, register_point.y])
        regp_rotated = self._rotate_point(regp, Rz, tool_center, register_point)
        
        # 3. Transform Reg P
        # Robot_regp = T * regp_rotated
        uv_reg = np.array([regp_rotated[0], regp_rotated[1], 1.0])
        rob_regp = trans_pose @ uv_reg
        
        # 4. Transform Image P
        # Robot_imgp = T * image_xytheta
        uv_img = np.array([image_xytheta.x, image_xytheta.y, 1.0])
        rob_imgp = trans_pose @ uv_img
        
        # 5. Diff
        res_x = rob_imgp[0] - rob_regp[0]
        res_y = rob_imgp[1] - rob_regp[1]
        res_rz = Rz
        
        return Pose3D(res_x, res_y, 0, 0, 0, res_rz)
        
    def transform_image_to_robot(self, img_pt, reg_pt, center, affine_mat):
        # Public wrapper for validation in calib.py
        # img_pt: [x,y,theta]
        # reg_pt: [x,y,rz]
        # center: [x,y] (relative)
        
        img_p = Pose3D(img_pt[0], img_pt[1], 0, 0, 0, img_pt[2] if len(img_pt)>2 else 0)
        reg_p = Pose3D(reg_pt[0], reg_pt[1], 0, 0, 0, reg_pt[2])
        
        res = self._transform_image_to_robot_pose(img_p, reg_p, center, affine_mat)
        return np.array([res.x, res.y, res.rz])
