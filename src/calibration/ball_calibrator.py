import numpy as np
from ..common.transforms import Pose3D
from ..common.rpy import RPYType, ReferenceType
from .robot_camera_calibrator_3d import RobotCameraCalibrator3D, CalibrationType3D

class BallCalibrator:
    def __init__(self):
        self.robot_poses = []
        self.mark_points = [] # 标记点 (球心位置)
        self.calib_type = CalibrationType3D.ETH
        self.rpy_type = RPYType.XYZ
        self.ref_type = ReferenceType.EXTRINSIC
        self.calibrator = RobotCameraCalibrator3D()
        self.result_pose = Pose3D()
        self.errors = []

    def set_robot_poses(self, poses):
        self.robot_poses = poses

    def set_mark_points(self, points):
        self.mark_points = points

    def set_calibration_type(self, type_3d):
        self.calib_type = type_3d

    def set_rpy_type(self, rpy_type, ref_type=ReferenceType.EXTRINSIC):
        self.rpy_type = rpy_type
        self.ref_type = ref_type

    def run_calibration(self):
        if not self.robot_poses or not self.mark_points:
            return False

        self.calibrator.set_rpy_type(self.rpy_type, self.ref_type)
        
        marks_np = [np.array(p, dtype=np.float64) for p in self.mark_points]
        
        rmse = self.calibrator.calibrate(marks_np, self.robot_poses, self.calib_type)
        
        if np.isnan(rmse) or rmse < 0:
            return False
            
        self.errors = self.calibrator.calibration_error
        self.result_pose = Pose3D.from_transform(self.calibrator.eye_to_fixed)
        return True

    def get_result_pose(self):
        return self.result_pose
        
    def get_errors(self):
        return self.errors
