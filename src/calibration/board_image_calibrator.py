import cv2
import numpy as np
import sys
from enum import Enum
from ..common.transforms import Pose3D
from ..common.rpy import RPYType, ReferenceType
from .robot_camera_calibrator_3d import RobotCameraCalibrator3D, CalibrationType3D

def format_mat_opencv(mat):
    # 模仿 OpenCV 格式: [e1, e2, e3; e4, e5, e6; ...]
    rows = mat.shape[0]
    cols = mat.shape[1] if len(mat.shape) > 1 else 1
    
    s = "["
    for i in range(rows):
        if len(mat.shape) == 1:
             s += f"{mat[i]}"
             if i < rows - 1: s += ", "
        else:
            for j in range(cols):
                s += f"{mat[i, j]}"
                if j < cols - 1: s += ", "
            if i < rows - 1: s += ";\n "
    s += "]"
    return s

class PatternType(Enum):
    CHESSBOARD = 0
    CIRCLES_GRID = 1
    CIRCLES_ASYM = 2

class CalibConfig:
    def __init__(self):
        self.pattern = PatternType.CIRCLES_ASYM
        self.cols = 4
        self.rows = 11
        self.interval_mm = 25.0
        self.measure_scale = 1.0 
        self.xml_intrinsic_1st = "intrinsic_1st.xml"
        self.xml_intrinsic_2nd = "intrinsic_2nd.xml"
        self.xml_extrinsic = "calib_extrinsic.xml"
        self.calib_type = CalibrationType3D.ETH

class BoardImageCalibrator:
    def __init__(self):
        self.cfg = CalibConfig()
        self.img_files = []
        self.robot_poses = []
        self.rpy_type = RPYType.XYZ
        self.ref_type = ReferenceType.EXTRINSIC
        self.calibration_type = CalibrationType3D.ETH
        
        self.result_pose = Pose3D()
        self.errors = []
        self.marker_points = []
        self.marker_success = []
        
        # Blob 检测器参数
        params = cv2.SimpleBlobDetector_Params()
        params.blobColor = 255
        params.filterByColor = True
        params.maxArea = 50000
        params.minArea = 1200
        params.filterByArea = True
        params.minDistBetweenBlobs = 36
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
        params.minConvexity = 0.8
        params.thresholdStep = 20
        self.blob_detector = cv2.SimpleBlobDetector_create(params)

    def set_config(self, cfg):
        self.cfg = cfg

    def set_image_files(self, files):
        self.img_files = files

    def set_robot_poses(self, poses):
        self.robot_poses = poses

    def set_calibration_type(self, c_type):
        self.calibration_type = c_type

    def set_rpy_type(self, rpy_type, ref_type=ReferenceType.EXTRINSIC):
        self.rpy_type = rpy_type
        self.ref_type = ref_type

    def run_calibration(self):
        fs = cv2.FileStorage(self.cfg.xml_intrinsic_1st, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
             print(f"Failed to open intrinsic file: {self.cfg.xml_intrinsic_1st}")
             return False
        
        K = fs.getNode("K").mat()
        dist = fs.getNode("distortion").mat()
        fs.release()
        
        if K is None:
            print("Intrinsic parameters not found!")
            return False

        all_images = []
        worlds_list = []
        pixels_list = []
        img_size = None
        
        obj_pts_full = self._generate_object_points()
        valid_indices = []
        
        for i, fpath in enumerate(self.img_files):
            raw = cv2.imread(fpath)
            if raw is None:
                continue
                
            if img_size is None:
                img_size = (raw.shape[1], raw.shape[0])
                
            undistorted = cv2.undistort(raw, K, dist)
            all_images.append(undistorted)
            
            found, corners = self._detect_corners(undistorted)
            if found:
                # 应用过滤 (calib_type=0) 逻辑用于标定阶段
                f_corners, f_objs = self._filter_points(corners, obj_pts_full, self.cfg.cols)
                
                worlds_list.append(f_objs)
                pixels_list.append(f_corners)
                valid_indices.append(i)
                
            sys.stdout.write(f"\r进度: {i+1}/{len(self.img_files)}  有效: {len(valid_indices)}")
            sys.stdout.flush()
            
        sys.stdout.write("\n")
        if len(valid_indices) < 5:
            print("至少 5 张！") 
            return False

        ret, K_2nd, dist_2nd, rvecs, tvecs = cv2.calibrateCamera(
            worlds_list, pixels_list, img_size, None, None, flags=0
        )
        
        print(f"第二次标定成功！RMS = {ret:.4f}")
        print(f"内参: \n{format_mat_opencv(K_2nd)}")
        print(f"畸变: \n{format_mat_opencv(dist_2nd.flatten())}")
        
        fs2 = cv2.FileStorage(self.cfg.xml_intrinsic_2nd, cv2.FILE_STORAGE_WRITE)
        fs2.write("K", K_2nd)
        fs2.write("distortion", dist_2nd)
        fs2.release()
        
        success_poses = []
        success_marks = []
        self.marker_points = [None] * len(self.robot_poses)
        self.marker_success = [False] * len(self.robot_poses)
        
        for i in range(len(self.robot_poses)):
            if i >= len(all_images):
                break
                
            img = all_images[i]
            found, corners = self._detect_corners(img)
            
            if found:
                # PnP 阶段使用全部点 (calib_type=1)
                success, rvec, tvec = cv2.solvePnP(obj_pts_full, corners, K_2nd, dist_2nd)
                if success:
                    pt = tvec.flatten()
                    success_marks.append(pt)
                    success_poses.append(self.robot_poses[i])
                    self.marker_points[i] = pt
                    self.marker_success[i] = True

        if not success_poses:
            return False

        calibrator = RobotCameraCalibrator3D()
        calibrator.set_rpy_type(self.rpy_type, self.ref_type)
        
        rmse = calibrator.calibrate(success_marks, success_poses, self.calibration_type)
        
        if np.isnan(rmse) or rmse < 0:
            return False
        
        raw_errors = calibrator.calibration_error
        self.errors = [-1.0] * len(self.robot_poses)
        
        err_idx = 0
        for i in range(len(self.robot_poses)):
            if self.marker_success[i] and err_idx < len(raw_errors):
                self.errors[i] = raw_errors[err_idx]
                err_idx += 1
        
        self.calibrator = calibrator
        self.result_pose = Pose3D.from_transform(calibrator.eye_to_fixed)
        return True
        
    def _generate_object_points(self):
        pts = []
        if self.cfg.pattern == PatternType.CIRCLES_ASYM:
            for i in range(self.cfg.rows):
                for j in range(self.cfg.cols):
                    if i % 2 == 0:
                        x = j * self.cfg.interval_mm
                    else:
                        x = j * self.cfg.interval_mm + self.cfg.interval_mm / 2.0
                    y = (i * self.cfg.interval_mm) / 2.0
                    pts.append([x, y, 0])
        else:
             for i in range(self.cfg.rows):
                for j in range(self.cfg.cols):
                    x = j * self.cfg.interval_mm
                    y = i * self.cfg.interval_mm
                    pts.append([x, y, 0])
        return np.array(pts, dtype=np.float32)

    def _filter_points(self, corners, obj_pts, cols):
        new_corners = []
        new_objs = []
        for i in range(len(corners)):
            row_idx = i // cols
            if row_idx != 1 and row_idx != 3:
                new_corners.append(corners[i])
                new_objs.append(obj_pts[i])
        return np.array(new_corners, dtype=np.float32), np.array(new_objs, dtype=np.float32)

    def _detect_corners(self, img):
        if self.cfg.pattern == PatternType.CIRCLES_ASYM:
            flags = cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
            size = (self.cfg.cols, self.cfg.rows)
            found, corners = cv2.findCirclesGrid(img, size, flags=flags, blobDetector=self.blob_detector)
            return found, corners
        elif self.cfg.pattern == PatternType.CHESSBOARD:
            size = (self.cfg.cols, self.cfg.rows)
            found, corners = cv2.findChessboardCorners(img, size)
            if found:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), 
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            return found, corners
        else:
            flags = cv2.CALIB_CB_SYMMETRIC_GRID
            size = (self.cfg.cols, self.cfg.rows)
            found, corners = cv2.findCirclesGrid(img, size, flags=flags)
            return found, corners

    def get_result_pose(self):
        return self.result_pose
    def get_errors(self):
        return self.errors
    def get_marker_points(self):
        return self.marker_points
    def get_marker_success(self):
        return self.marker_success
