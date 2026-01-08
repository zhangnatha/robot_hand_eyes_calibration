import sys
import os
import cv2
import numpy as np
from src.common.transforms import Pose3D
from src.common.rpy import RPYType, ReferenceType
from src.calibration.robot_camera_calibrator_3d import CalibrationType3D
from src.calibration.board_image_calibrator import BoardImageCalibrator, CalibConfig, PatternType
from src.calibration.twelve_point_calibrator import TwelvePointCalibrator, CameraInstallType, CalibrationType2D
from src.calibration.ball_calibrator import BallCalibrator
from src.calibration.intrinsic_calibrator import IntrinsicCalibrator

# 2D/3D相机内参、外参标定类型选择
RUN_INTRINSIC = True
RUN_HAND_EYE_3D_BOARD_IMG = True
RUN_FOUR_POINT_2D = True
RUN_TWELVE_POINT_2D = True
RUN_HAND_EYE_3D_BALL = True

def log_section(msg):
    print(f"\n\033[1;35m====================================================================\n   {msg}\n====================================================================\033[0m\n")

def info_log(msg):
    print(f"\033[1;36m[INFO] {msg}\033[0m")

def success_log(msg):
    print(f"\033[1;32m[SUCCESS] {msg}\033[0m")

def warn_log(msg):
    print(f"\033[1;33m[WARN] {msg}\033[0m")
    
def error_log(msg):
    print(f"\033[1;31m[ERROR] {msg}\033[0m")

def main():
    log_section("Industrial Camera + Robot Hand-Eye Calibration Program (Python Port)")
    print("Included: 2.5D Hand-Eye | 2D Four-Point | 2D Twelve-Point High Precision Calibration\n")
    
    # 标定相关配置
    cfg = CalibConfig()
    cfg.pattern = PatternType.CIRCLES_ASYM
    cfg.calib_type = CalibrationType3D.ETH
    cfg.cols = 4
    cfg.rows = 5
    cfg.interval_mm = 20.0
    cfg.xml_intrinsic_1st = "intrinsic_1st.xml"
    cfg.xml_intrinsic_2nd = "intrinsic_2nd.xml"
    cfg.xml_extrinsic = "calib_extrinsic.xml"

    # 机器人位姿数据
    poses_data = [
        [211.892, -415.103, -536.213, -91.135, 44.296, 11.726],
        [217.353, -412.223, -530.437, -89.057, 44.896, 13.651],
        [235.681, -408.397, -527.87, -87.607, 45.096, 15.771],
        [235.678, -411.597, -527.868, -84.762, 46.641, 21.056],
        [252.301, -427.281, -524.428, -82.312, 47.657, 23.928],
        [251.664, -412.156, -518.114, -80.204, 48.466, 27.648],
        [250.344, -422.522, -520.923, -78.108, 49.324, 32.218],
        [252.918, -401.931, -541.12, -83.448, 46.17, 23.515],
        [251.206, -386.262, -542.755, -85.8, 45.12, 20.078],
        [223.281, -378.458, -542.76, -88.365, 44.261, 16.83],
        [219.702, -418.79, -536.205, -89.737, 50.569, 12.734],
        [217.392, -412.203, -530.434, -91.857, 39.008, 18.558],
        [235.686, -408.395, -527.868, -79.344, 39.103, 18.983],
        [235.683, -411.597, -535.154, -79.481, 39.239, 23.953],
        [245.722, -414.54, -523.743, -84.142, 50.207, 28.574],
        [256.122, -412.152, -526.174, -79.561, 44.677, 33.859],
        [238.346, -422.975, -530.894, -84.73, 48.306, 25.753],
        [252.926, -401.932, -536.669, -83.512, 38.762, 28.218],
        [251.213, -386.262, -542.757, -82.352, 48.152, 39.276],
        [229.352, -374.528, -536.877, -86.592, 38.116, 17.081]
    ]
    robot_poses = [Pose3D(*p) for p in poses_data]

    # 离线获取相机拍摄的标定板图片
    base_path = "assert/camera/"
    img_files = []
    for i in range(10, 30):
        p = os.path.join(base_path, f"{i}.png")
        if os.path.exists(p):
            img_files.append(p)
    
    if len(img_files) == 0:
        error_log("No images found! Check 'assert/camera' directory.")
    
    # -------------------------------------------------------------------------
    # 类别 1: 相机 内参标定
    # -------------------------------------------------------------------------
    if RUN_INTRINSIC:
        log_section("Category 1: Intrinsic Calibration")
        intr_calib = IntrinsicCalibrator(cfg)
        if intr_calib.intrinsic_calibrate(img_files):
            success_log("Intrinsic calibration success!")
        else:
            error_log("Intrinsic calibration failed!")

    # -------------------------------------------------------------------------
    # 类别 2: 2D相机 基于标定板图像的 3D 手眼标定
    # -------------------------------------------------------------------------
    if RUN_HAND_EYE_3D_BOARD_IMG:
        log_section("Category 2: 3D Camera Hand-Eye Calibration - Based on Board Image")
        
        calib = BoardImageCalibrator()
        calib.set_config(cfg)
        calib.set_image_files(img_files)
        calib.set_robot_poses(robot_poses)
        calib.set_calibration_type(cfg.calib_type)
        calib.set_rpy_type(RPYType.XYZ, ReferenceType.EXTRINSIC)
        
        info_log("Starting hand-eye calibration...")
        if calib.run_calibration():
            success_log("3D Hand-Eye Calibration success!")
            
            # 打印误差表
            print("\n[INFO] Index | Robot(x,y,z,rx,ry,rz) | Marker(x,y,z) | Error")
            print("--------------------------------------------------------------")
            
            poses = calib.robot_poses 
            markers = calib.get_marker_points()
            success = calib.get_marker_success()
            errors = calib.get_errors()
            
            valid_errs = [e for e in errors if e >= 0]
            if not valid_errs:
                avg_err, max_err, min_err = 0, 0, 0
            else:
                avg_err = sum(valid_errs)/len(valid_errs)
                max_err = max(valid_errs)
                min_err = min(valid_errs)
            
            for i in range(len(poses)):
                p = poses[i]
                if i < len(success) and success[i]:
                     m = markers[i]
                     e = errors[i]
                     print(f"{i:3d} | {p.x:10.4f}, {p.y:10.4f}, {p.z:10.4f}, {p.rx:10.4f}, {p.ry:10.4f}, {p.rz:10.4f} | {m[0]:10.4f}, {m[1]:10.4f}, {m[2]:10.4f} | {e:10.4f}")
                else:
                     print(f"{i:3d} | {p.x:10.4f}, {p.y:10.4f}, {p.z:10.4f}, {p.rx:10.4f}, {p.ry:10.4f}, {p.rz:10.4f} |        N/A,        N/A,        N/A |        N/A")

            success_log(f"3D Hand-Eye Calibration successful! Avg Error: {avg_err:.6f}, Max: {max_err:.6f}, Min: {min_err:.6f}")
            
            pose, tcp = calib.calibrator.get_result()
            print("[INFO] hand_eyes Homogeneous Matrix (Eigen format):")
            for row in pose:
                print(f"  {row[0]:.4f}  {row[1]:.4f}  {row[2]:.4f}  {row[3]:.4f}")
            print("")
            
            res_p = Pose3D.from_transform(pose)
            print("[RESULT] hand_eyes (Pose3D format):")
            print(f"{res_p.x:.4f}, {res_p.y:.4f}, {res_p.z:.4f}, {res_p.rx:.4f}, {res_p.ry:.4f}, {res_p.rz:.4f}")
            info_log(f"Hand-eye calibration results (R | t) saved to: {cfg.xml_extrinsic}")
            success_log("3D Hand-Eye Calibration complete!")
        else:
            error_log("3D Hand-Eye Calibration failed!")


    # -------------------------------------------------------------------------
    # 类别 3: 2D 相机 4点插针标定
    # -------------------------------------------------------------------------
    if RUN_FOUR_POINT_2D:
        log_section("Category 3: 2D Camera Hand-Eye Calibration - 4-Point Pin")
        
        img_points = np.array([[2035, 1093], [3129, 1087], [3130, 1549], [2038, 1556]], dtype=np.float32)
        robot_points = np.array([[800.94, 886.14], [805.45, 570.15], [671.78, 568.56], [667.07, 884.8]], dtype=np.float32)
        
        M, _ = cv2.estimateAffine2D(img_points, robot_points)
        
        print("[INFO] hand_eyes Homogeneous Matrix (2x3):")
        if M is not None:
             print(f"[{M[0,0]}, {M[0,1]}, {M[0,2]};\n {M[1,0]}, {M[1,1]}, {M[1,2]}]")
             
             test_pt = np.array([2035, 1093, 1], dtype=np.float32)
             res = M @ test_pt
             print(f"\n\033[1;32m[VERIFY] Four-point pin verification: Image({test_pt[0]:.4f},{test_pt[1]:.4f}) -> Robot({res[0]:.4f}, {res[1]:.4f})\033[0m")
             print("")

    # -------------------------------------------------------------------------
    # 类别 4: 2D 相机 12点标定
    # -------------------------------------------------------------------------
    if RUN_TWELVE_POINT_2D:
        log_section("Category 4: 2D Camera Hand-Eye Calibration - 12-Point")
        
        calib12 = TwelvePointCalibrator()
        calib12.set_hand_eye_type(CalibrationType2D.EIH)
        calib12.set_camera_install_type(CameraInstallType.SameToTCPZ)
        calib12.set_registration_point([1188, 468, 0])
        
        img_points = [
            [1188, 468, -0], [1173, 467, -0], [1195, 464, -0], [1320, 474, -0],
            [1249, 593, 5],  [1292, 545, 7],  [1346, 632, 6],  [1439, 520, 5],
            [1482, 595, 4],  [1329, 437, 6],  [1239, 472, -0], [1420, 550, -14]
        ]
        
        robot_poses_12 = [
            [0, 0, 0],    [0, 30, 0],  [30, 30, 0],   [30, 0, 0],
            [30, -30, 0], [0, -30, 0], [-30, -30, 0], [-30, 0, 0],
            [-30, 30, 0], [0, 0, -10], [0, 0, 0],     [0, 0, 10]
        ]
        
        calib12.set_image_mark_points(img_points)
        calib12.set_robot_poses(robot_poses_12)
        
        if calib12.run_calibration():
            print("[INFO] Index | Robot(dx,dy,rz) | Image(x,y,theta) | Error")
            print("--------------------------------------------------------------")
            for i, idx in enumerate(range(len(img_points))):
                rp = robot_poses_12[i]
                ip = img_points[i]
                err = calib12.errors[i] if i < len(calib12.errors) else 0.0
                print(f"{i:3d} | {rp[0]:10.4f}, {rp[1]:10.4f}, {rp[2]:10.4f} | {ip[0]:10.4f}, {ip[1]:10.4f}, {ip[2]:10.4f} | {err:10.4f}")

            success_log("2D Twelve-point Calibration successful!")
            
            M = calib12.transform_pose
            print("[INFO] Transform Pose (2x3 Affine Matrix):")
            print(f"{M[0,0]:.4f}, {M[0,1]:.4f}, {M[0,2]:.4f}")
            print(f"{M[1,0]:.4f}, {M[1,1]:.4f}, {M[1,2]:.4f}")
            
            c = calib12.rotation_center
            print(f"\n[INFO] Circle Fitting Result: \ncenter=({c[0] + calib12.registration_point[0]:.4f}, {c[1] + calib12.registration_point[1]:.4f}), R=({calib12.fitting_radius:.4f})")
            print(f"\n[RESULT] TCP Rotation Center (relative to registration point): {c[0]:.4f}, {c[1]:.4f}")
            
            test_img = np.array([1188, 468, 0])
            res = calib12.transform_image_to_robot(test_img, calib12.registration_point, 
                                                   calib12.rotation_center, calib12.transform_pose)
            print(f"[VERIFY] Twelve-point verification: Image({test_img[0]:.4f},{test_img[1]:.4f}) -> Robot({res[0]:.4f}, {res[1]:.4f}, {res[2]:.4f}°)")


    # -------------------------------------------------------------------------
    # 类别 5: 3D相机 标定球标定
    # -------------------------------------------------------------------------
    if RUN_HAND_EYE_3D_BALL:
        log_section("Category 5: 3D Camera Hand-Eye Calibration - Based on Ball Center XYZ")
        
        robot_poses_data = [
            [-1223.07, 1493.14, 774.351, -172.691, -14.074, 170.725],
            [-1105.25, 1370.07, 774.303, -171.213, -18.84, 153.076],
            [-1064.56, 1424.31, 808.086, 166.736, -18.035, 160.874],
            [-1154.64, 1396.67, 780.382, 168.416, -13.818, 164.482],
            [-1099.28, 1471.44, 637.932, -170.201, -0.875, 160.177],
            [-1215.18, 1368.72, 656.108, -173.238, 14.414, 173.631],
            [-1148.45, 1422.8, 730.753, -176.372, 10.735, 163.365],
            [-1132.76, 1358.11, 775.324, 178.924, 4.649, 171.376],
            [-1231.24, 1314.03, 762.005, 169.6, -10.535, 173.234],
            [-1186.72, 1408.22, 682.34, 174.96, -4.641, 165.91],
            [-1106.26, 1441.8, 686.117, 179.956, 0.57, 152.826]
        ]
        
        marker_points_data = [
            [-107, -2.25, 632.75],   [-3.75, -39.25, 528],
            [48.75, -37, 627.5],     [-29.75, -14.25, 580.25],
            [46.75, 108.75, 584.5],  [-19.5, 59.75, 490.75],
            [21, 2.5, 563],          [38.75, -47.25, 525.75],
            [-89.25, -16.75, 487.5], [-36, 78.25, 554.25],
            [39.25, 65.5, 575.5]
        ]
        
        ball_calib = BallCalibrator()
        rob_poses = [Pose3D(*p) for p in robot_poses_data]
        
        ball_calib.set_robot_poses(rob_poses)
        ball_calib.set_mark_points(marker_points_data)
        ball_calib.set_calibration_type(CalibrationType3D.ETH)
        ball_calib.set_rpy_type(RPYType.XYZ, ReferenceType.EXTRINSIC)
        
        if ball_calib.run_calibration():
            # 打印误差表
            print("\n[INFO] Index | Robot(x,y,z,rx,ry,rz) | Marker(x,y,z) | Error")
            print("--------------------------------------------------------------")
            
            errors = ball_calib.get_errors()
            for i in range(len(rob_poses)):
                 p = rob_poses[i]
                 m = marker_points_data[i]
                 e = errors[i]
                 print(f"{i:3d} | {p.x:10.4f}, {p.y:10.4f}, {p.z:10.4f}, {p.rx:10.4f}, {p.ry:10.4f}, {p.rz:10.4f} | {m[0]:10.4f}, {m[1]:10.4f}, {m[2]:10.4f} | {e:10.4f}")
                 
            success_log(f"3D Ball-based Calibration successful! Avg Error: {sum(errors)/len(errors):.6f}")
            pose = ball_calib.get_result_pose()
            
            pose_mat = pose.to_transform_matrix()
            print("[INFO] hand_eyes Homogeneous Matrix (Eigen format):")
            for row in pose_mat:
                print(f"  {row[0]:.4f}  {row[1]:.4f}  {row[2]:.4f}  {row[3]:.4f}")
            print("")
            
            print("[RESULT] hand_eyes (Pose3D format):")
            print(f"{pose.x:.4f}, {pose.y:.4f}, {pose.z:.4f}, {pose.rx:.4f}, {pose.ry:.4f}, {pose.rz:.4f}")

    log_section("All Calibrations Complete!")

if __name__ == "__main__":
    main()
