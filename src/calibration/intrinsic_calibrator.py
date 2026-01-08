import cv2
import numpy as np
import os
import sys
from .board_image_calibrator import BoardImageCalibrator, PatternType, format_mat_opencv

class IntrinsicCalibrator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.helper = BoardImageCalibrator()
        self.helper.set_config(cfg)

    def intrinsic_calibrate(self, img_files):
        worlds_list = []
        pixels_list = []
        img_size = None
        valid_count = 0
        
        obj_pts_full = self.helper._generate_object_points()

        for i, fpath in enumerate(img_files):
            if not os.path.exists(fpath):
                 continue

            raw = cv2.imread(fpath)
            if raw is None:
                continue
                
            if img_size is None:
                img_size = (raw.shape[1], raw.shape[0])
                
            found, corners = self.helper._detect_corners(raw)
            if found:
                # 应用过滤 (calib_type=0)
                f_corners, f_objs = self.helper._filter_points(corners, obj_pts_full, self.cfg.cols)
                
                worlds_list.append(f_objs)
                pixels_list.append(f_corners)
                valid_count += 1
            
            sys.stdout.write(f"\r进度: {i+1}/{len(img_files)}  有效: {valid_count}")
            sys.stdout.flush()
            
        sys.stdout.write("\n")
        
        if valid_count < 5:
            print("至少 5 张！") 
            return False
            
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            worlds_list, pixels_list, img_size, None, None, flags=0
        )
        
        print(f"第一次标定成功！RMS = {ret:.4f}") 
        print(f"内参: \n{format_mat_opencv(K)}")
        print(f"畸变: \n{format_mat_opencv(dist.flatten())}")
        
        fs = cv2.FileStorage(self.cfg.xml_intrinsic_1st, cv2.FILE_STORAGE_WRITE)
        fs.write("K", K)
        fs.write("distortion", dist)
        fs.write("pattern", int(self.cfg.pattern.value))
        fs.release()
        
        return True
