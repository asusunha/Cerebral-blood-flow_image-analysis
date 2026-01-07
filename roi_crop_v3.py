'''
# 입력
Baseline Image: Video_00056_00001_0s
Target Image: Video_00056_00030_0s

# 출력
Aligned_ROI_v3 폴더
- crop된 Baseline Image
- crop된 Aligned Image
- 블랙 픽셀이 제거된 Baseline Image
- 블랙 픽셀이 제거된 Aligned Image

# 결과
정확도 좋으나, 블랙 픽셀이 완전히 제거되지는 않는 모습
'''

import cv2
import numpy as np
import os
from skimage.filters import sato
from skimage import exposure

class UltraPreciseVesselAligner:
    def __init__(self, baseline_roi_size=(1648, 1424)):
        self.baseline_roi_size = baseline_roi_size
        self.baseline_roi_gray = None
        self.baseline_roi_color = None
        self.baseline_vesselness = None
        self.roi_offset = None
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)

    def preprocess_vesselness(self, img_gray):
        v_map = sato(img_gray, sigmas=range(2, 10, 2), black_ridges=False)
        v_map = exposure.rescale_intensity(v_map, out_range=(0, 255)).astype(np.uint8)
        v_map = cv2.GaussianBlur(v_map, (3, 3), 0)
        return v_map

    def set_baseline(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")

        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        roi_w, roi_h = self.baseline_roi_size

        x1 = max(0, center_x - (roi_w // 2))
        y1 = max(0, center_y - (roi_h // 2))

        self.baseline_roi_color = img[y1:y1+roi_h, x1:x1+roi_w]
        self.baseline_roi_gray = cv2.cvtColor(self.baseline_roi_color, cv2.COLOR_BGR2GRAY)
        self.roi_offset = (x1, y1)

        self.baseline_vesselness = self.preprocess_vesselness(self.baseline_roi_gray)
        self.baseline_name = os.path.splitext(os.path.basename(img_path))[0]

        print(f"Baseline 설정 완료: {os.path.basename(img_path)}")
        return self.baseline_roi_color

    def find_valid_crop_area(self, aligned_img):
        gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)

    def align_and_crop(self, target_img_path, save_dir):
        target_img = cv2.imread(target_img_path)
        if target_img is None:
            return None

        target_name = os.path.basename(target_img_path)
        target_name_wo_ext = os.path.splitext(target_name)[0]

        output_filename = f"Aligned_{target_name_wo_ext}.tiff"
        output_crop_filename = f"Aligned_{target_name_wo_ext}_crop.tiff"

        baseline_filename = f"Baseline_{self.baseline_name}.tiff"
        baseline_crop_filename = f"Baseline_{self.baseline_name}_crop.tiff"

        baseline_save_path = os.path.join(save_dir, baseline_filename)
        baseline_crop_save_path = os.path.join(save_dir, baseline_crop_filename)

        if not os.path.exists(baseline_save_path):
            cv2.imwrite(baseline_save_path, self.baseline_roi_color)

        save_path = os.path.join(save_dir, output_filename)
        save_crop_path = os.path.join(save_dir, output_crop_filename)

        x1, y1 = self.roi_offset
        roi_w, roi_h = self.baseline_roi_size

        target_roi_color = target_img[y1:y1+roi_h, x1:x1+roi_w]
        target_roi_gray = cv2.cvtColor(target_roi_color, cv2.COLOR_BGR2GRAY)

        target_vesselness = self.preprocess_vesselness(target_roi_gray)

        warp_matrix = np.eye(2, 3, dtype=np.float32)

        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(self.baseline_vesselness, None)
        kp2, des2 = orb.detectAndCompute(target_vesselness, None)

        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            if len(matches) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                M_init, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                if M_init is not None:
                    warp_matrix = M_init.astype(np.float32)

        try:
            cc, warp_matrix = cv2.findTransformECC(
                self.baseline_vesselness,
                target_vesselness,
                warp_matrix,
                cv2.MOTION_EUCLIDEAN,
                self.criteria,
                None,
                5
            )

            aligned_roi = cv2.warpAffine(
                target_roi_color,
                warp_matrix,
                (roi_w, roi_h),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP
            )

            cv2.imwrite(save_path, aligned_roi)

            crop_area = self.find_valid_crop_area(aligned_roi)
            if crop_area is not None:
                x, y, w, h = crop_area
                aligned_cropped = aligned_roi[y:y+h, x:x+w]
                cv2.imwrite(save_crop_path, aligned_cropped)

                baseline_cropped = self.baseline_roi_color[y:y+h, x:x+w]
                cv2.imwrite(baseline_crop_save_path, baseline_cropped)

            return aligned_roi

        except cv2.error:
            aligned_roi = cv2.warpAffine(
                target_roi_color,
                warp_matrix,
                (roi_w, roi_h),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP
            )
            cv2.imwrite(save_path, aligned_roi)

            crop_area = self.find_valid_crop_area(aligned_roi)
            if crop_area is not None:
                x, y, w, h = crop_area
                aligned_cropped = aligned_roi[y:y+h, x:x+w]
                cv2.imwrite(save_crop_path, aligned_cropped)

                baseline_cropped = self.baseline_roi_color[y:y+h, x:x+w]
                cv2.imwrite(baseline_crop_save_path, baseline_cropped)

            return aligned_roi


if __name__ == "__main__":
    BASELINE_DIR = r'D:\VIDEO\M-24_captured\Video_00056_00001'
    TARGET_DIR = r'D:\VIDEO\M-24_captured\Video_00056_00030'

    BASELINE_NAME = 'Video_00056_00001_0s.tiff'
    TARGET_NAME = 'Video_00056_00030_0s.tiff'

    SAVE_DIR = os.path.join(TARGET_DIR, "Aligned_ROI_v3")
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    aligner = UltraPreciseVesselAligner(baseline_roi_size=(1848, 1624))

    baseline_full_path = os.path.join(BASELINE_DIR, BASELINE_NAME)
    aligner.set_baseline(baseline_full_path)

    target_full_path = os.path.join(TARGET_DIR, TARGET_NAME)
    result = aligner.align_and_crop(target_full_path, SAVE_DIR)

    if result is not None:
        h, w = result.shape[:2]
        comparison = np.zeros((h, w, 3), dtype=np.uint8)
        comparison[:, :, 0] = aligner.baseline_roi_color[:, :, 0]
        comparison[:, :, 2] = result[:, :, 2]

