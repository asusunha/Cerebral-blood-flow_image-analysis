import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

class FinalVesselSystem:
    def __init__(self):
        pass

    def get_color_roi(self, raw_img):
        """[로직 1] 색상 기반 ROI 추출 (유효 영역 파악)"""
        lab = cv2.cvtColor(raw_img, cv2.COLOR_BGR2Lab)
        a_channel = lab[:, :, 1]
        
        blur = cv2.GaussianBlur(a_channel, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 135, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((25, 25), np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, None
        
        main_cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(main_cnt)
        
        mask = np.zeros(raw_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        return mask, hull

    def get_clean_vessels(self, raw_img, roi_mask):
        """[로직 2] 혈관 정밀 추출 (해상도 유지 및 대비 강화)"""
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=roi_mask)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(masked)
        
        vessel = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 21, 7)
        return cv2.bitwise_and(vessel, vessel, mask=roi_mask)

    def analyze(self, img_path):
        # TIFF 파일의 고해상도 데이터를 그대로 읽기 위해 IMREAD_UNCHANGED 사용
        raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if raw is None: 
            print("파일을 불러올 수 없습니다.")
            return
        
        # 1. ROI 및 혈관 추출 로직 유지
        roi_mask, hull = self.get_color_roi(raw)
        if roi_mask is None: return
        
        vessel = self.get_clean_vessels(raw, roi_mask)
        skeleton = skeletonize(vessel > 0).astype(np.uint8) * 255
        
        # 2. 지표 계산
        dist_map = distance_transform_edt(vessel > 0)
        avg_dia = np.mean(dist_map[skeleton > 0]) * 2 if np.any(skeleton) else 0
        density = np.sum(vessel > 0) / np.sum(roi_mask > 0)
        
        # 3. 시각화 데이터 구성 (요청하신 4종류)
        final_viz = raw.copy()
        cv2.drawContours(final_viz, [hull], -1, (255, 0, 0), 3) # ROI 경계(파란색)
        final_viz[vessel > 0] = [0, 255, 0] # 혈관 영역(초록색)
        
        # [수정] 4가지 이미지만 리스트업
        imgs = [raw, vessel, skeleton, final_viz]
        titles = ['Original', 'Vessel Map', 'Skeleton', 'Final Overlay']
        
        # 2x2 레이아웃으로 출력
        plt.figure(figsize=(14, 12))
        for i, (img, title) in enumerate(zip(imgs, titles)):
            plt.subplot(2, 2, i+1)
            # 3채널(Color)과 1채널(Gray/Binary) 구분 처리
            if len(img.shape) == 3:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(img, cmap='gray')
            plt.title(title, fontsize=14)
            plt.axis('off')
        
        report = f"Density: {density:.4f} | Avg Diameter: {avg_dia:.2f} px"
        plt.figtext(0.5, 0.02, report, ha="center", fontsize=15, 
                    bbox={"facecolor":"white", "edgecolor":"blue", "alpha":0.8, "pad":10})
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    analyzer = FinalVesselSystem()
    # 확장자가 .tif 또는 .tiff여도 동일하게 동작합니다.
    analyzer.analyze('ex1.tiff')