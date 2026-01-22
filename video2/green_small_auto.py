import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi
from skimage.morphology import skeletonize

class FullyAdaptiveVesselAnalyzer:

    def estimate_vessel_scale(self, gray):
        """
        Frangi 필터를 이용해 혈관 반경 분포를 추정
        """
        sigmas = np.arange(1, 16, 2)
        vesselness = frangi(gray, sigmas=sigmas, black_ridges=True)

        # 가장 강한 혈관 반응 픽셀들만 사용
        mask = vesselness > np.percentile(vesselness, 85)
        if np.count_nonzero(mask) < 100:
            return 5  # fallback

        # 반응이 가장 강한 sigma → 혈관 반경
        radius_map = np.argmax(
            np.stack([frangi(gray, sigmas=[s], black_ridges=True) for s in sigmas]),
            axis=0
        )

        median_radius = np.median(radius_map[mask])
        return max(2, int(median_radius))

    def get_auto_roi(self, raw, vessel_radius):
        lab = cv2.cvtColor(raw, cv2.COLOR_BGR2Lab)
        a = lab[:,:,1]

        blur_size = int(6 * vessel_radius) | 1
        blur = cv2.GaussianBlur(a, (blur_size, blur_size), 0)

        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        k = int(4 * vessel_radius)
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        roi = np.zeros(raw.shape[:2], np.uint8)
        cv2.drawContours(roi, [hull], -1, 255, -1)

        return roi

    def get_adaptive_vessels(self, raw, roi, vessel_radius):
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=roi)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(vessel_radius*2, vessel_radius*2))
        enhanced = clahe.apply(masked)

        # 혈관 대비 자동 추정
        fr = frangi(enhanced, sigmas=range(1, vessel_radius*3, 2), black_ridges=True)
        vessel_pixels = enhanced[fr > np.percentile(fr, 85)]

        μ, σ = vessel_pixels.mean(), vessel_pixels.std()
        thresh = μ - 0.4 * σ

        binary = (enhanced < thresh).astype(np.uint8) * 255

        k = int(1.5 * vessel_radius)
        kernel = np.ones((k, k), np.uint8)
        it = max(1, vessel_radius // 2)

        filled = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=it)
        return cv2.bitwise_and(filled, filled, mask=roi)

    def analyze(self, img_path):
        raw = cv2.imread(img_path)
        if raw is None:
            print("이미지 로드 실패")
            return

        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        vessel_radius = self.estimate_vessel_scale(gray)
        print(f"자동 추정 혈관 반경: {vessel_radius}px")

        roi = self.get_auto_roi(raw, vessel_radius)
        vessel = self.get_adaptive_vessels(raw, roi, vessel_radius)

        skeleton = skeletonize(vessel > 0).astype(np.uint8) * 255

        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.imshow(vessel, cmap='gray')
        plt.title("Fully Adaptive Vessel Map")
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.imshow(skeleton, cmap='gray')
        plt.title("Topology-Preserving Skeleton")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    analyzer = FullyAdaptiveVesselAnalyzer()
    analyzer.analyze("ex1.jpg")
