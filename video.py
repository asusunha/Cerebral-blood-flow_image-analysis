import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # 데이터 저장을 위해 추가
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

class RobustVesselAnalyzer:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("오류: 영상을 불러올 수 없습니다. 경로를 확인하세요.")
            return

        # 분석 지표 저장소
        self.history = {"frame": [], "density": [], "velocity": [], "diameter": []}
        
        # --- [최적화 설정] ---
        self.STR_PERIOD = 5    # 5프레임마다 정밀 분석
        self.SCALE = 0.6       # 분석 해상도 (60% 축소로 렉 방지)
        self.DISPLAY_W = 1280  # 화면 출력 크기
        self.DISPLAY_H = 640
        # --------------------

        cv2.namedWindow('Microvascular Analysis System', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Microvascular Analysis System', self.DISPLAY_W, self.DISPLAY_H)

    def get_refined_roi(self, frame):
        """[단계 1&2] a-채널 기반 ROI 및 매끄러운 Convex Hull 추출"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        a_channel = lab[:, :, 1] # 붉은색 채널 활용
        blur = cv2.GaussianBlur(a_channel, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 135, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((25, 25), np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # 파편화 방지
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return None, None
        main_cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(main_cnt) # 매끄러운 단일 ROI 형성
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        return mask, hull

    def get_vessel_mask(self, frame, roi_mask):
        """[단계 3&4] CLAHE 3.0 + Adaptive 21/7 + 모폴로지 연결"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=roi_mask)
        
        # 1. 대비 강화 (이미지 분석 베스트 설정)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(masked)
        
        # 2. 적응형 이진화
        vessel = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 21, 7)
        
        # 3. 모폴로지 연결: 끊어진 혈관 마디를 붙여줌
        # [Image of image morphology]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vessel = cv2.morphologyEx(vessel, cv2.MORPH_CLOSE, kernel)
        
        # 4. 노이즈 제거: 20px 이하 자잘한 점 삭제
        nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(vessel)
        clean_vessel = np.zeros_like(vessel)
        for i in range(1, nlabels):
            if stats[i, cv2.CC_STAT_AREA] >= 20:
                clean_vessel[labels == i] = 255
        return clean_vessel

    def process(self):
        frame_idx = 0
        ret, prev_frame = self.cap.read()
        if not ret: return
        
        prev_proc = cv2.resize(prev_frame, (0,0), fx=self.SCALE, fy=self.SCALE)
        prev_gray = cv2.cvtColor(prev_proc, cv2.COLOR_BGR2GRAY)
        last_overlay = None

        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame_idx += 1
            
            proc_img = cv2.resize(frame, (0,0), fx=self.SCALE, fy=self.SCALE)
            curr_gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            roi_mask, hull = self.get_refined_roi(proc_img)
            
            if roi_mask is not None:
                # [주기적 정밀 분석]
                if frame_idx % self.STR_PERIOD == 0:
                    vessel_map = self.get_vessel_mask(proc_img, roi_mask)
                    
                    # 지표 계산: 밀도 및 직경
                    density = np.sum(vessel_map > 0) / np.sum(roi_mask > 0)
                    skeleton = skeletonize(vessel_map > 0).astype(np.uint8)
                    dist_map = distance_transform_edt(vessel_map > 0)
                    avg_dia = np.mean(dist_map[skeleton > 0]) * 2 if np.any(skeleton) else 0
                    
                    # 혈류 속도 계산 (Optical Flow)
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    velocity = np.mean(mag[roi_mask > 0])

                    # 데이터 기록
                    self.history["frame"].append(frame_idx)
                    self.history["density"].append(density)
                    self.history["velocity"].append(velocity)
                    self.history["diameter"].append(avg_dia)
                    
                    # 시각화용 오버레이
                    last_overlay = cv2.resize(vessel_map, (frame.shape[1], frame.shape[0]))

            # --- [시각화 및 출력] ---
            viz = frame.copy()
            if last_overlay is not None:
                viz[last_overlay > 0] = [0, 255, 0] # 초록색 혈관
            if hull is not None:
                hull_orig = (hull / self.SCALE).astype(np.int32)
                cv2.drawContours(viz, [hull_orig], -1, (255, 0, 0), 2) # 파란색 ROI

            combined = np.hstack((frame, viz))
            cv2.imshow('Microvascular Analysis System', combined)
            
            prev_gray = curr_gray
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()
        self.save_and_plot()

    def save_and_plot(self):
        """결과 저장(CSV) 및 그래프 출력"""
        df = pd.DataFrame(self.history)
        df.to_csv("vessel_analysis_results.csv", index=False)
        print("데이터가 'vessel_analysis_results.csv'로 저장되었습니다.")

        plt.figure(figsize=(10, 8))
        plt.subplot(3, 1, 1); plt.plot(df["density"], 'g'); plt.title("Vessel Density")
        plt.subplot(3, 1, 2); plt.plot(df["velocity"], 'b'); plt.title("Flow Velocity")
        plt.subplot(3, 1, 3); plt.plot(df["diameter"], 'r'); plt.title("Vessel Diameter (px)")
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    v_path = r'D:\VIDEO\M-24\video2 (2)\Video_00055.mp4'
    analyzer = RobustVesselAnalyzer(v_path)
    analyzer.process()