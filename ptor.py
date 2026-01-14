import cv2
import math
import os

# 1. 경로 및 설정값 세팅
video_path = r'D:\VIDEO\M-24\Video_00056_00001.mp4'
FOV_WIDTH_MM = 6.5
IMG_WIDTH_PX = 3840

# 픽셀당 마이크로미터 비율 (1px = N um)
UM_PER_PX = (FOV_WIDTH_MM * 1000) / IMG_WIDTH_PX

# 마우스 클릭 이벤트 처리를 위한 변수
points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # 클릭한 지점에 점 표시
        cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Measurement", display_frame)
        
        if len(points) == 2:
            # 두 점 사이의 픽셀 거리 계산 (피타고라스 정리)
            pixel_dist = math.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
            # um 단위로 변환
            um_dist = pixel_dist * UM_PER_PX
            
            print(f"--- 측정 결과 ---")
            print(f"픽셀 거리: {pixel_dist:.2f} px")
            print(f"실제 거리: {um_dist:.2f} um")
            print(f"실제 거리: {um_dist/1000:.4f} mm")
            
            # 화면에 결과 그리기
            cv2.line(display_frame, points[0], points[1], (0, 255, 0), 2)
            cv2.putText(display_frame, f"{um_dist:.2f} um", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Measurement", display_frame)
            points.clear() # 다음 측정을 위해 초기화

# 2. 영상 불러오기
if not os.path.exists(video_path):
    print(f"파일을 찾을 수 없습니다: {video_path}")
else:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if ret:
        # 4K 영상은 너무 크므로 모니터 확인용으로 리사이즈 (계산은 원본 비율 유지)
        # 만약 원본 크기 그대로 보고 싶다면 아래 resize 줄을 주석 처리하세요.
        display_frame = cv2.resize(frame, (1280, 720))
        
        # 리사이즈 시 비율 조정 (원본 3840 -> 1280 이므로 1/3 축소)
        resize_ratio = 3840 / 1280
        DISPLAY_UM_PER_PX = UM_PER_PX * resize_ratio
        
        # 실제 계산용 비율 업데이트 (화면 표시용)
        UM_PER_PX = DISPLAY_UM_PER_PX

        print("이미지 위에서 두 점을 클릭하면 거리가 계산됩니다. (종료: 'q' 또는 'ESC')")
        cv2.imshow("Measurement", display_frame)
        cv2.setMouseCallback("Measurement", mouse_callback)
        
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
                break
    else:
        print("영상을 읽어올 수 없습니다.")

    cap.release()
    cv2.destroyAllWindows()