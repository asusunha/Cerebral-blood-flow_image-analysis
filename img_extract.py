'''
video 기반 시간대 별 이미지 추출(해상도 유지)
- 현재는 30초 간격으로 crop없이 원본 크기대로 이미지화 중
'''

import imageio
import os
import numpy as np
from PIL import Image

# --- [사용자 설정 경로 및 파일명] ---
BASE_DIR = r"D:\VIDEO"
VIDEO_DIR = os.path.join(BASE_DIR, "M-24")
# VIDEO_NAME = "Video_00055"
VIDEO_NAME = "Video_00056_00060"
VIDEO_EXTENSION = ".mp4"  # 영상의 확장자에 맞게 수정하세요 (.mp4, .avi 등)

VIDEO_PATH = os.path.join(VIDEO_DIR, VIDEO_NAME + VIDEO_EXTENSION)
OUTPUT_DIR = os.path.join(BASE_DIR, "M-24_captured", VIDEO_NAME)

# 캡처 설정
INTERVAL_SEC = 30  # 30초 단위
# ROI_SIZE = (1648, 1424)  # 가로, 세로 추출 크기 (4K 영상 중앙)
ROI_SIZE = (3840, 2160)  # 가로, 세로 추출 크기

def run_capture():
    # 1. 저장 경로 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"폴더 생성 완료: {OUTPUT_DIR}")

    # 2. 비디오 리더 초기화
    try:
        reader = imageio.get_reader(VIDEO_PATH, 'ffmpeg')
        meta = reader.get_meta_data()
        fps = meta['fps']
        v_width, v_height = meta['size']
        duration = meta['duration']
        
        print(f"영상 로드 성공: {v_width}x{v_height}, {fps} FPS")
    except Exception as e:
        print(f"파일을 읽을 수 없습니다. 경로를 확인해주세요: {e}")
        return

    # 3. ROI 좌표 계산 (3840x2160의 중앙)
    roi_w, roi_h = ROI_SIZE
    left = (v_width - roi_w) // 2
    top = (v_height - roi_h) // 2
    right = left + roi_w
    bottom = top + roi_h

    # 4. 루프를 돌며 프레임 추출
    current_sec = 0
    while current_sec <= duration:
        frame_idx = int(current_sec * fps)
        
        try:
            # 특정 프레임 가져오기
            frame = reader.get_data(frame_idx)
            
            # 중앙 ROI 크롭
            roi = frame[top:bottom, left:right]
            
            # 이미지 객체 변환 및 저장
            img = Image.fromarray(roi)
            # 파일명 규칙: Video_00055_30s.tiff
            file_name = f"{VIDEO_NAME}_{current_sec}s.tiff"
            save_path = os.path.join(OUTPUT_DIR, file_name)
            
            # 무손실 TIFF 저장 (compression_level 0)
            img.save(save_path, format='TIFF')
            print(f"저장됨: {file_name}")
            
        except IndexError:
            break
        
        current_sec += INTERVAL_SEC

    reader.close()
    print("--- 모든 작업이 완료되었습니다 ---")

if __name__ == "__main__":
    run_capture()