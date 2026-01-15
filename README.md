**video 이미지 추출**
- D:\VIDEO\code\image_code\img_extract.py
- 입력: .mp4 파일 1개
- 출력: 설정된 초 단위로 이미지 추출

**ROI 실행 파일**

- 4분할 x ROI 코드: D:\VIDEO\code\grid_roi_code\final_roi_um_no_crop.py
- 입력: .mp4 파일 2개 (baseline/target)
- 출력: 설정된 초 단위 간 설정된 상위 n장의 ROI 결과 (no crop)
- 4분할 o ROI 코드: D:\VIDEO\code\grid_roi_code\final_roi_um2.py
- 입력: .mp4 파일 2개 (baseline/target)
- 출력: 설정된 초 단위 간 설정된 상위 n장의 ROI 결과 (crop)

**빛 픽셀 제거 파일**
- 제거 및 inpainting 코드: D:\VIDEO\code\vessel\remove_white.py
- 입력: roi 설정된 .tiff 파일 1개
- 출력: 빛 필셀 이진 이미지 3개 (small_large_full) + 수행 완료 이미지 1개
 
---

**가상환경**
- anaconda 설치 후 추가로 필요한 패키지를 pip 명령어를 통해 설치.
- conda 명령어를 통해서 가상환경 설정.
- llm 물어보면서 과정 수행했습니다!
