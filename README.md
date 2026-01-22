# 🎥 Vessel Video Analysis Pipeline

...

---

## 📂 1. 이미지 추출 (Image Extraction)
비디오 파일에서 설정된 시간 간격으로 프레임을 추출합니다.

- **스크립트 경로:** `D:\VIDEO\code\image_code\img_extract.py`
- **입력:** `.mp4` 파일 (1개)
- **출력:** 설정된 초 단위 프레임 이미지
- **비고:** 분석에 필요한 기초 정지 영상 확보용

---

## 🎯 2. ROI 설정 (Region of Interest)
분석 효율을 높이기 위해 영상 내 주요 관심 영역을 선정합니다. (상위 n장 추출)

| 구분 | 스크립트 경로 | 특징 | 출력 결과 |
|:---:|:---|:---|:---|
| **일반** | `D:\VIDEO\code\grid_roi_code\final_roi_um_no_crop.py` | 4분할 미사용 | 원본 크기 ROI (No Crop) |
| **분할** | `D:\VIDEO\code\grid_roi_code\final_roi_um2.py` | 4분할 적용 | 크롭된 ROI (Crop) |

- **입력:** `.mp4` 파일 2개 (Baseline / Target)
- **출력:** 설정된 초 단위 간격 내 상위 **n장**의 결과물

---

## 📸 3. 빛 픽셀 제거 (Light Pixel Removal & Inpainting)
영상의 빛 번짐이나 노이즈를 제거하고 손실된 부분을 보정합니다.

- **스크립트 경로:** `D:\VIDEO\code\vessel\remove_white.py`
- **입력:** ROI 설정이 완료된 `.tiff` 파일 (1개)
- **출력:**
    - 빛 픽셀 이진 이미지 3종 (Small, Large, Full)
    - 노이즈 제거 및 Inpainting이 완료된 최종 이미지 (1개)

---

## 🌿 4. 혈관 네트워크 추출 (Vessel Extraction)
혈관 구조를 추출합니다.

- **위치:** `D:\VIDEO\code\video2\`

| 버전 | 스크립트명 | 설명 및 특징 |
|:---:|:---|:---|
| **v1** | `green_tiff3.py` | Sato 필터 기반 기본 추출 로직 |
| **v2** | `green_tiff3_xblur.py` | 가우시안 블러를 제거하여 경계선 강조 |
| **v3** | `green_vessel.py` | **얇은 혈관** 중심 추출 최적화 |
| **v4** | `green_vessel_background.py` | 배경 노이즈 제거 로직 강화 버전 |
| **New** | `green_vessel2.py` | 현재 실험 중인 신규 알고리즘 |

---