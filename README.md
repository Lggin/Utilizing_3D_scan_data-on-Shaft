# LiDAR 3D Point Cloud — Elevator Shaft Analysis

> LiDAR로 취득한 승강로 Point Cloud를 구조화해 **벽면·개구부·레일 설치 후보 영역**을 분석하는 3D 센서 데이터 프로젝트

## Overview

엘리베이터 승강로 내부를 LiDAR로 스캔해 얻은 비정형 PCD(Point Cloud Data)를 전처리하고, 구조적 특징을 추출하는 알고리즘을 개발했습니다.

프로젝트의 주요 목표는 다음 두 가지였습니다.

1. **승강로 개구부 밑면 위치 탐지**
2. **승강로 내벽 분석을 통한 레일 설치 적합 영역 탐색**

단순 시각화가 아니라, 센서 좌표를 정렬하고 2D 격자·밀도·통계량으로 구조화한 뒤 다시 3D 영역으로 복원하는 데이터 처리 흐름을 구현했습니다.

---

## Pipeline

```text
Raw LiDAR Point Cloud
        ↓
Coordinate Rotation / Alignment
        ↓
XZ Plane Projection
        ↓
5 cm Grid Generation
        ↓
Point Density Analysis
        ↓
Wall Surface Candidate Extraction
        ↓
Median / Mode-based Filtering
        ↓
3D Reconstruction
        ↓
Sliding Window + Depth Std Analysis
        ↓
Opening / Flat-area Candidate Detection
```

---

## Core Methods

### 1. Coordinate Alignment

승강로 스캔 데이터의 방향을 맞추기 위해 X/Y/Z축 회전 행렬을 적용해 Point Cloud를 정렬했습니다.

### 2. XZ Projection & Grid Density

3D Point Cloud를 XZ 평면으로 정사영하고 기본 **0.05 m(5 cm) 격자**로 나누어 각 셀의 포인트 밀도를 계산했습니다.

각 X 위치에서 밀도가 높은 Z 영역을 우선적으로 선택해 벽면 후보를 찾고, 중앙값·최빈값 기반 필터를 적용해 노이즈 영향을 줄였습니다.

### 3. 3D Reconstruction

선택된 2D 격자와 원본 3D 포인트의 매핑을 유지하여, 필터링한 영역을 다시 3차원 Point Cloud로 복원하고 Plotly로 시각화했습니다.

### 4. Opening Detection

실제 개구부 크기를 반영한 Sliding Window를 이용했습니다.

- 기본 window width: **1.2 m**
- 기본 window height: **0.1 m**
- 이동 step: **0.01 m**

각 window 내부의 깊이 방향(Z) 표준편차를 계산해 상대적으로 평탄한 영역을 탐색하고 개구부 및 구조 후보를 분석했습니다.

---

## What I Implemented

- LiDAR Point Cloud 좌표 회전·정렬
- XZ 평면 정사영 및 5 cm Grid 생성
- Grid별 Point Density 계산 및 고밀도 영역 추출
- Median / Mode 기반 이상치 필터링
- 선택 영역의 3D Point Cloud 복원 및 시각화
- Sliding Window 기반 개구부 후보 탐지
- 깊이 방향 표준편차를 이용한 평탄 영역 평가

---

## Tech Stack

`Python` `NumPy` `SciPy` `Matplotlib` `Plotly` `Point Cloud Processing`

---

## Key Learning

이 프로젝트를 통해 센서 원천 데이터는 바로 모델에 넣기보다 먼저 **좌표계·밀도·노이즈·공간 구조를 이해하고 문제에 맞게 구조화하는 과정**이 중요하다는 점을 배웠습니다.

특히 3D 데이터를 2D 격자 통계로 단순화하고 다시 원본 좌표로 복원하는 방식으로, 복잡한 Point Cloud 문제를 해석 가능한 데이터 처리 단계로 분해해 해결했습니다.