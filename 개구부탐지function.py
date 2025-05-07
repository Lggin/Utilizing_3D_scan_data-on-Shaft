import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import mode
from tqdm import tqdm

# 회전 함수 (각도 단위: 도)
def rotate_point_cloud(points, angle_x, angle_y, angle_z):
    angle_x, angle_y, angle_z = np.deg2rad([angle_x, angle_y, angle_z])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return (R @ points.T).T

# xz 평면 격자 생성 및 정사영 함수
def create_xz_grid(points, grid_size=0.05):  # XZ 평면 기준 격자 생성
    """ XZ 평면에 정사영하여 격자를 생성하고 포인트 밀도를 계산 """
    if points.size == 0:
        return np.array([]), None, None, {}

    xz_points = points[:, [0, 2]]  # XZ 평면 좌표만 사용
    min_bound = xz_points.min(axis=0)
    max_bound = xz_points.max(axis=0)

    x_bins = int((max_bound[0] - min_bound[0]) / grid_size) + 1
    z_bins = int((max_bound[1] - min_bound[1]) / grid_size) + 1

    grid_counts = np.zeros((x_bins, z_bins))
    grid_positions = []
    density_values = []
    grid_to_points = {}

    for i, (x, z) in enumerate(xz_points):
        x_idx = int((x - min_bound[0]) / grid_size)
        z_idx = int((z - min_bound[1]) / grid_size)
        grid_counts[x_idx, z_idx] += 1

        if (x_idx, z_idx) not in grid_to_points:
            grid_to_points[(x_idx, z_idx)] = []
        grid_to_points[(x_idx, z_idx)].append(points[i])

    for x in range(x_bins):
        for z in range(z_bins):
            grid_center = [
                min_bound[0] + x * grid_size + grid_size / 2,
                min_bound[1] + z * grid_size + grid_size / 2
            ]
            grid_positions.append(grid_center)
            density_values.append(grid_counts[x, z])

    return np.array(grid_positions, dtype=np.float64), np.array(density_values), grid_to_points, grid_counts

# 격자들을 밀도에 따라 색깔별로 시각화하는 함수
def visualize_xz_grid(grid_positions, density_values):
    """ XZ 평면의 격자를 시각화 (밀도별 색상 표시) """
    if grid_positions.size == 0:
        print("No grid positions to visualize.")
        return

    plt.figure(figsize=(10, 8))
    x_vals, z_vals = grid_positions[:, 0], grid_positions[:, 1]

    scatter = plt.scatter(x_vals, z_vals, c=density_values, cmap='jet', s=50, edgecolors='k')
    plt.colorbar(scatter, label='Point Density')
    plt.xlabel("X Axis")
    plt.ylabel("Z Axis")
    plt.title("Projected XZ Grid Density")
    plt.show()

# 최대 밀도를 가진 격자(selected_grids)를 x값 별로 추출해주는 함수
def select_highest_density_grids(grid_counts):
    """ 각 x 인덱스별로 가장 밀도가 높은 z 인덱스를 선택 """
    selected_grids = []
    x_bins, z_bins = grid_counts.shape

    for x in range(x_bins):
        max_z_idx = np.argmax(grid_counts[x])  # 해당 x에서 가장 밀도가 높은 z 찾기
        if grid_counts[x, max_z_idx] > 0:  # 밀도가 0보다 큰 경우만 선택
            selected_grids.append((x, max_z_idx))

    return selected_grids

# 추출된 격자들의 z값 중 중앙값을 구해주는 함수
def compute_median_z(selected_grids):
    """ 선택된 그리드의 z 중앙값 계산 """
    z_values = [z for _, z in selected_grids]
    return np.median(z_values)

# 추출된 격자들의 z값 중 최빈값을 구해주는 함수
def compute_mode_z(selected_grids):
    """ 선택된 그리드의 z 중앙값 계산 """
    z_values = [z for _, z in selected_grids]
    z_mode_result = mode(z_values, keepdims=True)
    z_mode = z_mode_result[0]
    return z_mode

# 최빈값을 기준으로 위아래 2칸에 해당하는 selected_grids 선택 함수
def filter_by_median_z(selected_grids, mode_z, tolerance=2):
    """ 중앙값 기준으로 특정 범위 내의 격자만 선택 """
    return [(x, z) for x, z in selected_grids if abs(mode_z - z) < tolerance]

# 격자 내 점들을 다시 3d로 전환해주는 함수
def get_selected_grid_points(selected_grids, grid_to_points):
    """ 선택된 그리드 내의 점들을 반환 """
    selected_points = []
    for grid in selected_grids:
        selected_points.extend(grid_to_points.get(grid, []))
    return np.array(selected_points) if selected_points else np.array([])

# 점들을 시각화 시키는 함수
def visualize_points(selected_points):
    fig_3d = go.Figure()
    fig_3d.add_trace(go.Scatter3d(
        x=selected_points[:, 0], y=selected_points[:, 1], z=selected_points[:, 2],
        mode='markers',
        marker=dict(size=1, color='white', opacity=1)
        ))
    fig_3d.update_layout(
        title="Filtered 3D Point Cloud",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        width=900, height=900,
        template="plotly_dark"
    )
    fig_3d.show()

# 점들의 z값의 평균과 최빈값을 구하고 시각화 해주는 함수
def getting_info(selected_points):
    z_values = selected_points[:, 2]
    z_min = np.min(z_values)
    z_max = np.max(z_values)
    z_median = np.median(z_values)
    z_mean = np.mean(z_values)
    z_mode_result = mode(z_values, keepdims=True)
    z_mode = z_mode_result.mode[0] if z_mode_result.count.size > 0 else np.nan

    return z_min, z_max, z_mean, z_median, z_mode

# 점들의 z값의 평균과 최빈값을 구하고 시각화 해주는 함수
def getting_meanandmode(selected_points):
    z_values = selected_points[:, 2]
    z_min = np.min(z_values)
    z_max = np.max(z_values)
    z_median = np.median(z_values)
    z_mean = np.mean(z_values)
    z_mode_result = mode(z_values, keepdims=True)
    z_mode = z_mode_result.mode[0] if z_mode_result.count.size > 0 else np.nan
    print(z_min, z_max, z_mean, z_median, z_mode)

    # X축과 Z축의 범위 설정
    X_MIN, X_MAX = -0.5, 0.5
    Z_MIN, Z_MAX = -3.5, -3

    # Z 기준 산점도 그리기
    fig = go.Figure()

    # 5.필터링된 점들의 산점도
    fig.add_trace(go.Scatter(
        x=selected_points[:, 0],
        y=selected_points[:, 2],
        mode='markers',
        marker=dict(size=2, color='white', opacity=1),
        name="Filtered Points"
        ))

    # 적합된 Z 값 (평균) 직선 표시
    fig.add_trace(go.Scatter(
        x=[X_MIN, X_MAX],
        y=[z_mean, z_mean],
        mode='lines',
        line=dict(color='#CD00CD', width=3),
        name=f"Fitted Z (Mean) = {z_mean:.3f}"
        ))

    # 적합된 Z 값 (최빈값) 직선 표시
    fig.add_trace(go.Scatter(
        x=[X_MIN, X_MAX],
        y=[z_mode, z_mode],
        mode='lines',
        line=dict(color='#FFFF00', width=3),
        name=f"Fitted Z (Mode) = {z_mode:.3f}"
        ))

    # 그래프 레이아웃 설정
    fig.update_layout(
        title="Scatter Plot of Filtered Points (Z vs X)",
        xaxis_title="X Value",
        yaxis_title="Z Value",
        template="plotly_dark",
        width=900,
        height=900
        )

    fig.show()

# 개구부의 크기 맞춰서 표준편차에 따라 가장 개구부를 잘 나타내는 곳을 찾는 함수
def extract_best_opening(
    points: np.ndarray,
    window_width: float = 1.2,    ###개구부의 가로폭
    window_height: float = 0.1,  ###개구부의 세로폭
    step: float = 0.01,
    min_points_threshold: int = 0
    ) -> np.ndarray:

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    best_std = float('inf')
    best_window = None


    for y in tqdm(np.arange(y_min, y_max - window_height, step), disable = True):
        for x in np.arange(x_min, x_max - window_width, step):
            x0, x1 = x, x + window_width
            y0, y1 = y, y + window_height

            mask = (points[:, 0] >= x0) & (points[:, 0] <= x1) & \
                   (points[:, 1] >= y0) & (points[:, 1] <= y1)

            window_points = points[mask]

            if len(window_points) < min_points_threshold:
                continue

            z_std = np.std(window_points[:, 2])

            if z_std <= best_std:
                best_std = z_std
                best_window = (x0, x1, y0, y1)

    if best_window is None:
        print("⚠️ 적절한 개구부 영역을 찾지 못했습니다.")
        return np.empty((0, 3))  # 빈 배열 반환


    #print(f"✅ 최적 표준편차: {best_std:.5f}, 위치: {best_window}")

    x0, x1, y0, y1 = best_window
    final_mask = (points[:, 0] >= x0) & (points[:, 0] <= x1) & \
                 (points[:, 1] >= y0) & (points[:, 1] <= y1)

    return points[final_mask]

