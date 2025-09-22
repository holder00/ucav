import math
import numpy as np
from typing import List, Tuple

class StanleyController:
    def __init__(self, k: float = 1.0, k_soft: float = 1.0, lookahead_dist: float = 1.0):
        """
        Stanley制御のパラメータ
        k: 横方向誤差のゲイン
        k_soft: 速度スケーリングの調整
        lookahead_dist: 先読み距離（最近点探索の補助）
        """
        self.k = k
        self.k_soft = k_soft
        self.lookahead_dist = lookahead_dist

    def find_nearest_point(self, x: float, y: float, path: List[Tuple[float, float]]) -> Tuple[int, float, float]:
        """
        現在位置(x, y)から目標経路上の最近点を探索
        戻り値: (最近点のインデックス, 最近点のx座標, 最近点のy座標)
        """
        distances = [math.sqrt((x - px)**2 + (y - py)**2) for px, py in path]
        nearest_idx = np.argmin(distances)
        return nearest_idx, path[nearest_idx][0], path[nearest_idx][1]

    def calculate_cross_track_error(self, x: float, y: float, px: float, py: float) -> float:
        """
        横方向誤差を計算（点(x, y)から最近点(px, py)への距離）
        """
        return math.sqrt((x - px)**2 + (y - py)**2)

    def calculate_heading_error(self, theta: float, path: List[Tuple[float, float]], nearest_idx: int) -> float:
        """
        角度誤差を計算（車両のヨー角と経路の接線角度の差）
        """
        # 最近点の次の点を基に接線角度を計算
        if nearest_idx < len(path) - 1:
            px1, py1 = path[nearest_idx]
            px2, py2 = path[nearest_idx + 1]
            path_angle = math.atan2(py2 - py1, px2 - px1)
        else:
            # 終点では前の点から角度を推定
            px1, py1 = path[nearest_idx - 1]
            px2, py2 = path[nearest_idx]
            path_angle = math.atan2(py2 - py1, px2 - px1)
        
        # 角度誤差を計算し、[-pi, pi]に正規化
        heading_error = path_angle - theta
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        return heading_error

    def stanley_control(self, x: float, y: float, theta: float, v: float, path: List[Tuple[float, float]]) -> float:
        """
        Stanley制御による操舵角の計算
        x, y: 車両の現在位置
        theta: 車両のヨー角（ラジアン）
        v: 車両の速度
        path: 目標経路の点列 [(x1, y1), (x2, y2), ...]
        """
        # 最近点を探索
        nearest_idx, px, py = self.find_nearest_point(x, y, path)
        
        # 横方向誤差
        e = self.calculate_cross_track_error(x, y, px, py)
        
        # 角度誤差
        theta_e = self.calculate_heading_error(theta, path, nearest_idx)
        
        # Stanley制御の操舵角計算
        steering_angle = theta_e + math.atan2(self.k * e, self.k_soft + v)
        
        # 操舵角を制限（例: ±30度）
        max_steering = math.radians(30)
        steering_angle = max(min(steering_angle, max_steering), -max_steering)
        
        return steering_angle

# シミュレーション例
def simulate_stanley_curve():
    # Stanley制御のインスタンス
    controller = StanleyController(k=2.0, k_soft=1.0, lookahead_dist=1.0)
    
    # 目標経路（曲線: サインカーブを例に）
    path = [(x, math.sin(x)) for x in np.arange(0, 10, 0.1)]
    
    # 車両の初期状態
    x, y = 0.0, 0.5  # 初期位置（経路から0.5ずれた位置）
    theta = math.radians(0)  # 初期ヨー角
    v = 1.0  # 速度（固定）
    
    # シミュレーション結果の記録
    trajectory = [(x, y)]
    
    # シミュレーションループ
    dt = 0.1  # 時間ステップ
    for _ in range(200):
        # Stanley制御で操舵角を計算
        steering = controller.stanley_control(x, y, theta, v, path)
        
        # 車両の状態更新（簡略化した自転車モデル）
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta += v * math.tan(steering) * dt
        
        # 軌跡を記録
        trajectory.append((x, y))
        
        # ログ出力
        print(f"Pos: ({x:.2f}, {y:.2f}), Theta: {math.degrees(theta):.2f}deg, Steering: {math.degrees(steering):.2f}deg")
    
    # 可視化（Matplotlibを使用）
    import matplotlib.pyplot as plt
    path_x, path_y = zip(*path)
    traj_x, traj_y = zip(*trajectory)
    
    plt.plot(path_x, path_y, 'b-', label='Target Path')
    plt.plot(traj_x, traj_y, 'r--', label='Vehicle Trajectory')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Stanley Control: Curve Following')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    simulate_stanley_curve()