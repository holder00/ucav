"""
AeroSandbox による航空機全体の最適化デモンストレーション
- グライダーの3D形状定義(主翼、水平尾翼、垂直尾翼、胴体)
- 渦格子法(VLM)による空力解析
- 全要素を含めた揚抗比(L/D)の最大化
- 全翼面の翼弦分布と胴体形状の同時最適化
"""

import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p


# ==============================================================================
# 1. 翼型の定義
# ==============================================================================
wing_airfoil = asb.Airfoil("sd7037")      # 主翼用翼型
tail_airfoil = asb.Airfoil("naca0010")    # 尾翼用翼型


# ==============================================================================
# 2. 航空機の3D形状定義(Peter's Glider)
# ==============================================================================
# 全ての距離はメートル、角度は度数法
airplane = asb.Airplane(
    name="Peter's Glider",
    xyz_ref=[0, 0, 0],  # 重心位置
    
    wings=[
        # ----------------------------------------------------------------------
        # 主翼
        # ----------------------------------------------------------------------
        asb.Wing(
            name="Main Wing",
            symmetric=True,  # XZ平面で左右対称
            xsecs=[
                # 翼根
                asb.WingXSec(
                    xyz_le=[0, 0, 0],      # 前縁位置(翼の前縁からの相対座標)
                    chord=0.18,            # 翼弦長[m]
                    twist=2,               # ねじり角[度]
                    airfoil=wing_airfoil,
                ),
                # 翼中央
                asb.WingXSec(
                    xyz_le=[0.01, 0.5, 0],
                    chord=0.16,
                    twist=0,
                    airfoil=wing_airfoil,
                ),
                # 翼端
                asb.WingXSec(
                    xyz_le=[0.08, 1, 0.1],
                    chord=0.08,
                    twist=-2,
                    airfoil=wing_airfoil,
                ),
            ],
        ),
        
        # ----------------------------------------------------------------------
        # 水平尾翼
        # ----------------------------------------------------------------------
        asb.Wing(
            name="Horizontal Stabilizer",
            symmetric=True,
            xsecs=[
                # 翼根
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=0.1,
                    twist=-10,
                    airfoil=tail_airfoil,
                ),
                # 翼端
                asb.WingXSec(
                    xyz_le=[0.02, 0.17, 0],
                    chord=0.08,
                    twist=-10,
                    airfoil=tail_airfoil,
                ),
            ],
        ).translate([0.6, 0, 0.06]),  # 後方に移動
        
        # ----------------------------------------------------------------------
        # 垂直尾翼
        # ----------------------------------------------------------------------
        asb.Wing(
            name="Vertical Stabilizer",
            symmetric=False,  # 左右対称ではない
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=0.1,
                    twist=0,
                    airfoil=tail_airfoil,
                ),
                asb.WingXSec(
                    xyz_le=[0.04, 0, 0.15],
                    chord=0.06,
                    twist=0,
                    airfoil=tail_airfoil,
                ),
            ],
        ).translate([0.6, 0, 0.07]),  # 後方に移動
    ],
    
    # --------------------------------------------------------------------------
    # 胴体
    # --------------------------------------------------------------------------
    fuselages=[
        asb.Fuselage(
            name="Fuselage",
            xsecs=[
                asb.FuselageXSec(
                    xyz_c=[0.8 * xi - 0.1, 0, 0.1 * xi - 0.03],
                    radius=0.6 * asb.Airfoil("dae51").local_thickness(x_over_c=xi),
                )
                for xi in np.cosspace(0, 1, 30)
            ],
        )
    ],
)


# ==============================================================================
# 3. 渦格子法(VLM)による空力解析(初期形状)
# ==============================================================================
print("\n" + "="*70)
print("初期形状の空力解析")
print("="*70)

vlm = asb.VortexLatticeMethod(
    airplane=airplane,
    op_point=asb.OperatingPoint(
        velocity=25,  # 速度[m/s]
        alpha=5,      # 迎角[度]
    ),
)

# 解析実行
aero = vlm.run()

# 結果表示
for k, v in aero.items():
    print(f"{k.rjust(4)} : {v}")

initial_L_over_D = aero["CL"] / aero["CD"]
print(f"\n初期揚抗比 L/D: {initial_L_over_D:.4f}")

# 可視化
vlm.draw(show_kwargs=dict(jupyter_backend="static"))


# ==============================================================================
# 4. 最大揚抗比(L/D)の迎角最適化(全要素を含む)
# ==============================================================================
print("\n" + "="*70)
print("最大揚抗比の迎角最適化(全要素を含む)")
print("="*70)

opti = asb.Opti()
alpha = opti.variable(init_guess=5)

vlm = asb.VortexLatticeMethod(
    airplane=airplane,
    op_point=asb.OperatingPoint(velocity=25, alpha=alpha),
    align_trailing_vortices_with_wind=False,
)

aero = vlm.run()
L_over_D = aero["CL"] / aero["CD"]

opti.minimize(-L_over_D)
sol = opti.solve()

best_alpha = sol(alpha)
best_L_over_D = sol(L_over_D)
print(f"最大L/Dを与える迎角: {best_alpha:.3f}°")
print(f"最大揚抗比 L/D: {best_L_over_D:.4f}")


# ==============================================================================
# 5. 全翼面と胴体の包括的最適化
# ==============================================================================
print("\n" + "="*70)
print("全航空機要素の包括的最適化")
print("="*70)

opti = asb.Opti()

# ------------------------------------------------------------------------------
# 主翼の最適化パラメータ
# ------------------------------------------------------------------------------
N_main = 16  # 主翼のセクション数
N = N_main
section_y_main = np.sinspace(0, 1, N, reverse_spacing=True)
chords_main = opti.variable(init_guess=np.linspace(0.18, 0.08, N_main))
twist_main = opti.variable(init_guess=np.linspace(2, -2, N_main))

main_wing = asb.Wing(
    name="Main Wing (Optimized)",
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=[
                -0.25 * chords_main[i],
                section_y_main[i],
                0.1 * section_y_main[i],  # わずかな上反角
            ],
            chord=chords_main[i],
            twist=twist_main[i],
            airfoil=wing_airfoil,
        )
        for i in range(N_main)
    ],
)

# ------------------------------------------------------------------------------
# 水平尾翼の最適化パラメータ
# ------------------------------------------------------------------------------
N_htail = 8  # 水平尾翼のセクション数
section_y_htail = np.sinspace(0, 0.17, N_htail, reverse_spacing=True)
chords_htail = opti.variable(init_guess=np.linspace(0.1, 0.08, N_htail))
twist_htail = opti.variable(init_guess=-10 * np.ones(N_htail))
htail_position_x = opti.variable(init_guess=0.6, lower_bound=0.5, upper_bound=0.8)

horizontal_stabilizer = asb.Wing(
    name="Horizontal Stabilizer (Optimized)",
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=[
                -0.25 * chords_htail[i],
                section_y_htail[i],
                0,
            ],
            chord=chords_htail[i],
            twist=twist_htail[i],
            airfoil=tail_airfoil,
        )
        for i in range(N_htail)
    ],
).translate([htail_position_x, 0, 0.06])

# ------------------------------------------------------------------------------
# 垂直尾翼の最適化パラメータ
# ------------------------------------------------------------------------------
N_vtail = 6  # 垂直尾翼のセクション数
section_z_vtail = np.sinspace(0, 0.15, N_vtail, reverse_spacing=True)
chords_vtail = opti.variable(init_guess=np.linspace(0.1, 0.06, N_vtail))
vtail_position_x = opti.variable(init_guess=0.6, lower_bound=0.5, upper_bound=0.8)

vertical_stabilizer = asb.Wing(
    name="Vertical Stabilizer (Optimized)",
    symmetric=False,
    xsecs=[
        asb.WingXSec(
            xyz_le=[
                -0.25 * chords_vtail[i],
                0,
                section_z_vtail[i],
            ],
            chord=chords_vtail[i],
            twist=0,
            airfoil=tail_airfoil,
        )
        for i in range(N_vtail)
    ],
).translate([vtail_position_x, 0, 0.07])

# ------------------------------------------------------------------------------
# 胴体の最適化パラメータ
# ------------------------------------------------------------------------------
N_fuselage = 20  # 胴体のセクション数
fuselage_positions = np.cosspace(0, 1, N_fuselage)
fuselage_radii = opti.variable(
    init_guess=np.array([
        0.6 * asb.Airfoil("dae51").local_thickness(x_over_c=xi)
        for xi in fuselage_positions
    ])
)

fuselage = asb.Fuselage(
    name="Fuselage (Optimized)",
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[0.8 * fuselage_positions[i] - 0.1, 0, 0.1 * fuselage_positions[i] - 0.03],
            radius=fuselage_radii[i],
        )
        for i in range(N_fuselage)
    ],
)

# ------------------------------------------------------------------------------
# 最適化する航空機の構築
# ------------------------------------------------------------------------------
airplane_optimized = asb.Airplane(
    name="Optimized Glider",
    xyz_ref=[0, 0, 0],
    wings=[main_wing, horizontal_stabilizer, vertical_stabilizer],
    fuselages=[fuselage],
)

# ------------------------------------------------------------------------------
# 制約条件の設定
# ------------------------------------------------------------------------------

# 主翼の制約
opti.subject_to([
    chords_main > 0.05,              # 最小翼弦長
    chords_main < 0.25,              # 最大翼弦長
    main_wing.area() > 0.22,         # 最小翼面積
    main_wing.area() < 0.28,         # 最大翼面積
    np.diff(chords_main) <= 0,       # 翼端に向かって減少
    twist_main > -5,                 # 最小ねじり角
    twist_main < 5,                  # 最大ねじり角
])

# 水平尾翼の制約
opti.subject_to([
    chords_htail > 0.04,             # 最小翼弦長
    chords_htail < 0.15,             # 最大翼弦長
    horizontal_stabilizer.area() > 0.025,  # 最小尾翼面積
    horizontal_stabilizer.area() < 0.04,   # 最大尾翼面積
    np.diff(chords_htail) <= 0,      # 翼端に向かって減少
    twist_htail > -15,               # 最小ねじり角
    twist_htail < -5,                # 最大ねじり角
])

# 垂直尾翼の制約
opti.subject_to([
    chords_vtail > 0.04,             # 最小翼弦長
    chords_vtail < 0.15,             # 最大翼弦長
    vertical_stabilizer.area() > 0.015,  # 最小尾翼面積
    vertical_stabilizer.area() < 0.03,   # 最大尾翼面積
    np.diff(chords_vtail) <= 0,      # 上端に向かって減少
])

# 胴体の制約
opti.subject_to([
    fuselage_radii > 0.01,           # 最小半径
    fuselage_radii < 0.08,           # 最大半径
])

# 尾翼位置の連動(同じ位置に配置)
opti.subject_to(htail_position_x == vtail_position_x)

# ------------------------------------------------------------------------------
# 作動点と空力解析
# ------------------------------------------------------------------------------
alpha = opti.variable(init_guess=5, lower_bound=-5, upper_bound=15)

op_point = asb.OperatingPoint(
    velocity=25,
    alpha=alpha,
)

vlm_opt = asb.VortexLatticeMethod(
    airplane=airplane_optimized,
    op_point=op_point,
    spanwise_resolution=2,
    chordwise_resolution=8,
)

aero_opt = vlm_opt.run()

# 揚力制約(一定の揚力を保つ)
opti.subject_to(aero_opt["CL"] > 0.4)

# 縦安定性の制約(ピッチングモーメント係数)
opti.subject_to(aero_opt["Cm"] < 0)  # 負のピッチングモーメント(安定)

# ------------------------------------------------------------------------------
# 目的関数:抗力を最小化(=揚抗比を最大化)
# ------------------------------------------------------------------------------
opti.minimize(aero_opt["CD"])

# 最適化実行
print("\n最適化を実行中...")
sol = opti.solve(verbose=True)

# ------------------------------------------------------------------------------
# 最適化結果の表示
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("最適化結果")
print("="*70)

print("\n[主翼]")
print(f"  翼面積: {sol(main_wing.area()):.4f} m²")
print(f"  翼根翼弦長: {sol(chords_main[0]):.4f} m")
print(f"  翼端翼弦長: {sol(chords_main[-1]):.4f} m")
print(f"  翼根ねじり角: {sol(twist_main[0]):.2f}°")
print(f"  翼端ねじり角: {sol(twist_main[-1]):.2f}°")

print("\n[水平尾翼]")
print(f"  翼面積: {sol(horizontal_stabilizer.area()):.4f} m²")
print(f"  翼根翼弦長: {sol(chords_htail[0]):.4f} m")
print(f"  翼端翼弦長: {sol(chords_htail[-1]):.4f} m")
print(f"  前後位置: {sol(htail_position_x):.4f} m")

print("\n[垂直尾翼]")
print(f"  翼面積: {sol(vertical_stabilizer.area()):.4f} m²")
print(f"  翼根翼弦長: {sol(chords_vtail[0]):.4f} m")
print(f"  翼端翼弦長: {sol(chords_vtail[-1]):.4f} m")
print(f"  前後位置: {sol(vtail_position_x):.4f} m")

print("\n[胴体]")
print(f"  最大半径: {sol(np.max(fuselage_radii)):.4f} m")
print(f"  最小半径: {sol(np.min(fuselage_radii)):.4f} m")

print("\n[空力性能]")
print(f"  最適迎角: {sol(alpha):.3f}°")
print(f"  揚力係数 CL: {sol(aero_opt['CL']):.4f}")
print(f"  抗力係数 CD: {sol(aero_opt['CD']):.4f}")
print(f"  揚抗比 L/D: {sol(aero_opt['CL'] / aero_opt['CD']):.4f}")
print(f"  ピッチングモーメント係数 Cm: {sol(aero_opt['Cm']):.4f}")

print("\n[性能向上]")
print(f"  初期 L/D: {initial_L_over_D:.4f}")
print(f"  最適化後 L/D: {sol(aero_opt['CL'] / aero_opt['CD']):.4f}")
print(f"  改善率: {(sol(aero_opt['CL'] / aero_opt['CD']) / initial_L_over_D - 1) * 100:.2f}%")

# 最適化された機体の可視化
vlm_optimized = sol(vlm_opt)
vlm_optimized.draw(show_kwargs=dict(jupyter_backend="static"))


# ==============================================================================
# 6. 翼弦分布の可視化
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 主翼の翼弦分布
ax1 = axes[0, 0]
ax1.plot(section_y_main, sol(chords_main), '.-', label='最適化後', linewidth=2, markersize=8)
ax1.plot([0, 0.5, 1], [0.18, 0.16, 0.08], 'o--', label='初期形状', alpha=0.5)
ax1.set_xlabel('スパン方向位置 [m]', fontsize=12)
ax1.set_ylabel('翼弦長 [m]', fontsize=12)
ax1.set_title('主翼の翼弦分布', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 主翼のねじり角分布
ax2 = axes[0, 1]
ax2.plot(section_y_main, sol(twist_main), '.-', label='最適化後', linewidth=2, markersize=8)
ax2.plot([0, 0.5, 1], [2, 0, -2], 'o--', label='初期形状', alpha=0.5)
ax2.set_xlabel('スパン方向位置 [m]', fontsize=12)
ax2.set_ylabel('ねじり角 [°]', fontsize=12)
ax2.set_title('主翼のねじり角分布', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 水平尾翼の翼弦分布
ax3 = axes[1, 0]
ax3.plot(section_y_htail, sol(chords_htail), '.-', label='最適化後', linewidth=2, markersize=8)
ax3.plot([0, 0.17], [0.1, 0.08], 'o--', label='初期形状', alpha=0.5)
ax3.set_xlabel('スパン方向位置 [m]', fontsize=12)
ax3.set_ylabel('翼弦長 [m]', fontsize=12)
ax3.set_title('水平尾翼の翼弦分布', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 垂直尾翼の翼弦分布
ax4 = axes[1, 1]
ax4.plot(section_z_vtail, sol(chords_vtail), '.-', label='最適化後', linewidth=2, markersize=8)
ax4.plot([0, 0.15], [0.1, 0.06], 'o--', label='初期形状', alpha=0.5)
ax4.set_xlabel('高さ方向位置 [m]', fontsize=12)
ax4.set_ylabel('翼弦長 [m]', fontsize=12)
ax4.set_title('垂直尾翼の翼弦分布', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/optimization_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n翼弦分布のグラフを保存しました: optimization_results.png")


# ==============================================================================
# 7. 胴体形状の可視化
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    fuselage_positions,
    sol(fuselage_radii),
    '.-',
    label='最適化後',
    linewidth=2,
    markersize=8
)

initial_radii = [
    0.6 * asb.Airfoil("dae51").local_thickness(x_over_c=xi)
    for xi in fuselage_positions
]
ax.plot(
    fuselage_positions,
    initial_radii,
    'o--',
    label='初期形状',
    alpha=0.5
)

ax.set_xlabel('機体前後方向位置 (正規化)', fontsize=12)
ax.set_ylabel('胴体半径 [m]', fontsize=12)
ax.set_title('胴体形状の最適化', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/fuselage_optimization.png', dpi=150, bbox_inches='tight')
plt.show()

print("胴体形状のグラフを保存しました: fuselage_optimization.png")

print("\n" + "="*70)
print("最適化完了!")
print("="*70)