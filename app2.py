import matplotlib.pyplot as plt
import numpy as np
import math
import streamlit as st
import pandas as pd

# 设置中文字体和负号显示
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 用户输入部分
st.sidebar.header("输入参数")

# 新增的输入部分
material = st.sidebar.selectbox("请选择环焊缝材质", ["X70环焊缝", "X80环焊缝"])
current_density = st.sidebar.number_input("负向电流密度 (mA/cm²)", value=10.0)

# 允许用户手动输入 CTOD 值
manual_ctod = st.sidebar.number_input("手动输入 CTOD 值 (mm),如果需要使用计算值，请点击X删掉该值", value=None)

# CTOD计算部分
if manual_ctod is not None:
    CTOD = manual_ctod  # 如果用户手动输入CTOD，则以其为准
else:
    if material == "X70环焊缝":
        CTOD = 0.36 - 0.104 * math.log(current_density)
    elif material == "X80环焊缝":
        CTOD = 0.18 - 0.031 * math.log(current_density)

# 显示计算的CTOD值
st.sidebar.write(f"预测的 CTOD值 (mm): {CTOD:.4f}")


# 其他输入参数
Y = st.sidebar.number_input("屈服强度 (MPa)", value=555.0)
U = st.sidebar.number_input("抗拉强度 (MPa)", value=625.0)
E = st.sidebar.number_input("弹性模量 (MPa)", value=205000.0)
v = st.sidebar.number_input("泊松比", value=0.3)
CTOD = st.sidebar.number_input("CTOD值 (mm)", value=CTOD)

# 显示图片
# 显示图片
image_url = "https://raw.githubusercontent.com/fushicorrosion/critical-defect-size/main/%E5%9B%BE%E7%89%871.png"
st.sidebar.image(image_url, use_column_width=True)



B = st.sidebar.number_input("管道壁厚 (mm)", value=18.9)
R = st.sidebar.number_input("管道半径 (mm)", value=610.0)

C = st.sidebar.number_input("缺陷半长度 c (mm)", value=100.0)
theta_deg = st.sidebar.number_input("角度值 (度)", value=90.0)

Pm = st.sidebar.number_input("轴向应力 Pm (MPa)", value=161.0)
Pb = st.sidebar.number_input("弯曲应力 Pb (MPa)", value=0.0)
Qm = st.sidebar.number_input("Qm 值", value=0.0)
Qb = st.sidebar.number_input("Qb 值", value=0.0)

# 计算内径 ri
ri = R - B
# 将角度转换为弧度
theta = math.radians(theta_deg)

# 显示输入的参数
params = {
    "屈服强度 (Y)": Y,
    "抗拉强度 (U)": U,
    "弹性模量 (E)": E,
    "泊松比 (v)": v,
    "管道壁厚 (B)": B,
    "管道半径 (R)": R,
    "缺陷半长度 (C)": C,
    "角度 (theta)": theta_deg,
    "轴向应力 (Pm)": Pm,
    "弯曲应力 (Pb)": Pb,
    "Qm 值": Qm,
    "Qb 值": Qb,
    "内径 (ri)": ri,
}



# 继续后续计算和绘图代码...

# 各项系数计算函数
M = 1
ktm = 1
Mkm = 1
ktb = 1
Mkb = 1
km = 1

# 定义计算函数...
def calculate_fw(C, W, a, B):
    if a / (2 * C) == 0:
        return 1.0
    fw = (1 / (np.cos(np.pi * C / W * (a / B) ** 0.5))) ** 0.5
    return fw


def calculate_M1(a, B, C):
    a_c = a / (2 * C)
    if a_c <= 0.5:
        return 1.13 - 0.09 * (a / C)
    else:
        return (C / a) ** 0.5 * (1 + 0.04 * (C / a))


def calculate_M2(a, B, C):
    a_c = a / (2 * C)
    if a_c <= 0.5:
        return 0.89 / (0.2 + (a / C)) - 0.54
    else:
        return 0.2 * (C / a) ** 4


def calculate_M3(a, C):
    a_c = a / (2 * C)
    if a_c <= 0.5:
        return 0.5 - 1 / (0.65 + (a / C)) + 14 * (1 - a / C) ** 24
    else:
        return -0.11 * (C / a) ** 4


def calculate_g(a, B, C, theta):
    a_c = a / (2 * C)
    if a_c <= 0.5:
        return 1 + (0.1 + 0.35 * (a / B) ** 2) * (1 - np.sin(theta)) ** 2
    else:
        return 1 + (0.1 + 0.35 * (C / a) * (a / B) ** 2) * (1 - np.sin(theta)) ** 2


def calculate_f_theta(a, C, theta):
    a_c = a / (2 * C)
    if a_c <= 0.5:
        return ((a / C) ** 2 * np.cos(theta) ** 2 + np.sin(theta) ** 2) ** 0.25
    else:
        return ((C / a) ** 2 * np.sin(theta) ** 2 + np.cos(theta) ** 2) ** 0.25


def calculate_phi(a, C):
    a_c = a / (2 * C)
    if a_c <= 0.5:
        return (1 + 1.464 * (a / C) ** 1.65) ** 0.5
    else:
        return (1 + 1.464 * (C / a) ** 1.65) ** 0.5


def calculate_Mm(a, B, C, theta):
    M1 = calculate_M1(a, B, C)
    M2 = calculate_M2(a, B, C)
    M3 = calculate_M3(a, C)
    g = calculate_g(a, B, C, theta)
    f_theta = calculate_f_theta(a, C, theta)
    phi = calculate_phi(a, C)
    Mm = (M1 + M2 * (a / B) ** 2 + M3 * (a / B) ** 4) * g * f_theta / phi
    return Mm


def calculate_q(a, B, C):
    a_c = a / (2 * C)
    if a_c <= 0.5:
        return 0.2 + (a / C) + 0.6 * (a / B)
    else:
        return 0.2 + (C / a) + 0.6 * (a / B)


def calculate_H1(a, B, C):
    a_c = a / (2 * C)
    if a_c <= 0.5:
        return 1 - 0.34 * (a / B) - 0.11 * (a / C) * (a / B)
    else:
        return 1 - (0.04 + 0.41 * (C / a)) * (a / B) + (0.55 - 1.93 * (C / a) ** 0.75 + 1.38 * (C / a) ** 1.5) * (
                    a / B) ** 2


def calculate_H2(a, B, C):
    a_c = a / (2 * C)
    G1 = -1.22 - 0.12 * (a / C) if a_c <= 0.5 else -2.11 + 0.77 * (C / a)
    G2 = 0.55 - 1.05 * (a / C) ** 0.75 + 0.47 * (a / C) ** 1.5 if a_c <= 0.5 else 0.55 - 0.72 * (
                C / a) ** 0.75 + 0.14 * (C / a) ** 1.5
    return 1 + G1 * (a / B) + G2 * (a / B) ** 2


def calculate_H(a, B, C, theta):
    H1 = calculate_H1(a, B, C)
    H2 = calculate_H2(a, B, C)
    q = calculate_q(a, B, C)
    return H1 + (H2 - H1) * np.sin(theta) ** q


def calculate_Mb(a, B, C, theta):
    H = calculate_H(a, B, C, theta)
    Mm = calculate_Mm(a, B, C, theta)
    return H * Mm


def comprehensive_KI(a, C, R, B, theta, ktm, Mkm, ktb, Mkb, km, Pm, Qm, Pb, Qb):
    # 计算各项中间值
    W = 2 * math.pi * R
    Mm = calculate_Mm(a, B, C, theta)
    Mb = calculate_Mb(a, B, C, theta)
    fw = calculate_fw(C, W, a, B)

    Y_sigma_p = M * fw * (ktm * Mkm * Mm * Pm + ktb * Mkb * Mb * (Pb + (km - 1) * Pm))
    Y_sigma_s = Mm * Qm + Mb * Qb
    Y_sigma = Y_sigma_p + Y_sigma_s

    # 计算 KI
    return (Y_sigma) * math.sqrt(math.pi * a / 1000)
# 计算 μ 和 N
μ = min(0.001 * E / Y, 0.6)
N = 0.3 * (1 - Y / U)
Lmax = 0.5+U/Y/2
# 定义 fL 函数
def fL(L):
    if 0 <= L <= 1:
        # 计算 fL 当 0 <= L <= 1 时的值
        return (1 + 0.5 * L ** 2) ** -0.5 * (0.3 + 0.7 * np.exp(-μ * L ** 6))
    elif 1 < L <= Lmax:
        # 计算 L = 1 时的 fL 值，并作为 fL1
        fL1 = (1 + 0.5 * 1 ** 2) ** -0.5 * (0.3 + 0.7 * np.exp(-μ * 1 ** 6))
        return fL1 * L ** ((N - 1) / (2 * N))
    else:
        # 当 L > Lmax 时，fL = 0
        return 0

# 生成 L 的范围
L_values = np.linspace(0, Lmax+0.0001, 500)  # 生成从 0 到 Lmax 的等间距点
fL_values = np.array([fL(L) for L in L_values])  # 计算对应的 fL 值

# 创建 Streamlit 应用
st.title("临界裂纹尺寸计算器")

# 计算按钮
if st.sidebar.button("计算临界裂纹尺寸"):
    a_values = np.arange(0.01, B, 0.01)  # 每隔 0.01 取一个值，从 0.01 到 B（包含 B）
    Lr_values = []
    Kr_values = []
    a_output = []

    # 计算 Lr 和 Kr
    for a in a_values:
        KI = comprehensive_KI(a, C, R, B, theta, ktm, Mkm, ktb, Mkb, km, Pm, Qm, Pb, Qb)

        # 计算与FAD曲线相交的 a 值
        m1 = 1.517 * (Y / U) ** (-0.3188)
        Kmat = (m1 * Y * CTOD / 1000 * E / (1 - v * v)) ** 0.5

        # 计算参考应力 Pref
        if math.pi * ri >= C + B:
            a2 = a / B / (1 + (B / C))
        else:
            a2 = a / B * (C / math.pi / ri)

        Pref = Pm * (math.pi * (1 - a / B) + 2 * (a / B) * math.sin(C / ri)) / (1 - a / B) / (
                    math.pi - (C / ri) * (a / B)) + 2 * Pb / (3 * (1 - a2) ** 2)

        Kr = KI / Kmat
        Lr = Pref / Y

        Lr_values.append(Lr)
        Kr_values.append(Kr)

        fL_value = fL(Lr)

        if Kr >= fL_value or Lr >= Lmax:
            a_output.append(a)

    # 绘制图形
    plt.figure(figsize=(10, 6))
    plt.plot(L_values, fL_values, label='FAC', color='b', linewidth=2)
    plt.plot(Lr_values, Kr_values, label='(Lr, Kr)', color='g', linestyle='--')

    first_a_value = None  # 存储第一个满足条件的 a 值
    for a in a_output:
        idx = np.where(a_values == a)[0][0]  # 获取对应的索引
        plt.scatter(Lr_values[idx], Kr_values[idx], color='r', s=100)
        if first_a_value is None:  # 找到第一个满足条件的 a
            first_a_value = a

    if first_a_value is not None:
        first_idx = np.where(a_values == first_a_value)[0][0]
        plt.text(Lr_values[first_idx] , Kr_values[first_idx] +0.2, f'Critical defect size = {first_a_value - 0.01:.2f}mm',
                 fontsize=16, color='black', ha='left', va='bottom',
                 bbox=dict(facecolor='white', alpha=0.5))

    plt.title("FAD", fontsize=20)  # 设置标题字体大小
    plt.xlabel("Lr", fontsize=16)  # 设置 X 轴标签字体大小
    plt.ylabel("Kr", fontsize=16)  # 设置 Y 轴标签字体大小
    plt.xlim(0, 1.5)
    plt.ylim(0, 2)
    
    # 设置刻度字体大小
    plt.tick_params(axis='both', labelsize=14)
    
    plt.legend()
    plt.grid(True)


    # 在 Streamlit 中显示图形
    st.pyplot(plt)

    # 将字典的项转换为列表以创建 DataFrame
    params_df = pd.DataFrame(params.items(), columns=["参数", "值"])
