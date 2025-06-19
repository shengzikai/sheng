import streamlit as st
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import io

# --- 核心分析函数 (从桌面版迁移) ---
# 这些函数是纯计算逻辑，无需修改

def apply_lowpass_filter(signal, sampling_rate, cutoff_freq=10000, order=4):
    """应用巴特沃斯低通滤波器"""
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def apply_savgol_filter(signal, window_length=51, polyorder=3):
    """应用Savitzky-Golay平滑滤波器"""
    if len(signal) <= window_length:
        return signal # 如果信号太短，不滤波
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(signal, window_length, polyorder)

def perform_fft(signal, t):
    n_points = len(signal)
    if n_points < 2: return np.array([]), np.array([])
    sampling_rate = 1 / (t[1] - t[0]) if (t[1] > t[0]) else 1
    yf = fft(signal)
    xf = fftfreq(n_points, 1 / sampling_rate)
    mask = xf >= 0
    return xf[mask], (2.0/n_points) * np.abs(yf[mask])

def double_exponential_decay(t, a1, tau1, a2, tau2, c):
    """双指数衰减模型"""
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + c

def calculate_time_constants(t, eddy_signal):
    """使用双指数模型计算时间常数"""
    if len(t) < 5: return None, None
    t_shifted = t - t[0]
    try:
        initial_guess = (eddy_signal[0]*0.7, 0.001, eddy_signal[0]*0.3, 0.0001, eddy_signal[-1])
        bounds = ([-np.inf, 1e-6, -np.inf, 1e-7, -np.inf], [np.inf, 1, np.inf, 1, np.inf])
        params, _ = curve_fit(double_exponential_decay, t_shifted, eddy_signal, p0=initial_guess, bounds=bounds, maxfev=10000)
        if params[1] < params[3]:
            params = [params[2], params[3], params[0], params[1], params[4]]
        fit_curve = double_exponential_decay(t_shifted, *params)
        return params, fit_curve
    except (RuntimeError, ValueError):
        return None, None

def find_gradient_end_time(gradient_signal, t):
    non_zero_indices = np.where(gradient_signal > 0.01 * np.max(gradient_signal))[0]
    if len(non_zero_indices) == 0: return t[int(len(t)/2)]
    last_high_index = non_zero_indices[-1]
    time_step = t[1] - t[0] if len(t) > 1 else 0
    buffer_points = int(0.0001 / time_step) if time_step > 0 else 2
    buffer_index = min(len(t) - 1, last_high_index + buffer_points)
    return t[buffer_index]

# --- Streamlit Web应用界面 ---

# 设置页面标题和布局
st.set_page_config(layout="wide", page_title="网页版涡流信号分析工具")

# 设置matplotlib字体以支持中文
# 注意：'WenQuanYi Zen Hei' 是我们通过 packages.txt 在 Streamlit Cloud 上安装的字体。
# 在本地Mac上运行时，如果未安装此字体，中文可能同样显示为方框。
# 为了在服务器和本地都能良好显示，您可以在本地也安装"文泉驿正黑"字体。
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

st.title("高级涡流信号分析工具 (网页版)")
st.write("上传包含`time_s`, `gradient_signal`, `mixed_signal`列的CSV文件以开始分析。")

# --- 侧边栏：文件上传和分析控制 ---
with st.sidebar:
    st.header("1. 加载数据")
    uploaded_file = st.file_uploader("选择CSV文件", type="csv")
    
    st.header("2. 分析选项")
    filter_type = st.selectbox(
        "数字滤波器",
        ("无", "低通滤波器", "Savitzky-Golay平滑")
    )
    
    roi_auto = st.checkbox("自动选择拟合范围", value=True)
    
    col1, col2 = st.columns(2)
    with col1:
        roi_start_ms = st.number_input("开始时间 (ms)", disabled=roi_auto, value=0.0)
    with col2:
        roi_end_ms = st.number_input("结束时间 (ms)", disabled=roi_auto, value=0.0)

    analyze_button = st.button("开始分析", type="primary", use_container_width=True, disabled=(uploaded_file is None))

# --- 主界面：结果显示 ---

# 初始化Session State来存储数据
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# 主分析逻辑
if analyze_button and uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = {'time_s', 'gradient_signal', 'mixed_signal'}
        if not required_cols.issubset(df.columns):
            st.error(f"加载失败: CSV需包含 'time_s', 'gradient_signal', 'mixed_signal' 列。")
        else:
            cache = {}
            cache['t'] = df['time_s'].values
            cache['gradient_v'] = df['gradient_signal'].values
            cache['mixed_v'] = df['mixed_signal'].values
            
            # 核心处理流程
            t = cache['t']
            
            # 1. 涡流电压与基线校正
            eddy_v = cache['mixed_v'] - cache['gradient_v']
            gradient_start_indices = np.where(cache['gradient_v'] > 0.01 * np.max(cache['gradient_v']))[0]
            if len(gradient_start_indices) > 0 and gradient_start_indices[0] > 20:
                baseline_end_index = gradient_start_indices[0]
                offset = np.mean(eddy_v[:baseline_end_index])
                corrected_eddy_v = eddy_v - offset
                baseline_info = f"是 (offset={offset:.4f}V, 在 t < {t[baseline_end_index]:.4f}s 区域计算)"
            else:
                corrected_eddy_v = eddy_v
                baseline_info = "否 (未找到清晰的基线区域)"
            cache['eddy_v'] = corrected_eddy_v

            # 2. 积分
            cache['eddy_i'] = cumulative_trapezoid(corrected_eddy_v, t, initial=0)
            
            # 3. 滤波
            sampling_rate = 1 / (t[1] - t[0])
            if filter_type == "低通滤波器":
                cache['eddy_i_filtered'] = apply_lowpass_filter(cache['eddy_i'], sampling_rate)
            elif filter_type == "Savitzky-Golay平滑":
                cache['eddy_i_filtered'] = apply_savgol_filter(cache['eddy_i'])
            else:
                cache['eddy_i_filtered'] = cache['eddy_i']
            
            # 4. ROI与拟合
            if roi_auto:
                g_end_time = find_gradient_end_time(cache['gradient_v'], t)
                start_time = g_end_time
                end_time = t[-1]
            else:
                start_time = roi_start_ms / 1000.0
                end_time = roi_end_ms / 1000.0 if roi_end_ms > roi_start_ms else t[-1]
            
            roi_mask = (t >= start_time) & (t <= end_time)
            params, fit_curve = calculate_time_constants(t[roi_mask], cache['eddy_i_filtered'][roi_mask])
            
            # **最终修复** 对电流信号进行最终的直流偏置校正
            if params is not None:
                final_offset = params[4]
                cache['eddy_i_final'] = cache['eddy_i_filtered'] - final_offset
                fit_curve = fit_curve - final_offset
                final_offset_info = f"是 (C={final_offset:.4f})"
            else:
                cache['eddy_i_final'] = cache['eddy_i_filtered']
                final_offset_info = "否 (拟合失败)"
                
            cache['fit_results'] = {'t': t[roi_mask], 'curve': fit_curve, 'params': params}

            # 5. FFT
            xf, yf = perform_fft(cache['eddy_i_final'], t)
            cache['fft_results'] = {'x': xf, 'y': yf}
            
            # 存储结果和元数据
            st.session_state.data_cache = cache
            st.session_state.analysis_done = True
            st.session_state.baseline_info = baseline_info
            st.session_state.final_offset_info = final_offset_info
            st.session_state.roi_info = f"({start_time*1000:.2f}ms - {end_time*1000:.2f}ms)"

    except Exception as e:
        st.error(f"分析过程中出现错误: {e}")
        st.session_state.analysis_done = False

# 如果分析完成，显示结果
if st.session_state.analysis_done:
    cache = st.session_state.data_cache
    t = cache['t']

    # --- 结果展示区 ---
    st.header("分析结果")

    # 1. 结果摘要
    with st.expander("显示/隐藏量化结果", expanded=True):
        results_md = f"""
        - **基线校正 (电压)**: {st.session_state.baseline_info}
        - **最终偏移校正 (电流)**: {st.session_state.final_offset_info}
        - **滤波器类型**: {filter_type}
        - **拟合时间范围 (ROI)**: {st.session_state.roi_info}
        """
        st.markdown(results_md)

        if cache['fit_results']['params'] is not None:
            params = cache['fit_results']['params']
            fit_text = (f"**双指数拟合结果:**\n"
                        f"- τ1 (慢) = {params[1]*1000:.4f} ms (A1={params[0]:.4f})\n"
                        f"- τ2 (快) = {params[3]*1000:.4f} ms (A2={params[2]:.4f})\n"
                        f"- 电流基线 C = {params[4]:.4f}")
            st.markdown(fit_text)
        else:
            st.warning("时间常数拟合失败。")
            
    # 2. 图表Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["输入信号", "涡流电压", "涡流电流", "时间常数拟合", "FFT频谱"])

    with tab1:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(t, cache['gradient_v'], label='梯度信号 (gradient_signal)')
        ax.plot(t, cache['mixed_v'], label='混合信号 (mixed_signal)', alpha=0.7)
        ax.set_title("A. 输入信号")
        ax.set_xlabel("时间 (s)"); ax.set_ylabel("电压 (V)"); ax.legend(); ax.grid(True)
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(t, cache['eddy_v'])
        ax.set_title("B. 提取的涡流电压 (已做基线校正)")
        ax.set_xlabel("时间 (s)"); ax.set_ylabel("电压 (V)"); ax.grid(True)
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(t, cache['eddy_i'], label='原始积分电流')
        ax.plot(t, cache['eddy_i_filtered'], label=f'滤波后电流 ({filter_type})', alpha=0.8)
        ax.set_title("C. 积分后的涡流电流")
        ax.set_xlabel("时间 (s)"); ax.set_ylabel("电流 (a.u.)"); ax.legend(); ax.grid(True)
        st.pyplot(fig)

    with tab4:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(t, cache['eddy_i_final'], 'b-', label='处理后电流 (已校正偏移)', alpha=0.8)
        
        roi_mask = cache['fit_results']['t']
        roi_indices = np.where((t >= roi_mask[0]) & (t <= roi_mask[-1]))[0] if len(roi_mask) > 0 else []

        ax.plot(t[roi_indices], cache['eddy_i_final'][roi_indices], 'c.', markersize=3, label='拟合区域数据点')
        
        if cache['fit_results']['params'] is not None:
            ax.plot(cache['fit_results']['t'], cache['fit_results']['curve'], 'k--', linewidth=2, label='双指数拟合曲线')
        
        ax.set_title("D. 时间常数拟合")
        ax.set_xlabel("时间 (s)"); ax.set_ylabel("电流 (a.u.)"); ax.legend(); ax.grid(True)
        st.pyplot(fig)

    with tab5:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(cache['fft_results']['x'], cache['fft_results']['y'])
        ax.set_title("E. 最终涡流电流的FFT频谱")
        ax.set_xlabel("频率 (Hz)"); ax.set_ylabel("幅度"); ax.grid(True); ax.set_xlim(0, 20000)
        st.pyplot(fig)

else:
    st.info('请在左侧上传CSV文件并点击"开始分析"以查看结果。') 