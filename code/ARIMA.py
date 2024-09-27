from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf #ACF与PACF
from statsmodels.tsa.arima.model import ARIMA #ARIMA模型
from statsmodels.graphics.api import qqplot  #qq图
from scipy import stats
from docx import Document
import warnings
warnings.filterwarnings("ignore")
import 数据处理
import seaborn as sns

# 绘图设置（适用于mac）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 模型检测
def Model_checking(model) -> None:
    # 残差检验:检验残差是否服从正态分布，画图查看，然后检验

    print('------------残差检验-----------')
    # model.resid：残差 = 实际观测值 – 模型预测值
    print(stats.normaltest(model.resid))

    # QQ图看正态性
    qqplot(model.resid, line="q", fit=True)
    plt.title("Q-Q图")
    plt.savefig('Q-Q图.png',dpi=800)
    # 绘制直方图
    plt.hist(model.resid, bins=50)
    plt.title("直方图")
    plt.savefig('直方图.png',dpi=800)

    # 进行Jarque-Bera检验:判断数据是否符合总体正态分布
    jb_test = sm.stats.stattools.jarque_bera(model.resid)
    print("==================================================")
    print('------------Jarque-Bera检验-----------')
    print('Jarque-Bera test:')
    print('JB:', jb_test[0])
    print('p-value:', jb_test[1])
    print('Skew:', jb_test[2])
    print('Kurtosis:', jb_test[3])

    # 残差序列自相关：残差序列是否独立
    print("==================================================")
    print('------DW检验:残差序列自相关----')
    print(sm.stats.stattools.durbin_watson(model.resid.values))


# 使用BIC矩阵计算p和q的值
def cal_pqValue(D_data, diff_num=0):
    # 定阶
    pmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
    qmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
    bic_matrix = []  # BIC矩阵
    # 差分阶数
    diff_num = 2

    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:
                tmp.append(ARIMA(D_data, order=(p, diff_num, q)).fit().bic)
            except Exception as e:
                print(e)
                tmp.append(None)
        bic_matrix.append(tmp)

    bic_matrix = pd.DataFrame(bic_matrix)  # 从中可以找出最小值

    # 创建 DataFrame
    bic_df = pd.DataFrame(bic_matrix, columns=range(1, 5), index=range(1, 5))
    # 画热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(bic_df, annot=True, fmt=".2f", cmap='YlGnBu', cbar_kws={'label': 'AIC Value'})
    plt.title('AIC Heatmap for ARMA Model Orders')
    plt.xlabel('MA Order')
    plt.ylabel('AR Order')
    plt.savefig('热图.png',dpi=800)

    p, q = bic_matrix.stack().idxmin()  # 先用stack展平，然后用idxmin找出最小值位置。
    print('BIC最小的p值和q值为：%s、%s' % (p, q))
    return p, q


# 计算时序序列模型
def cal_time_series(data, forecast_num):
    # 绘制时序图
    data.plot()
    # 存储图片
    plt.savefig('0阶拆分时序图.png',dpi=800)

    # 绘制自相关图
    plt.figure(figsize=(8,6))
    plot_acf(data)
    plt.title('Autocorrelation', fontsize='20')
    plt.savefig('0阶拆分自相关图.png',dpi=800)
    # 绘制偏自相关图
    plt.figure(figsize=(8, 6))
    plot_pacf(data)
    plt.title('Partial Autocorrelation', fontsize='20')
    plt.savefig('0阶拆分偏自相关图.png',dpi=800)

    # 时序数据平稳性检测
    original_ADF = ADF(data['deal_data'])
    print('原始序列的ADF检验结果为：', original_ADF)

    # 对数序数据进行d阶差分运算，化为平稳时间序列
    diff_num = 0 # 差分阶数
    diff_data = data     # 差分数序数据
    ADF_p_value = ADF(data['deal_data'])[1]
    while  ADF_p_value > 0.01:
        diff_data = diff_data.diff(periods=1).dropna()
        diff_num = diff_num + 1
        # _______________________绘制d阶的相关图————————————————————————
        # 绘制时序图
        diff_data.plot()
        # 存储图片
        plt.savefig(f'{diff_num}阶拆分时序图.png')

        # 绘制自相关图
        plot_acf(data)
        plt.savefig(f'{diff_num}阶拆分自相关图.png')
        # 绘制偏自相关图
        plot_pacf(data)
        plt.savefig(f'{diff_num}阶拆分偏自相关图.png')

        ADF_result = ADF(diff_data['deal_data'])
        ADF_p_value = ADF_result[1]
        print("ADF_p_value:{ADF_p_value}".format(ADF_p_value=ADF_p_value))
        print(u'{diff_num}差分的ADF检验结果为：'.format(diff_num = diff_num), ADF_result )

    # 白噪声检测
    print('差分序列的白噪声检验结果为：', acorr_ljungbox(diff_data, lags=1))  # 返回统计量和p值

    # 使用AIC和BIC准则定阶q和p的值(推荐)
    AIC = sm.tsa.stattools.arma_order_select_ic(diff_data, max_ar=4, max_ma=4, ic='aic')['aic_min_order']
    BIC = sm.tsa.stattools.arma_order_select_ic(diff_data, max_ar=4, max_ma=4, ic='bic')['bic_min_order']
    print('---AIC与BIC准则定阶---')
    print('the AIC is{}\nthe BIC is{}\n'.format(AIC, BIC), end='')
    p = BIC[0]
    q = BIC[1]

    BIC_result = sm.tsa.stattools.arma_order_select_ic(diff_data, max_ar=4, max_ma=4, ic='bic')
    bic_matrix = BIC_result['bic']
    # 创建 DataFrame
    bic_df = pd.DataFrame(bic_matrix, columns=range(1, 5), index=range(1, 5))
    # 画热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(bic_df, annot=True, fmt=".2f", cmap='YlGnBu', cbar_kws={'label': 'AIC Value'})
    plt.title('AIC Heatmap for ARMA Model Orders')
    plt.xlabel('MA Order')
    plt.ylabel('AR Order')
    plt.savefig('热图.png',dpi=800)
    # 使用BIC矩阵来计算q和p的值
    # pq_result = cal_pqValue(diff_data, diff_num)
    # p = pq_result[0]
    # q = pq_result[1]

    # 构建时间序列模型
    model = ARIMA(data, order=(p, diff_num, q)).fit()  # 建立ARIMA(p, diff+num, q)模型
    print('模型报告为：\n', model.summary())
    print("预测结果：\n", model.forecast(forecast_num))

    print("预测结果(详细版)：\n")
    forecast = model.get_forecast(steps=forecast_num)
    table = pd.DataFrame(forecast.summary_frame())
    print(table)

    # 绘制残差图
    diff_data.plot(color='orange', title='残差图')
    model.resid.plot(figsize=(10, 3))
    plt.title("残差图")
    plt.savefig('残差图.png',dpi=800)

    # 模型检查
    Model_checking(model)


if __name__ == '__main__':
    df = 数据处理.data_process()
    #  周期检测
    df.index = pd.date_range(start='2024/01/01', periods=len(df), freq='D')
    decomposition = sm.tsa.seasonal_decompose(df, model='additive', extrapolate_trend='freq')
    plt.rc('figure', figsize=(12, 8))
    fig = decomposition.plot()
    plt.savefig('季节性周期检测.png',dpi=800)
    #  周期检测
    cal_time_series(df, 15) # 模型调用
