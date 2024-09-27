import pandas as pd

# 原始数据:25帧/秒
# 处理后数据:2秒一合并,然后输出至excel
# 以处理编号“11”的视频数据为例，其他视频数据同理

# 视频总帧数和每gap帧进行一次合并
Frame_number = 27322
gap = 50  # 50帧，两秒

# 原始文件名和新文件名
origin_filename = '11origin' + '.xlsx'
filename = 'new11' + '.xlsx'

# 读取原始excel数据至程序中
origin_dataframe = pd.read_excel(io='origin_data/' + origin_filename)
# 取指定列,并存入列表中
origin_all = origin_dataframe['总车流量'].values.tolist()
origin_eme = origin_dataframe['应急车道车流量'].values.tolist()

# 定义新存储列表
new_all = []
new_eme = []
for i in range(0, Frame_number, gap):
    # 合并数据，并将合并后的数据添加至新列表
    new_all.append(sum(origin_all[i:i + gap]))
    new_eme.append(sum(origin_eme[i:i + gap]))

# 输出新数据至excel表格
pd.DataFrame({'总车流量': new_all, '应急车道车流量': new_eme}).to_excel('post_data/' + filename, sheet_name='sheet1', index=False)
