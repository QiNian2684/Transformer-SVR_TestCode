import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_scatter(csv_file):
    try:
        # 读取CSV文件
        data = pd.read_csv(csv_file)

        # 检查所需的列是否存在
        required_columns = ['LONGITUDE', 'LATITUDE', 'FLOOR']
        if not all(column in data.columns for column in required_columns):
            raise ValueError(f"CSV文件中缺少必要的列：{required_columns}")

        # 提取三列数据
        longitude = data['LONGITUDE']
        latitude = data['LATITUDE']
        floor = data['FLOOR']

        # 打印坐标范围以帮助理解
        print(f"Longitude 范围: {longitude.min()} 到 {longitude.max()}")
        print(f"Latitude 范围: {latitude.min()} 到 {latitude.max()}")
        print(f"Floor 范围: {floor.min()} 到 {floor.max()}")

        # 创建3D图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 根据楼层设置颜色
        scatter = ax.scatter(longitude, latitude, floor, c=floor, cmap='viridis', marker='o', alpha=0.6)

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Floor')

        # 设置轴标签
        ax.set_xlabel('Longitude (单位)')
        ax.set_ylabel('Latitude (单位)')
        ax.set_zlabel('Floor')

        # 设置标题
        ax.set_title('3D Scatter Plot of LONGITUDE, LATITUDE, and FLOOR')

        # 显示图形
        plt.show()

    except FileNotFoundError:
        print(f"文件未找到：{csv_file}")
    except pd.errors.EmptyDataError:
        print("CSV文件是空的或格式不正确。")
    except Exception as e:
        print(f"发生错误：{e}")

if __name__ == "__main__":
    csv_path = 'UJIndoorLoc/trainingData.csv'  # 替换为你的CSV文件路径
    plot_3d_scatter(csv_path)
