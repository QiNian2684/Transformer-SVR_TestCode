# save_pca_data.py

import pandas as pd
from data_preprocessing import load_and_preprocess_data


# 调用load_and_preprocess_data函数来加载和预处理数据
def save_pca_processed_data(train_path, test_path):
    try:
        # 加载数据，这里假设PCA组件设置为64（可根据需要调整）
        X_train, y_train, y_train_floor, X_val, y_val, y_val_floor, X_test, y_test, y_test_floor, scaler_X, scaler_y, label_encoder, filtered_test_indices = load_and_preprocess_data(
            train_path, test_path, pca_components=64)

        # 将PCA处理后的数据转换成DataFrame并保存
        train_df = pd.DataFrame(X_train)
        train_df.to_csv('X_train_pca.csv', index=False)
        print("训练集PCA处理后的数据已保存为 X_train_pca.csv")

        val_df = pd.DataFrame(X_val)
        val_df.to_csv('X_val_pca.csv', index=False)
        print("验证集PCA处理后的数据已保存为 X_val_pca.csv")

        test_df = pd.DataFrame(X_test)
        test_df.to_csv('X_test_pca.csv', index=False)
        print("测试集PCA处理后的数据已保存为 X_test_pca.csv")

    except Exception as e:
        print("数据处理或保存时发生错误：", str(e))


# 调用函数保存数据
if __name__ == "__main__":
    # 需要将'train.csv'和'test.csv'替换为实际数据文件的路径
    save_pca_processed_data('UJIndoorLoc/trainingData.csv', 'UJIndoorLoc/validationData.csv')
