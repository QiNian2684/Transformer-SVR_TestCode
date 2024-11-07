
# 项目README文档

## 项目概述

本项目旨在构建一个用于室内定位的模型，利用WiFi信号强度数据预测用户的经度、纬度和楼层信息。项目使用了Transformer自编码器进行特征提取，并使用支持向量机（SVM）进行回归和分类任务。通过对模型的超参数进行调优，以提高定位精度和分类准确率。

---

## 项目结构

```
├── data_preprocessing.py
├── model_definition.py
├── training_and_evaluation.py
├── main.py
├── hyperparameter_tuning_classification.py
├── hyperparameter_tuning_regression.py
├── UJIndoorLoc/
│   ├── trainingData.csv
│   └── validationData.csv
├── results/
│   ├── classification/
│   └── regression/
└── README.md
```

---

## 文件说明

### 1. `data_preprocessing.py`

**作用：**

- 负责数据的加载和预处理，包括缺失值处理、特征缩放和标签编码等。

**实现逻辑：**

- **数据加载：** 使用`pandas`加载训练和测试数据集。
- **缺失值处理：** 将WiFi信号强度中的缺失值（数值为100）替换为-105，表示极弱的信号。
- **特征缩放：**
  - 对特征数据（WiFi信号强度）使用`MinMaxScaler`进行归一化，将数值缩放到(0, 1)区间。
  - 对目标变量（经度和纬度）使用`StandardScaler`进行标准化。
- **标签编码：** 使用`LabelEncoder`对楼层标签进行编码，将分类变量转换为数字。
- **数据划分：** 将训练数据进一步划分为训练集和验证集。

**返回值：**

- 返回预处理后的训练集、验证集、测试集的数据和标签，以及用于逆转换的缩放器和编码器。

### 2. `model_definition.py`

**作用：**

- 定义了用于特征提取的Transformer自编码器模型`WiFiTransformerAutoencoder`。

**实现逻辑：**

- **编码器部分：**
  - 输入层将输入特征映射到模型维度（`model_dim`）。
  - 使用多层`TransformerEncoder`对输入进行编码。
  - 使用`AdaptiveAvgPool1d`对编码结果进行池化，得到固定长度的特征向量。
- **解码器部分：**
  - 使用全连接层将编码器的输出映射回输入维度，实现重构。
- **激活函数：**
  - 使用`ReLU`作为激活函数。

### 3. `training_and_evaluation.py`

**作用：**

- 包含了模型的训练和评估函数，包括自编码器的训练、特征提取、回归和分类模型的训练与评估。

**实现逻辑：**

- **`train_autoencoder`：**
  - 使用自编码器对输入进行重构，计算重构误差（MSE损失）。
  - 使用Adam优化器进行参数更新，包含梯度裁剪和早停机制。
- **`extract_features`：**
  - 使用训练好的自编码器的编码器部分提取输入数据的特征。
- **`train_and_evaluate_regression_model`：**
  - 使用提取的特征训练多输出回归模型（`SVR`），预测经度和纬度。
  - 计算模型的评估指标（MSE、MAE、R²）和平均误差距离。
  - 生成结果图表并保存到指定目录。
- **`train_and_evaluate_classification_model`：**
  - 使用提取的特征训练分类模型（`SVC`），预测楼层信息。
  - 计算分类模型的准确率和生成分类报告。
  - 生成结果图表并保存到指定目录。

**结果保存：**

- 在项目根目录的`results`文件夹下，分类和回归结果分别保存在`classification`和`regression`文件夹中。
- 每次运行会创建一个以当前时间戳命名的文件夹，图表以四位数编号（如`0001.png`）保存。
- 同时在每次optuna调参结束后立即判断是否为最优模型，如果是则保存模型。将对应json、pkl文件保存在`results`文件夹下。“0000”图片为最优模型的结果图片。

### 4. `main.py`

**作用：**

- 主程序，执行数据加载、模型训练、评估和保存的全过程。

**实现逻辑：**

- 加载并预处理数据。
- 初始化并训练Transformer自编码器模型。
- 提取训练集和测试集的特征。
- 逆标准化目标变量（经度和纬度）。
- 定义`SVR`和`SVC`的超参数。
- 训练并评估回归和分类模型。
- 保存训练好的模型、缩放器和编码器。
- 保存预测结果到`y_pred_final.csv`。

### 5. `hyperparameter_tuning_classification.py`

**作用：**

- 使用`Optuna`进行分类模型的超参数调优。

**实现逻辑：**

- 设置搜索空间，包括Transformer自编码器和`SVC`的超参数。
- 定义目标函数`objective`，在每个试验中：
  - 初始化并训练自编码器。
  - 提取特征并训练分类模型。
  - 评估模型的准确率，作为优化目标。
- 使用`Optuna`的`study`对象进行优化，指定优化方向为最大化准确率。
- 每次试验的结果图表保存到`results/classification/`下以时间戳命名的文件夹中，图片编号为四位数。
- 保存最佳超参数到`best_hyperparameters_classification.json`。


### 6. `hyperparameter_tuning_regression.py`

**作用：**

- 使用`Optuna`进行回归模型的超参数调优。

**实现逻辑：**

- 设置搜索空间，包括Transformer自编码器和`SVR`的超参数。
- 定义目标函数`objective`，在每个试验中：
  - 初始化并训练自编码器。
  - 提取特征并训练回归模型。
  - 评估模型的平均误差距离，作为优化目标。
- 使用`Optuna`的`study`对象进行优化，指定优化方向为最小化平均误差距离。
- 每次试验的结果图表保存到`results/regression/`下以时间戳命名的文件夹中，图片编号为四位数。
- 保存最佳超参数到`best_hyperparameters_regression.json`。

### 7. `UJIndoorLoc/`

**作用：**

- 存放数据集文件，包括`trainingData.csv`和`validationData.csv`。

---

## 最终结果的呈现方式

- **结果保存：**
  - 所有生成的结果图表保存在`results`文件夹中，分类和回归的结果分别保存在`classification`和`regression`子文件夹中。
  - 每次运行超参数调优脚本，会在对应的子文件夹下创建一个以当前时间戳命名的文件夹，用于区分不同的实验结果。
  - 结果图表以四位数编号命名，如`0001.png`、`0002.png`，按照试验顺序递增。

- **图表内容：**
  - **回归模型：**
    - 2D预测误差散点图，展示经度和纬度的预测误差分布。
    - 训练参数和模型评估指标，包括MSE、MAE、R²得分、平均误差距离等。
    - 训练和验证损失曲线，观察模型的收敛情况。
  - **分类模型：**
    - 分类报告，展示精确率、召回率、F1得分等指标。
    - 训练参数和模型评估指标，包括准确率等。
    - 训练和验证损失曲线，观察模型的收敛情况。

- **模型和参数保存：**
  - 训练好的模型（自编码器、回归模型、分类模型）保存为`.pkl`或`.pth`文件，便于后续加载和使用。
  - 最佳超参数保存在`best_hyperparameters_classification.json`和`best_hyperparameters_regression.json`中。

---

## 运行指南

1. **准备数据：**

   - 确保`UJIndoorLoc`文件夹中存在`trainingData.csv`和`validationData.csv`数据文件。

2. **运行主程序：**

   - 直接运行`main.py`，将执行数据加载、模型训练、评估和保存的全过程。

     ```bash
     python main.py
     ```

3. **超参数调优：**

   - **分类模型超参数调优：**

     ```bash
     python hyperparameter_tuning_classification.py
     ```

   - **回归模型超参数调优：**

     ```bash
     python hyperparameter_tuning_regression.py
     ```

   - 调优过程中会自动保存结果图表和最佳超参数配置。

4. **查看结果：**

   - 进入`results`文件夹，选择对应的实验日期文件夹，查看保存的结果图表和超参数配置。

---

## 注意事项

- **计算资源：**

  - 超参数调优过程可能需要大量的计算资源和时间，建议在有GPU加速的环境下运行，或者适当减少`n_trials`和`epochs`的数量。

- **环境依赖：**

  - 需要安装以下主要的Python库：

    - `pandas`
    - `numpy`
    - `scikit-learn`
    - `torch`
    - `optuna`
    - `matplotlib`
    - `joblib`

- **随机种子：**

  - 在超参数调优脚本中设置了随机种子，以保证实验的可重复性。

---

## 结论

本项目通过使用Transformer自编码器提取WiFi信号强度的特征，并结合支持向量机进行回归和分类任务，实现了室内定位的功能。通过超参数调优，模型的性能得到了提升。结果以图表和模型文件的形式保存，便于后续的分析和应用。
