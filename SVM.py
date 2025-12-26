import pandas as pd
import numpy as np
import os
import warnings
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from scipy import stats
import shap  # 导入SHAP库
import seaborn as sns  # 新增seaborn依赖
from matplotlib.colors import LinearSegmentedColormap  # 新增颜色映射依赖

# 忽略警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 设置工作目录
os.chdir("/home/syy/syy-model/swnt-protein-corona-ML-main")


# 计算置信区间的函数
def confidence_interval(data, confidence=0.95):
    if len(data) <= 1:
        return np.nan
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


# 读取数据
def load_data():
    plasma_data = pd.read_excel("data/gt15_plasma_features_names_biopy_gravy.xlsx", header=0, index_col=0)
    csf_data = pd.read_excel("data/gt15_csf_features_names_biopy_gravy.xlsx", header=0, index_col=0)

    # 提取特征和标签
    features_plasma = plasma_data.drop(['Corona'], axis=1)
    names_plasma = plasma_data['Corona'].copy()
    features_csf = csf_data.drop(['Corona'], axis=1)
    names_csf = csf_data['Corona'].copy()

    # 合并数据
    features_plasma_labeled = features_plasma.copy()
    features_csf_labeled = features_csf.copy()
    features_plasma_labeled['phase_plasma'] = 1
    features_csf_labeled['phase_plasma'] = 0

    features_merged = pd.concat([features_plasma_labeled, features_csf_labeled], ignore_index=True)
    names_merged = pd.concat([names_plasma, names_csf], ignore_index=True)

    # 不包含phase信息的特征集
    features_merged_naive = features_merged.drop(['phase_plasma'], axis=1)

    # 读取测试数据
    features_test = pd.read_excel("data/netsurfp_2_proteins_selected_for_testing_processed_updated.xlsx",
                                  header=0, index_col=0)

    # 统一列名结构
    if 'entry' in features_test.columns and 'Protein names' not in features_test.columns:
        features_test = features_test.rename(columns={'entry': 'Protein names'})

    # 确保所有必需的列存在
    required_columns = ['Protein names', 'mass']
    for col in required_columns:
        if col not in features_test.columns:
            features_test[col] = 0  # 添加缺失列并填充默认值

    return features_merged_naive, names_merged, features_test


# 数据预处理
def preprocess_data(features_merged_naive, names_merged, features_test):
    # 保存蛋白质名称列
    protein_names = features_test['Protein names']

    # 提取训练特征并填充缺失值
    train_features = features_merged_naive.copy()
    train_features = train_features.fillna(0)

    # 保存训练数据的列名
    train_columns = train_features.columns.tolist()

    # 删除非特征列
    train_features = train_features.drop(['Protein names', 'mass'], axis=1)

    # 标准化（SVM对特征缩放更敏感，使用StandardScaler）
    scaler = StandardScaler()
    scaled_train = pd.DataFrame(scaler.fit_transform(train_features), columns=train_features.columns)

    # 预处理测试数据
    test_features = features_test.copy()
    test_features = test_features.fillna(0)

    # 确保测试数据包含所有训练数据的列
    test_features_full = pd.DataFrame(columns=train_columns)
    for col in test_features.columns:
        if col in train_columns:
            test_features_full[col] = test_features[col]
    for col in train_columns:
        if col not in test_features.columns:
            test_features_full[col] = 0
    test_features_full = test_features_full[train_columns]

    # 删除非特征列
    test_features_full = test_features_full.drop(['Protein names', 'mass'], axis=1)

    # 标准化测试数据
    scaled_test = pd.DataFrame(scaler.transform(test_features_full), columns=test_features_full.columns)

    return scaled_train, names_merged, scaled_test, protein_names, train_features.columns, scaler


# SHAP分析与可视化（适用于SVM模型）
def shap_analysis(model, X_data, feature_names, selector, sample_size=100, save_dir='shap_plots_svm'):
    """使用训练好的SVM模型进行SHAP分析并绘图"""
    os.makedirs(save_dir, exist_ok=True)

    # 应用特征选择器
    X_selected = selector.transform(X_data)

    # 获取被选择的特征名称
    mask = selector.get_support()
    selected_feature_names = [feature_names[i] for i, val in enumerate(mask) if val]

    # 采样减少计算量
    X_sample = shap.sample(X_selected, min(sample_size, len(X_selected)))

    # 创建Kernel SHAP解释器（适用于SVM等非树模型）
    print("正在计算SHAP值，这可能需要一些时间...")
    explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_sample, 10))
    shap_values = explainer.shap_values(X_sample, nsamples=100)

    # 对于二分类问题，只取阳性类别的SHAP值
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # 1. 汇总图（特征重要性排序）
    plt.figure(figsize=(12, 8))
    try:
        shap.summary_plot(shap_values, X_sample, feature_names=selected_feature_names, plot_type="dot")
    except AssertionError:
        print("dot图不支持当前SHAP值格式，自动切换为bar图")
        shap.summary_plot(shap_values, X_sample, feature_names=selected_feature_names, plot_type="bar")
    plt.savefig(f"{save_dir}/shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP汇总图已保存至 {save_dir}/shap_summary.png")

    # 2. 前5重要特征的依赖图
    feat_importance = np.abs(shap_values).mean(axis=0)
    top5_idx = np.argsort(feat_importance)[-5:][::-1]

    for idx in top5_idx:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            idx, shap_values, X_sample,
            feature_names=selected_feature_names,
            show=False
        )
        plt.savefig(f"{save_dir}/shap_dependence_{selected_feature_names[idx]}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"特征 '{selected_feature_names[idx]}' 的依赖图已保存")

    # 3. 典型样本的力场图
    preds = model.predict_proba(X_sample)[:, 1]
    mid_idx = np.argmin(np.abs(preds - 0.5))  # 最接近0.5的样本

    # 生成力场图
    try:
        shap_plot = shap.force_plot(
            explainer.expected_value[1],  # 阳性类别的基础值
            shap_values[mid_idx],
            X_sample[mid_idx],
            feature_names=selected_feature_names,
            matplotlib=True,
            show=False
        )
        plt.savefig(f"{save_dir}/shap_force_sample_{mid_idx}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"样本 {mid_idx} 的力场图已保存")
    except Exception as e:
        print(f"生成力场图时出错: {str(e)}")
        print("跳过力场图生成")

    # 4. 保存特征重要性排名
    importance_df = pd.DataFrame({
        'Feature': selected_feature_names,
        'SHAP Importance': np.abs(shap_values).mean(axis=0)
    })
    importance_df = importance_df.sort_values('SHAP Importance', ascending=False)
    importance_df.to_excel(f"{save_dir}/feature_importance.xlsx", index=False)
    print(f"特征重要性排名已保存至 {save_dir}/feature_importance.xlsx")

    return shap_values, importance_df


# 模型训练和评估（使用SVM）
def train_and_evaluate_model(X_train, y_train, X_test, protein_names, n_splits=100, k_features=38):
    predictions = pd.DataFrame()
    metrics_frame = pd.DataFrame()
    feature_importance = pd.DataFrame()  # SVM没有内置特征重要性，这里用于保持结构一致

    # 创建特征选择器
    selector = SelectKBest(f_classif, k=k_features)

    # 初始化StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=2016)
    splits = list(sss.split(X_train, y_train))

    i = 0
    # 使用tqdm添加进度条
    for train_index, test_index in tqdm(splits, total=n_splits, desc="训练进度"):
        X_train_split, X_test_split = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_split, y_test_split = y_train.iloc[train_index], y_train.iloc[test_index]

        # 特征选择
        X_train_selected = selector.fit_transform(X_train_split, y_train_split)
        X_test_selected = selector.transform(X_test_split)
        X_predict_selected = selector.transform(X_test)

        # 创建并训练SVM模型
        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=2016,
            max_iter=10000
        )

        # 过采样
        sme = SMOTE(random_state=2016, sampling_strategy=0.7, k_neighbors=12)
        X_train_oversampled, y_train_oversampled = sme.fit_resample(X_train_selected, y_train_split)
        svm.fit(X_train_oversampled, y_train_oversampled)

        # 评估模型
        y_pred = svm.predict(X_test_selected)
        y_prob = svm.predict_proba(X_test_selected)[:, 1]

        # 计算指标
        metrics_dict = {
            'AUC': metrics.roc_auc_score(y_test_split, y_prob),
            'Accuracy': svm.score(X_test_selected, y_test_split),
            'Recall': recall_score(y_test_split, y_pred),
            'Precision': precision_score(y_test_split, y_pred),
            'F1': f1_score(y_test_split, y_pred)
        }

        # 保存指标
        metrics_df = pd.DataFrame(metrics_dict, index=[0])
        metrics_df['Round'] = i
        metrics_frame = pd.concat([metrics_frame, metrics_df], ignore_index=True)

        # 保存预测结果
        pred_df = pd.DataFrame()
        pred_df['Protein Name'] = protein_names
        pred_df['In Corona Probability'] = svm.predict_proba(X_predict_selected)[:, 1]
        pred_df['Round'] = i
        pred_df['Test Accuracy'] = metrics_dict['Accuracy']
        predictions = pd.concat([predictions, pred_df], ignore_index=True)

        # SVM没有内置特征重要性，使用SHAP值作为特征重要性
        if i == 0:  # 只在第一轮计算，节省时间
            print("计算特征重要性（仅第一轮）...")
            shap_explainer = shap.KernelExplainer(svm.predict_proba, shap.kmeans(X_train_selected, 10))
            shap_vals = shap_explainer.shap_values(X_train_selected[:50], nsamples=100)  # 采样以加速

            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # 取阳性类别的SHAP值

            # 计算特征重要性（SHAP值的绝对值平均）
            feat_importance = np.abs(shap_vals).mean(axis=0)

            # 保存特征重要性
            feat_imp_df = pd.DataFrame(feat_importance,
                                       index=[f"Feature_{j}" for j in range(X_train_selected.shape[1])]).transpose()
            feat_imp_df['Round'] = i
            feature_importance = pd.concat([feature_importance, feat_imp_df], ignore_index=True)
        else:
            # 后续轮次使用第一轮的特征重要性（节省计算时间）
            feat_imp_df = feature_importance.iloc[0:1].copy()
            feat_imp_df['Round'] = i
            feature_importance = pd.concat([feature_importance, feat_imp_df], ignore_index=True)

        i += 1

    return predictions, metrics_frame, feature_importance, selector


# 主函数
def main():
    print("加载数据...")
    features_merged_naive, names_merged, features_test = load_data()

    print("预处理数据...")
    X_train, y_train, X_test, protein_names, feature_names, scaler = preprocess_data(
        features_merged_naive, names_merged, features_test
    )

    print("开始训练SVM模型...")
    predictions, metrics_frame, feature_importance, selector = train_and_evaluate_model(
        X_train, y_train, X_test, protein_names
    )

    # 计算平均指标
    avg_metrics = metrics_frame.mean()
    print("\n===== 模型评估指标 =====")
    print(f"AUC: {avg_metrics['AUC']:.4f}")
    print(f"准确率 (Accuracy): {avg_metrics['Accuracy']:.4f}")
    print(f"召回率 (Recall): {avg_metrics['Recall']:.4f}")
    print(f"精确率 (Precision): {avg_metrics['Precision']:.4f}")
    print(f"F1分数: {avg_metrics['F1']:.4f}")
    print("=======================")

    # 蛋白质预测平均值和置信区间
    unique_proteins = predictions['Protein Name'].unique()
    protein_avg_predictions = pd.DataFrame()
    for protein in unique_proteins:
        protein_data = predictions[predictions['Protein Name'] == protein]
        avg_prob = protein_data['In Corona Probability'].mean()
        ci = confidence_interval(protein_data['In Corona Probability'])
        protein_row = pd.DataFrame({
            'Protein Name': [protein],
            'Average In Corona Probability': [round(avg_prob, 3)],
            '95 Percent Confidence Interval': [round(ci, 3)]
        })
        protein_avg_predictions = pd.concat([protein_avg_predictions, protein_row], ignore_index=True)

    # 训练最终SVM模型
    print("训练最终SVM模型...")
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    svm_final = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=2016,
        max_iter=10000
    )
    sme_final = SMOTE(random_state=2016, sampling_strategy=0.7, k_neighbors=12)
    X_train_oversampled, y_train_oversampled = sme_final.fit_resample(X_train_selected, y_train)
    svm_final.fit(X_train_oversampled, y_train_oversampled)

    # 执行SHAP分析
    print("\n===== 开始SHAP特征重要性分析 =====")
    shap_values, importance_df = shap_analysis(
        model=svm_final,
        X_data=X_train,
        feature_names=feature_names,
        selector=selector,
        sample_size=100,
        save_dir='shap_plots_svm'
    )

    # 打印前10个重要特征
    print("\n===== 前10个重要特征 =====")
    print(importance_df.head(10))

    # ----------------------
    # 新增：绘制**全部选中特征**的热图（粉蓝渐变配色，去除数值）
    # ----------------------
    print("\n===== 绘制全部选中特征的热图 =====")
    if not importance_df.empty:
        # 获取选中的特征名称列表（来自selector）
        selected_feature_names = [feature_names[i] for i, val in enumerate(selector.get_support()) if val]
        if not selected_feature_names:
            print("错误：特征选择后的特征列表为空，无法绘制热图")
        else:
            # 提取所有选中特征的数据（特征选择后的训练集）
            X_all = X_train_selected  # 已筛选后的矩阵，列即为selected_feature_names

            # 计算皮尔逊相关系数矩阵
            corr_matrix = np.corrcoef(X_all, rowvar=False)

            # 定义自定义颜色并创建colormap（粉蓝渐变）
            custom_colors = ["#82d8ce", "#d6f0ed", "#eaf9f6", "#ffceeb", "#fdc7c7", "#fd9c99"]
            cmap = LinearSegmentedColormap.from_list("custom_heatmap", custom_colors)

            # 绘制热图（上三角掩码 + 隐藏数值 + 自定义配色，适配多特征）
            plt.figure(figsize=(15, 12))  # 增大画布尺寸，避免特征名称重叠
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 隐藏下三角
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=False,  # 去除中间数值显示
                cmap=cmap,  # 使用自定义颜色映射
                xticklabels=selected_feature_names,
                yticklabels=selected_feature_names,
                square=True  # 方形单元格
            )
            plt.title("All Selected Features Correlation Heatmap", fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=8)  # 减小x轴标签字体，避免拥挤
            plt.yticks(rotation=0, fontsize=8)  # 减小y轴标签字体
            plt.tight_layout()  # 自动调整布局，避免标签截断
            plt.savefig("shap_plots_svm/all_selected_features_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("全部选中特征的热图已保存至 shap_plots_svm/all_selected_features_heatmap.png")
    else:
        print("警告：未获取到特征重要性数据，无法绘制热图")

    # 最终预测
    final_predictions = pd.DataFrame()
    final_predictions['Protein Name'] = protein_names
    final_predictions['In Corona Probability'] = svm_final.predict_proba(X_test_selected)[:, 1]
    final_predictions['In Corona'] = final_predictions['In Corona Probability'] >= 0.5

    print(f"预测为阳性的蛋白质数量: {final_predictions['In Corona'].sum()}")

    # 保存结果
    print("保存结果...")
    with pd.ExcelWriter('2021_08_08_predictions_svm_gt15_with_shap.xlsx') as writer:
        predictions.to_excel(writer, sheet_name='Round Based Prediction', index=False)
        metrics_frame.to_excel(writer, sheet_name='Classifier Round Metrics', index=False)
        protein_avg_predictions.to_excel(writer, sheet_name='Protein Average Predictions', index=False)
        final_predictions.to_excel(writer, sheet_name='Final Predictions', index=False)
        if not importance_df.empty:
            importance_df.to_excel(writer, sheet_name='SHAP Feature Importance', index=False)

    print("完成!")


if __name__ == "__main__":
    main()