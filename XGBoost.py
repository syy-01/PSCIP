import pandas as pd
import numpy as np
import os
import warnings
from tqdm import tqdm
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from scipy import stats

# 解决numba与numpy冲突（在导入shap前执行）
import numpy as np

if not hasattr(np, 'long'):
    np.long = np.int64  # 为高版本numpy添加long别名，兼容numba
import shap  # 导入SHAP库

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


# SHAP分析与可视化（仅显示前20个特征，颜色适配示例图）
def shap_analysis(model, X_data, feature_names, selector, sample_size=100, save_dir='shap_plots_xgb'):
    """使用训练好的XGBoost模型进行SHAP分析，条形图仅显示前20个特征"""
    os.makedirs(save_dir, exist_ok=True)

    # 应用特征选择器（与模型训练保持一致）
    X_selected = selector.transform(X_data)

    # 获取被选择的特征名称
    mask = selector.get_support()
    selected_feature_names = [feature_names[i] for i, val in enumerate(mask) if val]

    # 采样减少计算量
    X_sample = shap.sample(X_selected, min(sample_size, len(X_selected)))

    # 创建解释器（XGBoost专用）
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # 处理多输出情况，确保只使用阳性类别的SHAP值
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # 取阳性类别的SHAP值
        print("检测到多输出SHAP值，已自动转换为阳性类别的SHAP值")

    # 1. SHAP汇总图（特征重要性排序）
    plt.figure(figsize=(12, 8))
    try:
        shap.summary_plot(shap_values, X_sample, feature_names=selected_feature_names, plot_type="dot")
    except AssertionError:
        print("dot图不支持当前SHAP值格式，自动切换为bar图")
        shap.summary_plot(shap_values, X_sample, feature_names=selected_feature_names, plot_type="bar")
    plt.savefig(f"{save_dir}/shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP汇总图已保存至 {save_dir}/shap_summary.png")

    # 2. 特征重要性条形图（仅显示前20个特征，颜色与示例图一致）
    feat_importance = np.abs(shap_values).mean(axis=0)  # 平均绝对SHAP值（衡量特征重要性）
    # 构建特征-重要性DataFrame，仅保留前20个特征
    feat_df = pd.DataFrame({
        'Feature': selected_feature_names,
        'Mean Absolute SHAP Value': feat_importance
    }).sort_values(by='Mean Absolute SHAP Value', ascending=False).head(20)  # 关键：仅取前20个

    # 绘制水平条形图（调整尺寸适配20个特征，颜色为steelblue）
    plt.figure(figsize=(8, 12))  # 高度适配20个特征，宽度保持8
    sns.barplot(
        x='Mean Absolute SHAP Value',
        y='Feature',
        data=feat_df,
        color='steelblue'  # 与示例图颜色一致
    )
    plt.xlabel('mean(|SHAP value|) (average impact on model output mag)')  # 与示例图一致的x轴标签
    plt.ylabel('Feature')
    plt.title('Top 20 Feature Importance by Mean Absolute SHAP Value')  # 明确标注“前20个”
    plt.tight_layout()  # 避免标签截断
    plt.savefig(f"{save_dir}/shap_feature_importance_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"前20个特征的重要性条形图已保存至 {save_dir}/shap_feature_importance_bar.png")

    # 3. 前5重要特征的依赖图（保留原始依赖图）
    top_indices = np.argsort(feat_importance)[-min(5, len(feat_importance)):][::-1]
    for idx in top_indices:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            idx, shap_values, X_sample,
            feature_names=selected_feature_names,
            show=False
        )
        plt.savefig(f"{save_dir}/shap_dependence_{selected_feature_names[idx]}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"特征 '{selected_feature_names[idx]}' 的依赖图已保存")

    # 4. 典型样本的力场图（修复参数，适配分类模型）
    try:
        preds = model.predict_proba(X_sample)[:, 1]
    except AttributeError:
        preds = model.predict(X_sample)  # fallback（若模型无predict_proba）

    if len(preds) > 0:
        mid_idx = np.argmin(np.abs(preds - 0.5))  # 最接近0.5概率的样本
        if mid_idx < 0 or mid_idx >= len(preds):
            mid_idx = 0  # 安全 fallback

        # 获取基准值（阳性类的log odds基准）
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)) and len(base_value) == 2:
            base_value = base_value[1]  # 取阳性类（类别1）的基准值

        # 绘制力场图：指定link='logit'，转换为概率影响
        shap_plot = shap.force_plot(
            base_value,
            shap_values[mid_idx],
            X_sample[mid_idx],
            feature_names=selected_feature_names,
            matplotlib=True,
            show=False,
            link='logit'  # 关键：适配分类模型的概率输出
        )
        plt.savefig(f"{save_dir}/shap_force_sample_{mid_idx}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"样本 {mid_idx} 的力场图已保存")
    else:
        print("警告：无法生成预测概率，跳过力场图绘制")

    return shap_values


# 读取数据
def load_data():
    plasma_data = pd.read_excel("data/gt6_plasma_features_names_biopy_gravy.xlsx", header=0, index_col=0)
    csf_data = pd.read_excel("data/gt6_csf_features_names_biopy_gravy.xlsx", header=0, index_col=0)

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
    train_features = train_features.drop(['Protein names', 'mass'], axis=1, errors='ignore')

    # 标准化
    scaler = MinMaxScaler()
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
    test_features_full = test_features_full.drop(['Protein names', 'mass'], axis=1, errors='ignore')

    # 标准化测试数据
    scaled_test = pd.DataFrame(scaler.transform(test_features_full), columns=test_features_full.columns)

    return scaled_train, names_merged, scaled_test, protein_names, train_features.columns


# 模型训练和评估
def train_and_evaluate_model(X_train, y_train, X_test, protein_names, feature_names, n_splits=100, k_features=38):
    predictions = pd.DataFrame()
    metrics_frame = pd.DataFrame()
    feature_importance = pd.DataFrame()

    # 创建特征选择器
    selector = SelectKBest(f_classif, k=k_features)

    # 初始化StratifiedShuffleSplit并转换为可迭代对象
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

        # 创建并训练XGBoost模型
        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=700,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=2016,
            n_jobs=-1
        )

        # 过采样
        sme = SMOTE(random_state=2016, sampling_strategy=0.7, k_neighbors=12)
        X_train_oversampled, y_train_oversampled = sme.fit_resample(X_train_selected, y_train_split)
        xgb_clf.fit(X_train_oversampled, y_train_oversampled)

        # 评估模型
        y_pred = xgb_clf.predict(X_test_selected)
        y_prob = xgb_clf.predict_proba(X_test_selected)[:, 1]

        # 计算指标
        metrics_dict = {
            'AUC': metrics.roc_auc_score(y_test_split, y_prob),
            'Accuracy': xgb_clf.score(X_test_selected, y_test_split),
            'Recall': recall_score(y_test_split, y_pred, zero_division=0),
            'Precision': precision_score(y_test_split, y_pred, zero_division=0),
            'F1': f1_score(y_test_split, y_pred, zero_division=0)
        }

        # 保存指标
        metrics_df = pd.DataFrame(metrics_dict, index=[0])
        metrics_df['Round'] = i
        metrics_frame = pd.concat([metrics_frame, metrics_df], ignore_index=True)

        # 保存预测结果
        pred_df = pd.DataFrame()
        pred_df['Protein Name'] = protein_names
        pred_df['In Corona Probability'] = xgb_clf.predict_proba(X_predict_selected)[:, 1]
        pred_df['Round'] = i
        pred_df['Test Accuracy'] = metrics_dict['Accuracy']
        predictions = pd.concat([predictions, pred_df], ignore_index=True)

        # 保存特征重要性
        feat_imp_df = pd.DataFrame(xgb_clf.feature_importances_,
                                   index=[f"Feature_{j}" for j in range(X_train_selected.shape[1])]).transpose()
        feat_imp_df['Round'] = i
        feature_importance = pd.concat([feature_importance, feat_imp_df], ignore_index=True)

        i += 1

    return predictions, metrics_frame, feature_importance, selector


# 主函数
def main():
    print("加载数据...")
    features_merged_naive, names_merged, features_test = load_data()

    print("预处理数据...")
    X_train, y_train, X_test, protein_names, feature_names = preprocess_data(
        features_merged_naive, names_merged, features_test
    )

    print("开始训练XGBoost模型...")
    predictions, metrics_frame, feature_importance, selector = train_and_evaluate_model(
        X_train, y_train, X_test, protein_names, feature_names
    )

    # 计算平均指标
    avg_metrics = metrics_frame.mean()
    print("\n===== XGBoost模型评估指标 =====")
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

    # 训练最终模型
    print("训练最终XGBoost模型...")
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    xgb_final = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=700,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=2016,
        n_jobs=-1
    )

    sme_final = SMOTE(random_state=2016, sampling_strategy=0.7, k_neighbors=12)
    X_train_oversampled, y_train_oversampled = sme_final.fit_resample(X_train_selected, y_train)
    xgb_final.fit(X_train_oversampled, y_train_oversampled)

    # 执行SHAP分析（仅前20个特征，颜色适配）
    print("\n===== 开始SHAP特征重要性分析 =====")
    shap_analysis(
        model=xgb_final,
        X_data=X_train,
        feature_names=feature_names,
        selector=selector,
        sample_size=100,
        save_dir='shap_plots_xgb'
    )

    # 最终预测
    final_predictions = pd.DataFrame()
    final_predictions['Protein Name'] = protein_names
    final_predictions['In Corona Probability'] = xgb_final.predict_proba(X_test_selected)[:, 1]
    final_predictions['In Corona'] = final_predictions['In Corona Probability'] >= 0.5

    print(f"预测为阳性的蛋白质数量: {final_predictions['In Corona'].sum()}")

    # 保存结果
    print("保存结果...")
    with pd.ExcelWriter('predictions_simplified_gt6_xgb_with_shap.xlsx') as writer:
        predictions.to_excel(writer, sheet_name='Round Based Prediction', index=False)
        metrics_frame.to_excel(writer, sheet_name='Classifier Round Metrics', index=False)
        protein_avg_predictions.to_excel(writer, sheet_name='Protein Average Predictions', index=False)
        final_predictions.to_excel(writer, sheet_name='Final Predictions', index=False)

    print("完成!")


if __name__ == "__main__":
    main()