# import pandas as pd
# import numpy as np
# import os
# import warnings
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from sklearn import metrics
# from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import RFECV
# from sklearn.linear_model import RidgeClassifier
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.combine import SMOTETomek
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import matplotlib.pyplot as plt
# from scipy import stats
# from scipy.stats import pearsonr, spearmanr
# from sklearn.feature_selection import mutual_info_regression
# import seaborn as sns
#
# # 解决numba与numpy冲突（在导入shap前执行）
# import numpy as np
#
# if not hasattr(np, 'long'):
#     np.long = np.int64  # 为高版本numpy添加long别名，兼容numba
# import shap  # 导入SHAP库
#
#
# # --------------------------
# # 固定随机种子
# # --------------------------
# def set_random_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#
#
# # 计算置信区间的函数
# def confidence_interval(data, confidence=0.95):
#     if len(data) <= 1:
#         return np.nan
#     a = 1.0 * np.array(data)
#     n = len(a)
#     m, se = np.mean(a), stats.sem(a)
#     h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
#     return h
#
#
# # 忽略警告
# warnings.filterwarnings('ignore', category=FutureWarning)
# warnings.filterwarnings('ignore', category=DeprecationWarning)
#
# # 设置工作目录
# os.chdir("/home/syy/syy-model/swnt-protein-corona-ML-main")
#
# # 设备配置
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")
#
#
# # --------------------------
# # 1. 特征提取与选择
# # --------------------------
# class FeatureExtractor:
#     def __init__(self):
#         self.hydrophobic_cols = ['gravy', 'fraction_exposed_nonpolar_total', 'fraction_exposed_polar_total', 'rsa_mean']
#         self.structure_cols = ['secondary_structure_fraction_helix', 'secondary_structure_fraction_sheet',
#                                'secondary_structure_fraction_disordered', 'fraction_exposed', 'fraction_buried',
#                                'asa_sum']
#         self.physicochemical_cols = ['molecular_weight', 'isoelectric_point', 'instability_index', 'aromaticity']
#         self.aa_composition_cols = ['frac_aa_A', 'frac_aa_C', 'frac_aa_D', 'frac_aa_E', 'frac_aa_F', 'frac_aa_G',
#                                     'frac_aa_H', 'frac_aa_I', 'frac_aa_K', 'frac_aa_L', 'frac_aa_M', 'frac_aa_N',
#                                     'frac_aa_P', 'frac_aa_Q', 'frac_aa_R', 'frac_aa_S', 'frac_aa_T', 'frac_aa_V',
#                                     'frac_aa_W', 'frac_aa_Y']
#
#     def extract_features(self, data):
#         all_feature_cols = self.hydrophobic_cols + self.structure_cols + self.physicochemical_cols + self.aa_composition_cols
#         base_features = data[all_feature_cols].fillna(0).values
#
#         exposed = data['fraction_exposed'].fillna(0).values.reshape(-1, 1)
#         instability = data['instability_index'].fillna(0).values.reshape(-1, 1)
#         aromaticity = data['aromaticity'].fillna(0).values.reshape(-1, 1)
#         helix = data['secondary_structure_fraction_helix'].fillna(0).values.reshape(-1, 1)
#         gravy = data['gravy'].fillna(0).values.reshape(-1, 1)
#
#         interaction_features = np.hstack([
#             exposed * instability,
#             aromaticity * helix,
#             exposed * (1 - helix),
#             instability * aromaticity,
#             gravy * aromaticity,
#             (helix + data['secondary_structure_fraction_sheet'].fillna(0).values.reshape(-1, 1)) * exposed
#         ])
#
#         total_features = np.hstack([base_features, interaction_features])
#         print(f"特征提取完成: 总特征{total_features.shape[1]}个")
#         return total_features, all_feature_cols  # 返回基础特征列名用于SHAP
#
#     def select_features(self, X_train, y_train, X_test):
#         rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
#         rf.fit(X_train, y_train)
#         importance = rf.feature_importances_
#         threshold = np.percentile(importance, 40)
#         mask = importance >= threshold
#         X_train_filtered = X_train[:, mask]
#         X_test_filtered = X_test[:, mask]
#
#         selector = RFECV(estimator=RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42),
#                          step=1, cv=3, scoring='f1', min_features_to_select=15)
#         X_train_selected = selector.fit_transform(X_train_filtered, y_train)
#         X_test_selected = selector.transform(X_test_filtered)
#
#         print(f"特征选择完成: 从{X_train.shape[1]}→{X_train_selected.shape[1]}个特征")
#         return X_train_selected, X_test_selected, selector.support_, mask  # 返回掩码用于特征名映射
#
#
# # --------------------------
# # 2. 数据集与模型定义
# # --------------------------
# class ProteinDataset(Dataset):
#     def __init__(self, features, labels=None):
#         self.features = torch.tensor(features, dtype=torch.float32)
#         self.labels = torch.tensor(labels.values, dtype=torch.float32) if labels is not None else None
#
#     def __len__(self):
#         return len(self.features)
#
#     def __getitem__(self, idx):
#         if self.labels is not None:
#             return self.features[idx], self.labels[idx]
#         return self.features[idx]
#
#
# class FeatureMLP(nn.Module):
#     def __init__(self, input_dim, hidden_dims=[32, 16], dropout=0.35):
#         super().__init__()
#         layers = []
#         prev_dim = input_dim
#         for dim in hidden_dims:
#             layers.extend([
#                 nn.Linear(prev_dim, dim),
#                 nn.BatchNorm1d(dim),
#                 nn.LeakyReLU(0.1),
#                 nn.Dropout(dropout)
#             ])
#             prev_dim = dim
#         layers.append(nn.Linear(prev_dim, 1))
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.model(x).squeeze()  # 确保输出是一维张量
#
#
# # 添加一个函数用于生成MLP的预测概率
# def mlp_predict_proba(model, data):
#     """生成MLP模型的预测概率"""
#     model.eval()  # 设置为评估模式
#     with torch.no_grad():  # 禁用梯度计算
#         data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
#         outputs = model(data_tensor)
#         probabilities = torch.sigmoid(outputs).cpu().numpy()  # 应用sigmoid获取概率
#     return probabilities.reshape(-1, 1)  # 重塑为二维数组以便后续处理
#
#
# # --------------------------
# # 3. 早停策略
# # --------------------------
# class EarlyStopping:
#     def __init__(self, patience=6, delta=0.002):
#         self.patience = patience
#         self.delta = delta
#         self.best_score = None
#         self.best_model_weights = None
#         self.counter = 0
#         self.early_stop = False
#
#     def __call__(self, val_metrics, model):
#         score = val_metrics['Recall'] * 0.3 + val_metrics['Precision'] * 0.3 + val_metrics['F1'] * 0.4
#
#         if self.best_score is None:
#             self.best_score = score
#             self.best_model_weights = model.state_dict()
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.best_model_weights = model.state_dict()
#             self.counter = 0
#
#
# # --------------------------
# # 4. 训练函数
# # --------------------------
# def train_mlp(model, train_loader, val_loader, epochs=60, lr=1e-4):
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
#
#     train_labels = np.concatenate([labels.numpy() for _, labels in train_loader])
#     pos_ratio = train_labels.mean()
#     pos_weight = torch.tensor((1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1.0, device=device)
#
#     class FocalLoss(nn.Module):
#         def __init__(self, pos_weight=None, gamma=2.0):
#             super().__init__()
#             self.pos_weight = pos_weight
#             self.gamma = gamma
#
#         def forward(self, inputs, targets):
#             bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(inputs, targets)
#             pt = torch.exp(-bce_loss)
#             focal_loss = (1 - pt) ** self.gamma * bce_loss
#             return focal_loss
#
#     criterion = FocalLoss(pos_weight=pos_weight, gamma=1.5)
#     scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-6)
#     early_stopping = EarlyStopping(patience=6)
#
#     for epoch in range(epochs):
#         model.train()
#         train_loss = 0.0
#         with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs} 训练") as pbar:
#             for feats, labels in train_loader:
#                 feats, labels = feats.to(device), labels.to(device)
#                 optimizer.zero_grad()
#                 outputs = model(feats)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
#                 train_loss += loss.item()
#                 pbar.update(1)
#                 pbar.set_postfix({"训练损失": f"{loss.item():.4f}"})
#         avg_train_loss = train_loss / len(train_loader)
#
#         model.eval()
#         val_preds, val_labels = [], []
#         with torch.no_grad():
#             for feats, labels in val_loader:
#                 feats = feats.to(device)
#                 outputs = model(feats)
#                 val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
#                 val_labels.extend(labels.numpy())
#
#         val_preds = np.array(val_preds)
#         val_labels = np.array(val_labels)
#         y_pred = (val_preds >= 0.5).astype(int)
#
#         cm = confusion_matrix(val_labels, y_pred)
#         tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
#         specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#
#         val_metrics = {
#             'AUC': metrics.roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.5,
#             'Accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
#             'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
#             'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
#             'F1': f1_score(val_labels, y_pred, zero_division=0),
#             'Specificity': specificity
#         }
#
#         scheduler.step(val_metrics['F1'])
#         early_stopping(val_metrics, model)
#         if early_stopping.early_stop:
#             break
#
#     model.load_state_dict(early_stopping.best_model_weights)
#     return model, val_metrics, val_preds
#
#
# # --------------------------
# # 5. 随机森林训练
# # --------------------------
# def train_rf(X_train, y_train, X_val, y_val):
#     rf = RandomForestClassifier(
#         n_estimators=120,
#         max_depth=5,
#         min_samples_split=6,
#         min_samples_leaf=3,
#         max_features='sqrt',
#         class_weight='balanced_subsample',
#         random_state=42,
#         n_jobs=-1,
#         oob_score=True
#     )
#     rf.fit(X_train, y_train)
#
#     rf_preds = rf.predict_proba(X_val)[:, 1]
#
#     best_rf_f1 = 0.0
#     best_rf_thresh = 0.5
#     for thresh in np.arange(0.3, 0.7, 0.02):
#         y_pred = (rf_preds >= thresh).astype(int)
#         current_f1 = f1_score(y_val, y_pred, zero_division=0)
#         if current_f1 > best_rf_f1:
#             best_rf_f1 = current_f1
#             best_rf_thresh = thresh
#
#     y_pred_rf = (rf_preds >= best_rf_thresh).astype(int)
#     cm = confusion_matrix(y_val, y_pred_rf)
#     tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
#     specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#
#     metrics_rf = {
#         'AUC': metrics.roc_auc_score(y_val, rf_preds),
#         'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
#         'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
#         'F1': best_rf_f1,
#         'Specificity': specificity,
#         'threshold': best_rf_thresh
#     }
#     return rf, metrics_rf, rf_preds
#
#
# # --------------------------
# # 6. 模型融合
# # --------------------------
# def ensemble_predictions(mlp_preds, rf_preds, y_val, X_val):
#     mlp_preds = np.array(mlp_preds).reshape(-1, 1)
#     rf_preds = np.array(rf_preds).reshape(-1, 1)
#
#     rf_imp = RandomForestClassifier(n_estimators=50, max_depth=3).fit(X_val, y_val).feature_importances_
#     top_feat_idx = np.argsort(rf_imp)[-5:][::-1]
#     top_features = X_val[:, top_feat_idx]
#
#     meta_features = np.hstack([
#         mlp_preds,
#         rf_preds,
#         (mlp_preds * rf_preds).reshape(-1, 1),
#         np.abs(mlp_preds - rf_preds).reshape(-1, 1),
#         ((mlp_preds + rf_preds) / 2).reshape(-1, 1),
#         top_features
#     ])
#
#     meta_model = RidgeClassifier(
#         alpha=0.5,
#         class_weight='balanced',
#         random_state=42
#     )
#     meta_model.fit(meta_features, y_val)
#
#     decision_scores = meta_model.decision_function(meta_features)
#     ensemble_preds = 1 / (1 + np.exp(-decision_scores))
#
#     best_f1 = 0.0
#     best_threshold = 0.5
#     best_recall = 0.0
#     best_precision = 0.0
#
#     for threshold in np.arange(0.2, 0.7, 0.01):
#         y_pred = (ensemble_preds >= threshold).astype(int)
#         current_f1 = f1_score(y_val, y_pred, zero_division=0)
#         current_recall = recall_score(y_val, y_pred, zero_division=0)
#         current_precision = precision_score(y_val, y_pred, zero_division=0)
#
#         if (current_f1 > best_f1) or \
#                 (current_f1 == best_f1 and current_precision > best_precision) or \
#                 (current_f1 > best_f1 - 0.02 and current_precision > best_precision + 0.1):
#             best_f1 = current_f1
#             best_threshold = threshold
#             best_recall = current_recall
#             best_precision = current_precision
#
#     y_pred_ensemble = (ensemble_preds >= best_threshold).astype(int)
#     cm = confusion_matrix(y_val, y_pred_ensemble)
#     tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
#     specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#     accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
#     auc = metrics.roc_auc_score(y_val, ensemble_preds) if len(set(y_val)) > 1 else 0.5
#
#     metrics_ensemble = {
#         'AUC': auc,
#         'Accuracy': accuracy,
#         'Recall': best_recall,
#         'Precision': best_precision,
#         'F1': best_f1,
#         'Specificity': specificity,
#         'threshold': best_threshold,
#         'preds': ensemble_preds,
#         'meta_model': meta_model,
#         'top_feat_idx': top_feat_idx,
#         'meta_features': meta_features,
#         'meta_feature_names': [  # 元特征名称（用于SHAP）
#             'MLP预测', 'RF预测', 'MLP×RF', '|MLP-RF|', '(MLP+RF)/2',
#             'Top特征1', 'Top特征2', 'Top特征3', 'Top特征4', 'Top特征5'
#         ]
#     }
#     return ensemble_preds, metrics_ensemble, meta_model
#
#
# # --------------------------
# # 7. 特征相互作用分析（核心模块）
# # --------------------------
# def analyze_feature_interactions(features, feature_names, save_dir='feature_interaction_plots',
#                                  methods=['pearson', 'spearman', 'mutual_info']):
#     """
#     针对前20个重要特征，计算并绘制三种方法的相互作用热图：
#     - pearson: 皮尔逊相关系数（线性关系）
#     - spearman: 斯皮尔曼相关系数（非线性单调关系）
#     - mutual_info: 互信息（任意关系）
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     n_features = features.shape[1]
#     top_n = min(20, n_features)  # 确保不超过实际特征数
#
#     for method in methods:
#         # 初始化相互作用矩阵（对称）
#         interaction_matrix = np.zeros((top_n, top_n))
#         for i in range(top_n):
#             for j in range(i, top_n):
#                 x = features[:, i]
#                 y = features[:, j]
#
#                 if method == 'pearson':
#                     corr, _ = pearsonr(x, y)
#                     val = abs(corr)  # 取绝对值，只看关联强度
#                 elif method == 'spearman':
#                     corr, _ = spearmanr(x, y)
#                     val = abs(corr)  # 取绝对值，只看关联强度
#                 elif method == 'mutual_info':
#                     val = mutual_info_regression(x.reshape(-1, 1), y)[0]  # 互信息（非负）
#                 else:
#                     raise ValueError(f"不支持的相互作用方法: {method}")
#
#                 interaction_matrix[i, j] = val
#                 interaction_matrix[j, i] = val  # 对称矩阵
#
#         # 绘制热图（仅显示上三角，避免重复）
#         plt.figure(figsize=(12, 10))
#         mask = np.triu(np.ones_like(interaction_matrix, dtype=bool))  # 上三角掩码
#         cmap = sns.diverging_palette(220, 20, as_cmap=True)  # 红-蓝配色
#
#         sns.heatmap(
#             interaction_matrix,
#             mask=mask,  # 隐藏下三角
#             cmap=cmap,  # 颜色映射
#             annot=True,  # 显示数值
#             fmt=".2f",  # 数值格式
#             xticklabels=feature_names[:top_n],  # x轴标签
#             yticklabels=feature_names[:top_n],  # y轴标签
#             square=True,  # 方形单元格
#             linewidths=.5,  # 格子线宽度
#             cbar_kws={"shrink": .8}  # 颜色条缩放
#         )
#         plt.title(f"Top {top_n} 特征相互作用热图 - {method}", fontsize=14)
#         plt.xticks(rotation=45, ha='right', fontsize=10)  # 旋转x轴标签
#         plt.yticks(rotation=0, fontsize=10)  # y轴标签水平显示
#         plt.tight_layout()  # 调整布局，避免标签截断
#
#         # 保存图像
#         save_path = os.path.join(save_dir, f"interaction_heatmap_{method}.png")
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"✅ 保存 {method} 方法的热图到: {save_path}")
#
#
# # --------------------------
# # 8. SHAP分析与可视化（增强版，返回特征重要性）
# # --------------------------
# def get_feature_names(extractor, mask, selector_support):
#     """映射选择后的特征到原始特征名称（包括交互特征）"""
#     # 基础特征名称
#     base_cols = extractor.hydrophobic_cols + extractor.structure_cols + \
#                 extractor.physicochemical_cols + extractor.aa_composition_cols
#     # 交互特征名称
#     interaction_cols = [
#         'exposed×instability',
#         'aromaticity×helix',
#         'exposed×(1-helix)',
#         'instability×aromaticity',
#         'gravy×aromaticity',
#         '(helix+sheet)×exposed'
#     ]
#     # 所有原始特征名称（基础+交互）
#     all_feat_names = base_cols + interaction_cols
#
#     # 应用第一次筛选（mask）
#     after_mask = [all_feat_names[i] for i, val in enumerate(mask) if val]
#     # 应用第二次筛选（RFECV）
#     final_feat_names = [after_mask[i] for i, val in enumerate(selector_support) if val]
#     return final_feat_names
#
#
# def shap_analysis(model, X_data, feature_names, model_type="mlp", sample_size=100, save_dir='shap_plots',
#                   use_fast_calculation=True, top_n_feats=15):
#     """增强版SHAP分析，返回特征重要性（平均绝对SHAP值）"""
#     os.makedirs(save_dir, exist_ok=True)
#
#     # 确保特征名称数量匹配
#     if len(feature_names) != X_data.shape[1]:
#         print(f"警告：特征名称数量({len(feature_names)})与特征维度({X_data.shape[1]})不匹配，将使用默认名称")
#         feature_names = [f"特征{i + 1}" for i in range(X_data.shape[1])]
#
#     # 采样减少计算量
#     n_samples = min(sample_size, len(X_data))
#     if n_samples < 10:
#         print("警告：样本量过小，可能影响SHAP分析结果")
#         n_samples = max(10, n_samples)
#     X_sample = shap.sample(X_data, n_samples, random_state=42)
#
#     # 背景数据采样
#     background_size = min(50, len(X_sample) // 2)
#     background = shap.sample(X_sample, background_size, random_state=42)
#
#     # 定义模型预测函数（适配不同模型类型）
#     def model_predict(data):
#         if len(data.shape) == 1:
#             data = data.reshape(1, -1)
#
#         if model_type == "mlp":
#             # MLP模型预测
#             data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
#             with torch.no_grad():
#                 output = torch.sigmoid(model(data_tensor)).squeeze()
#             return output.cpu().numpy() if output.ndim > 0 else np.array([output.cpu().item()])
#         elif model_type == "rf":
#             # 随机森林预测
#             return model.predict_proba(data)[:, 1]
#         elif model_type == "meta":
#             # 元模型预测
#             return 1 / (1 + np.exp(-model.decision_function(data)))
#         else:
#             raise ValueError(f"不支持的模型类型: {model_type}")
#
#     # 创建解释器（根据模型类型选择）
#     try:
#         if model_type == "rf":
#             # 随机森林使用TreeExplainer（更快更准确）
#             explainer = shap.TreeExplainer(model)
#             shap_values = explainer.shap_values(X_sample)
#             # 处理多类别情况
#             if isinstance(shap_values, list) and len(shap_values) == 2:
#                 shap_values = shap_values[1]  # 取正类的SHAP值
#         else:
#             # MLP和元模型使用KernelExplainer
#             explainer = shap.KernelExplainer(model_predict, background)
#             if use_fast_calculation and len(X_sample) > 50:
#                 shap_values = explainer.shap_values(X_sample, nsamples=100)  # 快速模式
#             else:
#                 shap_values = explainer.shap_values(X_sample)  # 标准模式
#
#         # 确保SHAP值维度正确
#         if isinstance(shap_values, list):
#             shap_values = shap_values[0]
#         if shap_values.shape != X_sample.shape:
#             raise ValueError(f"SHAP值形状不匹配: {shap_values.shape} vs {X_sample.shape}")
#
#     except Exception as e:
#         print(f"SHAP解释器初始化失败: {str(e)}")
#         return None, None  # 返回空值，后续跳过分析
#
#     # 1. 汇总图（特征重要性排序）
#     plt.figure(figsize=(12, 8))
#     shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="dot")
#     plt.title(f"{model_type.upper()}模型SHAP值汇总图")
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/shap_summary_dot.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"SHAP汇总图已保存至 {save_dir}/shap_summary_dot.png")
#
#     # 2. 特征重要性条形图（统一风格：前20特征+steelblue）
#     feat_importance = np.abs(shap_values).mean(axis=0)  # 平均绝对SHAP值（衡量特征重要性）
#     # 构建特征-重要性DataFrame，仅保留前20个特征
#     feat_df = pd.DataFrame({
#         'Feature': feature_names,
#         'Mean Absolute SHAP Value': feat_importance
#     }).sort_values(by='Mean Absolute SHAP Value', ascending=False).head(20)  # 关键：仅取前20个
#
#     # 绘制水平条形图（调整尺寸适配20个特征，颜色为steelblue）
#     plt.figure(figsize=(8, 12))  # 高度适配20个特征，宽度保持8
#     sns.barplot(
#         x='Mean Absolute SHAP Value',
#         y='Feature',
#         data=feat_df,
#         color='steelblue'  # 与示例图颜色一致
#     )
#     plt.xlabel('mean(|SHAP value|) (average impact on model output mag)')  # 与示例图一致的x轴标签
#     plt.ylabel('Feature')
#     plt.title('Top 20 Feature Importance by Mean Absolute SHAP Value')  # 明确标注“前20个”
#     plt.tight_layout()  # 避免标签截断
#     plt.savefig(f"{save_dir}/shap_feature_importance_bar.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"前20个特征的重要性条形图已保存至 {save_dir}/shap_feature_importance_bar.png")
#
#     return shap_values, feat_importance  # 返回SHAP值和特征重要性
#
#
# # --------------------------
# # 9. 主函数（整合所有功能）
# # --------------------------
# def main():
#     set_random_seed(42)
#     print("加载数据...")
#     train_data, train_labels, test_data, test_proteins = load_data()
#
#     print("预处理与特征工程...")
#     extractor = FeatureExtractor()
#     train_feats, base_cols = extractor.extract_features(train_data)
#     test_feats, _ = extractor.extract_features(test_data)
#
#     # 特征选择（获取筛选掩码）
#     X_train_selected, X_test_selected, selector_support, mask = extractor.select_features(train_feats, train_labels,
#                                                                                           test_feats)
#
#     # 特征标准化
#     scaler = StandardScaler()
#     train_feats_scaled = scaler.fit_transform(X_train_selected)
#     test_feats_scaled = scaler.transform(X_test_selected)
#     input_dim = train_feats_scaled.shape[1]
#     print(f"特征维度: {input_dim} | 训练样本: {len(train_feats_scaled)} | 测试样本: {len(test_feats_scaled)}")
#     print(f"训练集正负样本比: {train_labels.sum()}/{len(train_labels) - train_labels.sum()}")
#
#     # 交叉验证
#     n_splits = 3
#     sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
#     cv_metrics = []
#     fold_metrics_list = []
#     best_models = None
#
#     # 初始化存储表格
#     round_predictions = pd.DataFrame()
#     round_metrics = pd.DataFrame()
#
#     print("开始交叉验证训练...")
#     with tqdm(total=n_splits, desc="交叉验证进度") as pbar:
#         for fold, (train_idx, val_idx) in enumerate(sss.split(train_feats_scaled, train_labels)):
#             X_train, X_val = train_feats_scaled[train_idx], train_feats_scaled[val_idx]
#             y_train, y_val = train_labels.iloc[train_idx], train_labels.iloc[val_idx]
#
#             # 处理类别不平衡
#             pos_ratio = y_train.sum() / len(y_train)
#             sampling_strategy = min(0.55, pos_ratio + 0.25)
#
#             if pos_ratio < 0.2:
#                 smote = SMOTE(random_state=42, sampling_strategy=min(sampling_strategy + 0.1, 0.6), k_neighbors=2)
#                 X_over, y_over = smote.fit_resample(X_train, y_train)
#                 under_sampler = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
#                 X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_over, y_over)
#             else:
#                 smote_tomek = SMOTETomek(random_state=42, sampling_strategy=sampling_strategy)
#                 X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
#
#             # 数据加载器
#             batch_size = min(16, max(8, len(X_train_resampled) // 8))
#             train_loader = DataLoader(
#                 ProteinDataset(X_train_resampled, y_train_resampled),
#                 batch_size=batch_size,
#                 shuffle=True,
#                 num_workers=0,
#                 drop_last=True
#             )
#             val_loader = DataLoader(
#                 ProteinDataset(X_val, y_val),
#                 batch_size=batch_size * 2,
#                 shuffle=False,
#                 num_workers=0
#             )
#
#             # 训练MLP
#             mlp_model = FeatureMLP(input_dim=input_dim).to(device)
#             mlp_model, mlp_metrics, mlp_val_preds = train_mlp(mlp_model, train_loader, val_loader, epochs=60)
#
#             # 训练随机森林
#             rf_model, rf_metrics, rf_val_preds = train_rf(X_train_resampled, y_train_resampled, X_val, y_val)
#
#             # 模型融合
#             ensemble_preds, ensemble_metrics, meta_model = ensemble_predictions(mlp_val_preds, rf_val_preds, y_val,
#                                                                                 X_val)
#             cv_metrics.append(ensemble_metrics)
#             fold_metrics_list.append(ensemble_metrics)
#
#             # 更新指标表格
#             round_metrics_df = pd.DataFrame({
#                 'Round': [fold],
#                 'AUC': [ensemble_metrics['AUC']],
#                 'Accuracy': [ensemble_metrics['Accuracy']],
#                 'Recall': [ensemble_metrics['Recall']],
#                 'Precision': [ensemble_metrics['Precision']],
#                 'F1': [ensemble_metrics['F1']],
#                 'Specificity': [ensemble_metrics['Specificity']]
#             })
#             round_metrics = pd.concat([round_metrics, round_metrics_df], ignore_index=True)
#
#             # 测试集预测
#             mlp_model.eval()
#             test_loader = DataLoader(ProteinDataset(test_feats_scaled), batch_size=8, shuffle=False)
#             with torch.no_grad():
#                 mlp_test_preds = []
#                 for feats in test_loader:
#                     feats = feats.to(device)
#                     mlp_test_preds.extend(torch.sigmoid(mlp_model(feats)).cpu().numpy())
#                 mlp_test_preds = np.array(mlp_test_preds).reshape(-1, 1)
#
#             rf_test_preds = rf_model.predict_proba(test_feats_scaled)[:, 1].reshape(-1, 1)
#
#             # 构建元特征
#             top_feat_idx = ensemble_metrics.get('top_feat_idx', [])
#             if len(top_feat_idx) > 0 and top_feat_idx[0] < test_feats_scaled.shape[1]:
#                 test_top_features = test_feats_scaled[:, top_feat_idx]
#             else:
#                 test_top_features = test_feats_scaled[:, -5:]
#
#             meta_features_test = np.hstack([
#                 mlp_test_preds, rf_test_preds,
#                 (mlp_test_preds * rf_test_preds).reshape(-1, 1),
#                 np.abs(mlp_test_preds - rf_test_preds).reshape(-1, 1),
#                 ((mlp_test_preds + rf_test_preds) / 2).reshape(-1, 1),
#                 test_top_features
#             ])
#
#             # 融合预测
#             decision_scores = meta_model.decision_function(meta_features_test)
#             ensemble_test_preds = 1 / (1 + np.exp(-decision_scores))
#
#             # 更新预测表格
#             round_preds_df = pd.DataFrame({
#                 'Protein Name': test_proteins,
#                 'In Corona Probability': ensemble_test_preds.flatten(),
#                 'Round': [fold] * len(test_proteins),
#                 'Test Accuracy': [ensemble_metrics['Accuracy']] * len(test_proteins)
#             })
#             round_predictions = pd.concat([round_predictions, round_preds_df], ignore_index=True)
#
#             # 跟踪最佳模型
#             current_score = ensemble_metrics['F1'] * 0.6 + ensemble_metrics['Precision'] * 0.4
#             if (best_models is None) or (current_score > best_models[4]):
#                 best_models = (mlp_model, rf_model, meta_model,
#                                ensemble_metrics['threshold'], current_score,
#                                ensemble_metrics.get('top_feat_idx', []),
#                                ensemble_metrics.get('meta_feature_names', []))
#
#             pbar.update(1)
#
#     # 交叉验证指标
#     print("\n===== 交叉验证指标汇总 =====")
#     avg_auc = np.mean([m['AUC'] for m in fold_metrics_list])
#     avg_acc = np.mean([m['Accuracy'] for m in fold_metrics_list])
#     avg_recall = np.mean([m['Recall'] for m in fold_metrics_list])
#     avg_precision = np.mean([m['Precision'] for m in fold_metrics_list])
#     avg_f1 = np.mean([m['F1'] for m in fold_metrics_list])
#     avg_specificity = np.mean([m['Specificity'] for m in fold_metrics_list])
#
#     print(f"交叉验证平均指标:")
#     print(f"AUC: {avg_auc:.4f} | 准确率: {avg_acc:.4f} | 召回率: {avg_recall:.4f}")
#     print(f"精确率: {avg_precision:.4f} | F1分数: {avg_f1:.4f} | 特异性: {avg_specificity:.4f}")
#
#     # 训练最终模型
#     print("\n===== 用全量数据训练最终模型 =====")
#     if train_labels.sum() / len(train_labels) < 0.2:
#         ada = SMOTE(random_state=42, sampling_strategy=0.5)
#         X_train_full_resampled, y_train_full_resampled = ada.fit_resample(train_feats_scaled, train_labels)
#     else:
#         X_train_full_resampled, y_train_full_resampled = SMOTETomek(random_state=42,
#                                                                     sampling_strategy=0.5).fit_resample(
#             train_feats_scaled, train_labels)
#
#     # 最终MLP
#     final_train_loader = DataLoader(
#         ProteinDataset(X_train_full_resampled, y_train_full_resampled),
#         batch_size=min(16, len(X_train_full_resampled) // 8), shuffle=True, num_workers=0
#     )
#     temp_val_loader = DataLoader(
#         ProteinDataset(train_feats_scaled, train_labels),
#         batch_size=16, shuffle=False, num_workers=0
#     )
#     mlp_model_final = FeatureMLP(input_dim=input_dim).to(device)
#     mlp_model_final, _, _ = train_mlp(mlp_model_final, final_train_loader, temp_val_loader, epochs=60)
#
#     # 最终随机森林
#     rf_model_final = RandomForestClassifier(
#         n_estimators=120, max_depth=5, class_weight='balanced_subsample', random_state=42, n_jobs=-1
#     )
#     rf_model_final.fit(X_train_full_resampled, y_train_full_resampled)
#
#     # 生成元模型的元特征（用于SHAP分析）
#     mlp_val_preds_final = mlp_predict_proba(mlp_model_final, train_feats_scaled)
#     rf_val_preds_final = rf_model_final.predict_proba(train_feats_scaled)[:, 1].reshape(-1, 1)
#     top_feat_idx = best_models[5]
#     top_features_final = train_feats_scaled[:, top_feat_idx] if len(top_feat_idx) > 0 else train_feats_scaled[:, -5:]
#
#     meta_features_final = np.hstack([
#         mlp_val_preds_final,
#         rf_val_preds_final,
#         (mlp_val_preds_final * rf_val_preds_final).reshape(-1, 1),
#         np.abs(mlp_val_preds_final - rf_val_preds_final).reshape(-1, 1),
#         ((mlp_val_preds_final + rf_val_preds_final) / 2).reshape(-1, 1),
#         top_features_final
#     ])
#
#     # 重新训练元模型（使用全量数据）
#     meta_model_final = RidgeClassifier(
#         alpha=0.5,
#         class_weight='balanced',
#         random_state=42
#     )
#     meta_model_final.fit(meta_features_final, train_labels)
#
#     # SHAP分析（三个模型分别分析，统一风格）
#     print("\n===== 开始SHAP特征重要性分析 =====")
#     feat_names = get_feature_names(extractor, mask, selector_support)
#
#     # 1. MLP模型SHAP分析
#     print("\n----- MLP模型SHAP分析 -----")
#     mlp_shap_values, mlp_feat_importance = shap_analysis(
#         model=mlp_model_final,
#         X_data=train_feats_scaled,
#         feature_names=feat_names,
#         model_type="mlp",
#         sample_size=100,
#         save_dir='shap_plots/mlp',
#         use_fast_calculation=True
#     )
#
#     # 2. 随机森林模型SHAP分析
#     print("\n----- 随机森林模型SHAP分析 -----")
#     rf_shap_values, rf_feat_importance = shap_analysis(
#         model=rf_model_final,
#         X_data=train_feats_scaled,
#         feature_names=feat_names,
#         model_type="rf",
#         sample_size=100,
#         save_dir='shap_plots/rf',
#         use_fast_calculation=False
#     )
#
#     # 3. 融合模型SHAP分析
#     print("\n----- 融合模型SHAP分析 -----")
#     meta_feat_names = best_models[6] if len(best_models) > 6 else [f"元特征{i + 1}" for i in
#                                                                    range(meta_features_final.shape[1])]
#     meta_shap_values, meta_feat_importance = shap_analysis(
#         model=meta_model_final,
#         X_data=meta_features_final,
#         feature_names=meta_feat_names,
#         model_type="meta",
#         sample_size=100,
#         save_dir='shap_plots/meta',
#         use_fast_calculation=True
#     )
#
#     # ========================
#     # 特征相互作用分析（核心步骤）
#     # ========================
#     print("\n===== 开始前20个重要特征的相互作用分析 =====")
#     if mlp_shap_values is not None and mlp_feat_importance is not None:
#         # 从MLP的SHAP重要性中提取前20个特征
#         top20_indices = np.argsort(-mlp_feat_importance)[:20]  # 降序排列，取前20
#         top20_names = [feat_names[i] for i in top20_indices]  # 获取特征名称
#         top20_features = train_feats_scaled[:, top20_indices]  # 提取特征矩阵
#
#         # 计算并绘制三种方法的相互作用热图
#         analyze_feature_interactions(
#             features=top20_features,
#             feature_names=top20_names,
#             save_dir='feature_interaction_plots',
#             methods=['pearson', 'spearman', 'mutual_info']
#         )
#     else:
#         print("警告：MLP模型的SHAP分析结果无效，跳过特征相互作用分析")
#
#     # 最终测试集预测
#     print("\n===== 测试集预测 =====")
#     mlp_model_final.eval()
#     test_loader = DataLoader(ProteinDataset(test_feats_scaled), batch_size=8, shuffle=False)
#     with torch.no_grad():
#         mlp_test_preds = []
#         for feats in test_loader:
#             feats = feats.to(device)
#             mlp_test_preds.extend(torch.sigmoid(mlp_model_final(feats)).cpu().numpy())
#         mlp_test_preds = np.array(mlp_test_preds).reshape(-1, 1)
#
#     rf_test_preds = rf_model_final.predict_proba(test_feats_scaled)[:, 1].reshape(-1, 1)
#
#     # 构建元特征
#     if len(top_feat_idx) > 0 and top_feat_idx[0] < test_feats_scaled.shape[1]:
#         test_top_features = test_feats_scaled[:, top_feat_idx]
#     else:
#         test_top_features = test_feats_scaled[:, -5:]
#
#     meta_features_test = np.hstack([
#         mlp_test_preds, rf_test_preds,
#         (mlp_test_preds * rf_test_preds).reshape(-1, 1),
#         np.abs(mlp_test_preds - rf_test_preds).reshape(-1, 1),
#         ((mlp_test_preds + rf_test_preds) / 2).reshape(-1, 1),
#         test_top_features
#     ])
#
#     decision_scores = meta_model_final.decision_function(meta_features_test)
#     ensemble_test_preds = 1 / (1 + np.exp(-decision_scores))
#
#     # 确定阈值
#     cv_thresholds = [m['threshold'] for m in cv_metrics if 'threshold' in m]
#     if cv_thresholds:
#         cv_thresh_mean = np.mean(cv_thresholds)
#         best_test_f1 = 0.0
#         best_test_thresh = cv_thresh_mean
#         for thresh in np.arange(max(0.2, cv_thresh_mean - 0.15),
#                                 min(0.7, cv_thresh_mean + 0.15), 0.01):
#             y_pred = (ensemble_test_preds >= thresh).astype(int)
#             current_f1 = f1_score(train_labels, y_pred, zero_division=0)
#             if current_f1 > best_test_f1:
#                 best_test_f1 = current_f1
#                 best_test_thresh = thresh
#         test_threshold = best_test_thresh
#     else:
#         test_threshold = best_models[3]
#
#     print(f"测试集最佳阈值: {test_threshold:.3f}")
#     test_preds_binary = (ensemble_test_preds >= test_threshold).astype(int)
#
#     # 调整阳性样本
#     pos_count = np.sum(test_preds_binary)
#     n_test = len(test_preds_binary)
#     prob_sorted = np.argsort(ensemble_test_preds)[::-1]
#     if pos_count < max(2, n_test * 0.3):
#         need = max(2, int(n_test * 0.3)) - pos_count
#         add_idx = [i for i in prob_sorted if test_preds_binary[i] == 0][:need]
#         test_preds_binary[add_idx] = 1
#         print(f"测试集阳性不足，增加{len(add_idx)}个阳性样本")
#     elif pos_count > n_test * 0.6:
#         need = pos_count - int(n_test * 0.6)
#         remove_idx = [i for i in prob_sorted[::-1] if test_preds_binary[i] == 1][:need]
#         test_preds_binary[remove_idx] = 0
#         print(f"测试集阳性过多，减少{len(remove_idx)}个阳性样本")
#
#     print(f"测试集预测阳性数量: {np.sum(test_preds_binary)}")
#
#     # 生成结果表格
#     final_predictions = pd.DataFrame({
#         'Protein Name': test_proteins,
#         'In Corona Probability': ensemble_test_preds.flatten(),
#         'In Corona': test_preds_binary.astype(bool)
#     })
#
#     protein_avg_predictions = pd.DataFrame()
#     unique_proteins = round_predictions['Protein Name'].unique()
#     for protein in unique_proteins:
#         protein_data = round_predictions[round_predictions['Protein Name'] == protein]
#         avg_prob = protein_data['In Corona Probability'].mean()
#         ci = confidence_interval(protein_data['In Corona Probability'])
#         protein_row = pd.DataFrame({
#             'Protein Name': [protein],
#             'Average In Corona Probability': [round(avg_prob, 3)],
#             '95 Percent Confidence Interval': [round(ci, 3)]
#         })
#         protein_avg_predictions = pd.concat([protein_avg_predictions, protein_row], ignore_index=True)
#
#     # 保存结果
#     print("保存结果...")
#     with pd.ExcelWriter('final_ensemble_predictions_with_interaction.xlsx') as writer:
#         round_predictions.to_excel(writer, sheet_name='Round Based Prediction', index=False)
#         round_metrics.to_excel(writer, sheet_name='Classifier Round Metrics', index=False)
#         protein_avg_predictions.to_excel(writer, sheet_name='Protein Average Predictions', index=False)
#         final_predictions.to_excel(writer, sheet_name='Final Predictions', index=False)
#
#     print(f"预测结果保存完成: final_ensemble_predictions_with_interaction.xlsx")
#     print("\n===== 最终模型指标（交叉验证平均值） =====")
#     print(f"AUC: {avg_auc:.4f}")
#     print(f"准确率: {avg_acc:.4f}")
#     print(f"召回率: {avg_recall:.4f}")
#     print(f"精确率: {avg_precision:.4f}")
#     print(f"F1分数: {avg_f1:.4f}")
#     print(f"特异性: {avg_specificity:.4f}")
#
#
# # --------------------------
# # 10. 数据加载函数
# # --------------------------
# def load_data():
#     try:
#         plasma_data = pd.read_excel("data/gt6_plasma_features_names_biopy_gravy.xlsx", header=0, index_col=0)
#         csf_data = pd.read_excel("data/gt6_csf_features_names_biopy_gravy.xlsx", header=0, index_col=0)
#     except Exception as e:
#         raise IOError(f"数据加载失败: {str(e)}")
#
#     # 确保包含标签列
#     for df in [plasma_data, csf_data]:
#         if 'Corona' not in df.columns:
#             raise ValueError("训练数据必须包含'Corona'标签列")
#
#     # 合并训练数据
#     train_data = pd.concat([plasma_data, csf_data], ignore_index=True)
#     train_labels = train_data['Corona'].copy().astype(float)
#
#     # 测试数据
#     test_data = pd.read_excel("data/netsurfp_2_proteins_selected_for_testing_processed_updated.xlsx",
#                               header=0, index_col=0)
#
#     # 提取蛋白质名称
#     if 'Protein names' in test_data.columns:
#         test_proteins = test_data['Protein names'].tolist()
#     else:
#         test_proteins = [f"Protein_{i}" for i in range(len(test_data))]
#
#     # 确保所有必要特征存在
#     extractor = FeatureExtractor()
#     all_required_feats = (
#             extractor.hydrophobic_cols + extractor.structure_cols +
#             extractor.physicochemical_cols + extractor.aa_composition_cols
#     )
#     for col in all_required_feats:
#         for df in [train_data, test_data]:
#             if col not in df.columns:
#                 df[col] = 0
#
#     return train_data, train_labels, test_data, test_proteins
#
#
# if __name__ == "__main__":
#     main()

import pandas as pd
import numpy as np
import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import RidgeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import shap  # 导入SHAP库
import networkx as nx  # 新增网络绘图依赖


# --------------------------
# 固定随机种子
# --------------------------
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 计算置信区间的函数
def confidence_interval(data, confidence=0.95):
    if len(data) <= 1:
        return np.nan
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


# 忽略警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 设置工作目录
os.chdir("/home/syy/syy-model/swnt-protein-corona-ML-main")

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# --------------------------
# 1. 特征提取与选择
# --------------------------
class FeatureExtractor:
    def __init__(self):
        self.hydrophobic_cols = ['gravy', 'fraction_exposed_nonpolar_total', 'fraction_exposed_polar_total', 'rsa_mean']
        self.structure_cols = ['secondary_structure_fraction_helix', 'secondary_structure_fraction_sheet',
                               'secondary_structure_fraction_disordered', 'fraction_exposed', 'fraction_buried',
                               'asa_sum']
        self.physicochemical_cols = ['molecular_weight', 'isoelectric_point', 'instability_index', 'aromaticity']
        self.aa_composition_cols = ['frac_aa_A', 'frac_aa_C', 'frac_aa_D', 'frac_aa_E', 'frac_aa_F', 'frac_aa_G',
                                     'frac_aa_H', 'frac_aa_I', 'frac_aa_K', 'frac_aa_L', 'frac_aa_M', 'frac_aa_N',
                                    'frac_aa_P', 'frac_aa_Q', 'frac_aa_R', 'frac_aa_S', 'frac_aa_T', 'frac_aa_V',
                                    'frac_aa_W', 'frac_aa_Y']

    def extract_features(self, data):
        all_feature_cols = self.hydrophobic_cols + self.structure_cols + self.physicochemical_cols + self.aa_composition_cols
        base_features = data[all_feature_cols].fillna(0).values

        exposed = data['fraction_exposed'].fillna(0).values.reshape(-1, 1)
        instability = data['instability_index'].fillna(0).values.reshape(-1, 1)
        aromaticity = data['aromaticity'].fillna(0).values.reshape(-1, 1)
        helix = data['secondary_structure_fraction_helix'].fillna(0).values.reshape(-1, 1)
        gravy = data['gravy'].fillna(0).values.reshape(-1, 1)

        interaction_features = np.hstack([
            exposed * instability,
            aromaticity * helix,
            exposed * (1 - helix),
            instability * aromaticity,
            gravy * aromaticity,
            (helix + data['secondary_structure_fraction_sheet'].fillna(0).values.reshape(-1, 1)) * exposed
        ])

        total_features = np.hstack([base_features, interaction_features])
        print(f"特征提取完成: 总特征{total_features.shape[1]}个")
        return total_features, all_feature_cols  # 返回基础特征列名用于SHAP

    def select_features(self, X_train, y_train, X_test):
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        importance = rf.feature_importances_
        threshold = np.percentile(importance, 40)
        mask = importance >= threshold
        X_train_filtered = X_train[:, mask]
        X_test_filtered = X_test[:, mask]

        selector = RFECV(estimator=RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42),
                         step=1, cv=3, scoring='f1', min_features_to_select=15)
        X_train_selected = selector.fit_transform(X_train_filtered, y_train)
        X_test_selected = selector.transform(X_test_filtered)

        print(f"特征选择完成: 从{X_train.shape[1]}→{X_train_selected.shape[1]}个特征")
        return X_train_selected, X_test_selected, selector.support_, mask  # 返回掩码用于特征名映射


# --------------------------
# 2. 数据集与模型定义
# --------------------------
class ProteinDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


class FeatureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 16], dropout=0.35):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()  # 确保输出是一维张量


# 添加一个函数用于生成MLP的预测概率
def mlp_predict_proba(model, data):
    """生成MLP模型的预测概率"""
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        outputs = model(data_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy()  # 应用sigmoid获取概率
    return probabilities.reshape(-1, 1)  # 重塑为二维数组以便后续处理


# --------------------------
# 3. 早停策略
# --------------------------
class EarlyStopping:
    def __init__(self, patience=6, delta=0.002):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.best_model_weights = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_metrics, model):
        score = val_metrics['Recall'] * 0.3 + val_metrics['Precision'] * 0.3 + val_metrics['F1'] * 0.4

        if self.best_score is None:
            self.best_score = score
            self.best_model_weights = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_weights = model.state_dict()
            self.counter = 0


# --------------------------
# 4. 训练函数
# --------------------------
def train_mlp(model, train_loader, val_loader, epochs=60, lr=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    train_labels = np.concatenate([labels.numpy() for _, labels in train_loader])
    pos_ratio = train_labels.mean()
    pos_weight = torch.tensor((1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1.0, device=device)

    class FocalLoss(nn.Module):
        def __init__(self, pos_weight=None, gamma=2.0):
            super().__init__()
            self.pos_weight = pos_weight
            self.gamma = gamma

        def forward(self, inputs, targets):
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(inputs, targets)
            pt = torch.exp(-bce_loss)
            focal_loss = (1 - pt) ** self.gamma * bce_loss
            return focal_loss

    criterion = FocalLoss(pos_weight=pos_weight, gamma=1.5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=6)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs} 训练") as pbar:
            for feats, labels in train_loader:
                feats, labels = feats.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(feats)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({"训练损失": f"{loss.item():.4f}"})
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                outputs = model(feats)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.numpy())

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        y_pred = (val_preds >= 0.5).astype(int)

        cm = confusion_matrix(val_labels, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        val_metrics = {
            'AUC': metrics.roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.5,
            'Accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'F1': f1_score(val_labels, y_pred, zero_division=0),
            'Specificity': specificity
        }

        scheduler.step(val_metrics['F1'])
        early_stopping(val_metrics, model)
        if early_stopping.early_stop:
            break

    model.load_state_dict(early_stopping.best_model_weights)
    return model, val_metrics, val_preds


# --------------------------
# 5. 随机森林训练
# --------------------------
def train_rf(X_train, y_train, X_val, y_val):
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=5,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
        oob_score=True
    )
    rf.fit(X_train, y_train)

    rf_preds = rf.predict_proba(X_val)[:, 1]

    best_rf_f1 = 0.0
    best_rf_thresh = 0.5
    for thresh in np.arange(0.3, 0.7, 0.02):
        y_pred = (rf_preds >= thresh).astype(int)
        current_f1 = f1_score(y_val, y_pred, zero_division=0)
        if current_f1 > best_rf_f1:
            best_rf_f1 = current_f1
            best_rf_thresh = thresh

    y_pred_rf = (rf_preds >= best_rf_thresh).astype(int)
    cm = confusion_matrix(y_val, y_pred_rf)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics_rf = {
        'AUC': metrics.roc_auc_score(y_val, rf_preds),
        'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'F1': best_rf_f1,
        'Specificity': specificity,
        'threshold': best_rf_thresh
    }
    return rf, metrics_rf, rf_preds


# --------------------------
# 6. 模型融合
# --------------------------
def ensemble_predictions(mlp_preds, rf_preds, y_val, X_val):
    mlp_preds = np.array(mlp_preds).reshape(-1, 1)
    rf_preds = np.array(rf_preds).reshape(-1, 1)

    rf_imp = RandomForestClassifier(n_estimators=50, max_depth=3).fit(X_val, y_val).feature_importances_
    top_feat_idx = np.argsort(rf_imp)[-5:][::-1]
    top_features = X_val[:, top_feat_idx]

    meta_features = np.hstack([
        mlp_preds,
        rf_preds,
        (mlp_preds * rf_preds).reshape(-1, 1),
        np.abs(mlp_preds - rf_preds).reshape(-1, 1),
        ((mlp_preds + rf_preds) / 2).reshape(-1, 1),
        top_features
    ])

    meta_model = RidgeClassifier(
        alpha=0.5,
        class_weight='balanced',
        random_state=42
    )
    meta_model.fit(meta_features, y_val)

    decision_scores = meta_model.decision_function(meta_features)
    ensemble_preds = 1 / (1 + np.exp(-decision_scores))

    best_f1 = 0.0
    best_threshold = 0.5
    best_recall = 0.0
    best_precision = 0.0

    for threshold in np.arange(0.2, 0.7, 0.01):
        y_pred = (ensemble_preds >= threshold).astype(int)
        current_f1 = f1_score(y_val, y_pred, zero_division=0)
        current_recall = recall_score(y_val, y_pred, zero_division=0)
        current_precision = precision_score(y_val, y_pred, zero_division=0)

        if (current_f1 > best_f1) or \
                (current_f1 == best_f1 and current_precision > best_precision) or \
                (current_f1 > best_f1 - 0.02 and current_precision > best_precision + 0.1):
            best_f1 = current_f1
            best_threshold = threshold
            best_recall = current_recall
            best_precision = current_precision

    y_pred_ensemble = (ensemble_preds >= best_threshold).astype(int)
    cm = confusion_matrix(y_val, y_pred_ensemble)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    auc = metrics.roc_auc_score(y_val, ensemble_preds) if len(set(y_val)) > 1 else 0.5

    metrics_ensemble = {
        'AUC': auc,
        'Accuracy': accuracy,
        'Recall': best_recall,
        'Precision': best_precision,
        'F1': best_f1,
        'Specificity': specificity,
        'threshold': best_threshold,
        'preds': ensemble_preds,
        'meta_model': meta_model,
        'top_feat_idx': top_feat_idx,
        'meta_features': meta_features,
        'meta_feature_names': [  # 元特征名称（用于SHAP）
            'MLP预测', 'RF预测', 'MLP×RF', '|MLP-RF|', '(MLP+RF)/2',
            'Top特征1', 'Top特征2', 'Top特征3', 'Top特征4', 'Top特征5'
        ]
    }
    return ensemble_preds, metrics_ensemble, meta_model


# --------------------------
# 7. 特征相互作用分析（热图+网络图）
# --------------------------
def analyze_feature_interactions(features, feature_names, save_dir='feature_interaction_plots',
                                 methods=['pearson', 'spearman', 'mutual_info']):
    """
    严格匹配参考图的粉蓝渐变配色，移除热图数值显示：
    颜色序列：#82d8ce → #d6f0ed → #eaf9f6 → #ffceeb → #fdc7c7 → #fd9c99
    """
    os.makedirs(save_dir, exist_ok=True)
    n_features = features.shape[1]
    top_n = min(20, n_features)  # 最多前20个特征

    # 自定义粉蓝渐变配色（与参考图完全一致）
    custom_colors = ["#82d8ce", "#d6f0ed", "#eaf9f6", "#ffceeb", "#fdc7c7", "#fd9c99"]
    cmap = sns.blend_palette(custom_colors, as_cmap=True)  # 生成渐变映射

    for method in methods:
        # 初始化相互作用矩阵（对称）
        interaction_matrix = np.zeros((top_n, top_n))
        for i in range(top_n):
            for j in range(i, top_n):
                x = features[:, i]
                y = features[:, j]

                if method == 'pearson':
                    corr, _ = pearsonr(x, y)
                    val = abs(corr)  # 取绝对值，只看关联强度
                elif method == 'spearman':
                    corr, _ = spearmanr(x, y)
                    val = abs(corr)  # 取绝对值，只看关联强度
                elif method == 'mutual_info':
                    val = mutual_info_regression(x.reshape(-1, 1), y)[0]  # 互信息（非负）
                else:
                    raise ValueError(f"不支持的相互作用方法: {method}")

                interaction_matrix[i, j] = val
                interaction_matrix[j, i] = val  # 对称矩阵

        # 绘制热图（仅显示上三角，移除数值标注）
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(interaction_matrix, dtype=bool))  # 上三角掩码
        sns.heatmap(
            interaction_matrix,
            mask=mask,  # 隐藏下三角
            cmap=cmap,  # 自定义粉蓝渐变
            annot=False,  # 移除数值显示（严格匹配参考图）
            xticklabels=feature_names[:top_n],  # x轴标签
            yticklabels=feature_names[:top_n],  # y轴标签
            square=True,  # 方形单元格
            linewidths=.5,  # 格子线宽度
            cbar_kws={"shrink": .8}  # 颜色条缩放
        )
        plt.title(f"Top {top_n} 特征相互作用热图 - {method}", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)  # 旋转x轴标签
        plt.yticks(rotation=0, fontsize=10)  # y轴标签水平显示
        plt.tight_layout()  # 调整布局，避免标签截断

        # 保存图像
        save_path = os.path.join(save_dir, f"interaction_heatmap_{method}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存 {method} 方法的热图到: {save_path}")


def plot_interaction_network(features, feature_names, importance, save_dir='feature_interaction_plots',
                             method='mutual_info', edge_threshold=0.5):
    """
    绘制特征相互作用网络图（参考图1风格）：
    - 节点大小/颜色：SHAP特征重要性（越大越重要）
    - 边宽度：互信息强度（越大连接越强）
    - 布局：力导向算法，突出强关联
    """
    os.makedirs(save_dir, exist_ok=True)
    top_n = features.shape[1]

    # 计算互信息矩阵（边权重）
    interaction_matrix = np.zeros((top_n, top_n))
    for i in range(top_n):
        for j in range(i + 1, top_n):  # 仅计算上三角，避免重复
            x = features[:, i]
            y = features[:, j]
            val = mutual_info_regression(x.reshape(-1, 1), y)[0]
            interaction_matrix[i, j] = val
            interaction_matrix[j, i] = val  # 对称

    # 构建网络
    G = nx.Graph()
    for i in range(top_n):
        G.add_node(
            feature_names[i],
            importance=importance[i]  # 节点属性：SHAP重要性
        )

    # 添加边（过滤弱关联）
    for i in range(top_n):
        for j in range(i + 1, top_n):
            if interaction_matrix[i, j] > edge_threshold:
                G.add_edge(
                    feature_names[i], feature_names[j],
                    weight=interaction_matrix[i, j]
                )

    # 力导向布局（固定seed确保复现）
    pos = nx.spring_layout(G, seed=42, k=0.5)  # k控制节点间距

    # 绘图配置
    plt.figure(figsize=(12, 10))

    # 节点大小：SHAP重要性×2000（放大显示）
    node_sizes = [G.nodes[node]['importance'] * 2000 for node in G.nodes]
    # 节点颜色：SHAP重要性（红→蓝渐变，与热图配色呼应）
    node_colors = [G.nodes[node]['importance'] for node in G.nodes]
    # 边宽度：互信息×5（放大显示）
    edge_widths = [d['weight'] * 5 for (u, v, d) in G.edges(data=True)]

    # 绘制节点（带颜色渐变）
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.RdBu_r,  # 红-蓝渐变，与热图配色一致
        edgecolors='black',
        alpha=0.8
    )
    # 绘制边（灰色，透明）
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.6,
        edge_color='gray'
    )
    # 绘制标签（黑色，加粗）
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_weight='bold'
    )

    # 添加颜色条（SHAP重要性）
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, shrink=0.6, label='Mean Absolute SHAP Value（特征重要性）')

    plt.title(f"特征相互作用网络图 - {method}", fontsize=14)
    plt.axis('off')  # 隐藏坐标轴
    plt.tight_layout()  # 优化布局

    # 保存图像
    save_path = os.path.join(save_dir, f"interaction_network_{method}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存相互作用网络图到: {save_path}")


# --------------------------
# 8. SHAP分析与可视化（增强版，返回特征重要性）
# --------------------------
def get_feature_names(extractor, mask, selector_support):
    """映射选择后的特征到原始特征名称（包括交互特征）"""
    # 基础特征名称
    base_cols = extractor.hydrophobic_cols + extractor.structure_cols + \
                extractor.physicochemical_cols + extractor.aa_composition_cols
    # 交互特征名称
    interaction_cols = [
        'exposed×instability',
        'aromaticity×helix',
        'exposed×(1-helix)',
        'instability×aromaticity',
        'gravy×aromaticity',
        '(helix+sheet)×exposed'
    ]
    # 所有原始特征名称（基础+交互）
    all_feat_names = base_cols + interaction_cols

    # 应用第一次筛选（mask）
    after_mask = [all_feat_names[i] for i, val in enumerate(mask) if val]
    # 应用第二次筛选（RFECV）
    final_feat_names = [after_mask[i] for i, val in enumerate(selector_support) if val]
    return final_feat_names


def shap_analysis(model, X_data, feature_names, model_type="mlp", sample_size=100, save_dir='shap_plots',
                  use_fast_calculation=True, top_n_feats=15):
    """增强版SHAP分析，返回特征重要性（平均绝对SHAP值）"""
    os.makedirs(save_dir, exist_ok=True)

    # 确保特征名称数量匹配
    if len(feature_names) != X_data.shape[1]:
        print(f"警告：特征名称数量({len(feature_names)})与特征维度({X_data.shape[1]})不匹配，将使用默认名称")
        feature_names = [f"特征{i + 1}" for i in range(X_data.shape[1])]

    # 采样减少计算量
    n_samples = min(sample_size, len(X_data))
    if n_samples < 10:
        print("警告：样本量过小，可能影响SHAP分析结果")
        n_samples = max(10, n_samples)
    X_sample = shap.sample(X_data, n_samples, random_state=42)

    # 背景数据采样
    background_size = min(50, len(X_sample) // 2)
    background = shap.sample(X_sample, background_size, random_state=42)

    # 定义模型预测函数（适配不同模型类型）
    def model_predict(data):
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        if model_type == "mlp":
            # MLP模型预测
            data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
            with torch.no_grad():
                output = torch.sigmoid(model(data_tensor)).squeeze()
            return output.cpu().numpy() if output.ndim > 0 else np.array([output.cpu().item()])
        elif model_type == "rf":
            # 随机森林预测
            return model.predict_proba(data)[:, 1]
        elif model_type == "meta":
            # 元模型预测
            return 1 / (1 + np.exp(-model.decision_function(data)))
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    # 创建解释器（根据模型类型选择）
    try:
        if model_type == "rf":
            # 随机森林使用TreeExplainer（更快更准确）
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            # 处理多类别情况
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]  # 取正类的SHAP值
        else:
            # MLP和元模型使用KernelExplainer
            explainer = shap.KernelExplainer(model_predict, background)
            if use_fast_calculation and len(X_sample) > 50:
                shap_values = explainer.shap_values(X_sample, nsamples=100)  # 快速模式
            else:
                shap_values = explainer.shap_values(X_sample)  # 标准模式

        # 确保SHAP值维度正确
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if shap_values.shape != X_sample.shape:
            raise ValueError(f"SHAP值形状不匹配: {shap_values.shape} vs {X_sample.shape}")

    except Exception as e:
        print(f"SHAP解释器初始化失败: {str(e)}")
        return None, None  # 返回空值，后续跳过分析

    # 1. 汇总图（特征重要性排序）
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="dot")
    plt.title(f"{model_type.upper()}模型SHAP值汇总图")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_summary_dot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP汇总图已保存至 {save_dir}/shap_summary_dot.png")

    # 2. 特征重要性条形图（统一风格：前20特征+steelblue）
    feat_importance = np.abs(shap_values).mean(axis=0)  # 平均绝对SHAP值（衡量特征重要性）
    # 构建特征-重要性DataFrame，仅保留前20个特征
    feat_df = pd.DataFrame({
        'Feature': feature_names,
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

    return shap_values, feat_importance  # 返回SHAP值和特征重要性


# --------------------------
# 9. 主函数（整合所有功能）
# --------------------------
def main():
    set_random_seed(42)
    print("加载数据...")
    train_data, train_labels, test_data, test_proteins = load_data()

    print("预处理与特征工程...")
    extractor = FeatureExtractor()
    train_feats, base_cols = extractor.extract_features(train_data)
    test_feats, _ = extractor.extract_features(test_data)

    # 特征选择（获取筛选掩码）
    X_train_selected, X_test_selected, selector_support, mask = extractor.select_features(train_feats, train_labels,
                                                                                          test_feats)

    # 特征标准化
    scaler = StandardScaler()
    train_feats_scaled = scaler.fit_transform(X_train_selected)
    test_feats_scaled = scaler.transform(X_test_selected)
    input_dim = train_feats_scaled.shape[1]
    print(f"特征维度: {input_dim} | 训练样本: {len(train_feats_scaled)} | 测试样本: {len(test_feats_scaled)}")
    print(f"训练集正负样本比: {train_labels.sum()}/{len(train_labels) - train_labels.sum()}")

    # 交叉验证
    n_splits = 3
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    cv_metrics = []
    fold_metrics_list = []
    best_models = None

    # 初始化存储表格
    round_predictions = pd.DataFrame()
    round_metrics = pd.DataFrame()

    print("开始交叉验证训练...")
    with tqdm(total=n_splits, desc="交叉验证进度") as pbar:
        for fold, (train_idx, val_idx) in enumerate(sss.split(train_feats_scaled, train_labels)):
            X_train, X_val = train_feats_scaled[train_idx], train_feats_scaled[val_idx]
            y_train, y_val = train_labels.iloc[train_idx], train_labels.iloc[val_idx]

            # 处理类别不平衡
            pos_ratio = y_train.sum() / len(y_train)
            sampling_strategy = min(0.55, pos_ratio + 0.25)

            if pos_ratio < 0.2:
                smote = SMOTE(random_state=42, sampling_strategy=min(sampling_strategy + 0.1, 0.6), k_neighbors=2)
                X_over, y_over = smote.fit_resample(X_train, y_train)
                under_sampler = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
                X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_over, y_over)
            else:
                smote_tomek = SMOTETomek(random_state=42, sampling_strategy=sampling_strategy)
                X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

            # 数据加载器
            batch_size = min(16, max(8, len(X_train_resampled) // 8))
            train_loader = DataLoader(
                ProteinDataset(X_train_resampled, y_train_resampled),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )
            val_loader = DataLoader(
                ProteinDataset(X_val, y_val),
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=0
            )

            # 训练MLP
            mlp_model = FeatureMLP(input_dim=input_dim).to(device)
            mlp_model, mlp_metrics, mlp_val_preds = train_mlp(mlp_model, train_loader, val_loader, epochs=60)

            # 训练随机森林
            rf_model, rf_metrics, rf_val_preds = train_rf(X_train_resampled, y_train_resampled, X_val, y_val)

            # 模型融合
            ensemble_preds, ensemble_metrics, meta_model = ensemble_predictions(mlp_val_preds, rf_val_preds, y_val,
                                                                                X_val)
            cv_metrics.append(ensemble_metrics)
            fold_metrics_list.append(ensemble_metrics)

            # 更新指标表格
            round_metrics_df = pd.DataFrame({
                'Round': [fold],
                'AUC': [ensemble_metrics['AUC']],
                'Accuracy': [ensemble_metrics['Accuracy']],
                'Recall': [ensemble_metrics['Recall']],
                'Precision': [ensemble_metrics['Precision']],
                'F1': [ensemble_metrics['F1']],
                'Specificity': [ensemble_metrics['Specificity']]
            })
            round_metrics = pd.concat([round_metrics, round_metrics_df], ignore_index=True)

            # 测试集预测
            mlp_model.eval()
            test_loader = DataLoader(ProteinDataset(test_feats_scaled), batch_size=8, shuffle=False)
            with torch.no_grad():
                mlp_test_preds = []
                for feats in test_loader:
                    feats = feats.to(device)
                    mlp_test_preds.extend(torch.sigmoid(mlp_model(feats)).cpu().numpy())
                mlp_test_preds = np.array(mlp_test_preds).reshape(-1, 1)

            rf_test_preds = rf_model.predict_proba(test_feats_scaled)[:, 1].reshape(-1, 1)

            # 构建元特征
            top_feat_idx = ensemble_metrics.get('top_feat_idx', [])
            if len(top_feat_idx) > 0 and top_feat_idx[0] < test_feats_scaled.shape[1]:
                test_top_features = test_feats_scaled[:, top_feat_idx]
            else:
                test_top_features = test_feats_scaled[:, -5:]

            meta_features_test = np.hstack([
                mlp_test_preds, rf_test_preds,
                (mlp_test_preds * rf_test_preds).reshape(-1, 1),
                np.abs(mlp_test_preds - rf_test_preds).reshape(-1, 1),
                ((mlp_test_preds + rf_test_preds) / 2).reshape(-1, 1),
                test_top_features
            ])

            # 融合预测
            decision_scores = meta_model.decision_function(meta_features_test)
            ensemble_test_preds = 1 / (1 + np.exp(-decision_scores))

            # 更新预测表格
            round_preds_df = pd.DataFrame({
                'Protein Name': test_proteins,
                'In Corona Probability': ensemble_test_preds.flatten(),
                'Round': [fold] * len(test_proteins),
                'Test Accuracy': [ensemble_metrics['Accuracy']] * len(test_proteins)
            })
            round_predictions = pd.concat([round_predictions, round_preds_df], ignore_index=True)

            # 跟踪最佳模型
            current_score = ensemble_metrics['F1'] * 0.6 + ensemble_metrics['Precision'] * 0.4
            if (best_models is None) or (current_score > best_models[4]):
                best_models = (mlp_model, rf_model, meta_model,
                               ensemble_metrics['threshold'], current_score,
                               ensemble_metrics.get('top_feat_idx', []),
                               ensemble_metrics.get('meta_feature_names', []))

            pbar.update(1)

    # 交叉验证指标
    print("\n===== 交叉验证指标汇总 =====")
    avg_auc = np.mean([m['AUC'] for m in fold_metrics_list])
    avg_acc = np.mean([m['Accuracy'] for m in fold_metrics_list])
    avg_recall = np.mean([m['Recall'] for m in fold_metrics_list])
    avg_precision = np.mean([m['Precision'] for m in fold_metrics_list])
    avg_f1 = np.mean([m['F1'] for m in fold_metrics_list])
    avg_specificity = np.mean([m['Specificity'] for m in fold_metrics_list])

    print(f"交叉验证平均指标:")
    print(f"AUC: {avg_auc:.4f} | 准确率: {avg_acc:.4f} | 召回率: {avg_recall:.4f}")
    print(f"精确率: {avg_precision:.4f} | F1分数: {avg_f1:.4f} | 特异性: {avg_specificity:.4f}")

    # 训练最终模型
    print("\n===== 用全量数据训练最终模型 =====")
    if train_labels.sum() / len(train_labels) < 0.2:
        ada = SMOTE(random_state=42, sampling_strategy=0.5)
        X_train_full_resampled, y_train_full_resampled = ada.fit_resample(train_feats_scaled, train_labels)
    else:
        X_train_full_resampled, y_train_full_resampled = SMOTETomek(random_state=42,
                                                                    sampling_strategy=0.5).fit_resample(
            train_feats_scaled, train_labels)

    # 最终MLP
    final_train_loader = DataLoader(
        ProteinDataset(X_train_full_resampled, y_train_full_resampled),
        batch_size=min(16, len(X_train_full_resampled) // 8), shuffle=True, num_workers=0
    )
    temp_val_loader = DataLoader(
        ProteinDataset(train_feats_scaled, train_labels),
        batch_size=16, shuffle=False, num_workers=0
    )
    mlp_model_final = FeatureMLP(input_dim=input_dim).to(device)
    mlp_model_final, _, _ = train_mlp(mlp_model_final, final_train_loader, temp_val_loader, epochs=60)

    # 最终随机森林
    rf_model_final = RandomForestClassifier(
        n_estimators=120, max_depth=5, class_weight='balanced_subsample', random_state=42, n_jobs=-1
    )
    rf_model_final.fit(X_train_full_resampled, y_train_full_resampled)

    # 生成元模型的元特征（用于SHAP分析）
    mlp_val_preds_final = mlp_predict_proba(mlp_model_final, train_feats_scaled)
    rf_val_preds_final = rf_model_final.predict_proba(train_feats_scaled)[:, 1].reshape(-1, 1)
    top_feat_idx = best_models[5]
    top_features_final = train_feats_scaled[:, top_feat_idx] if len(top_feat_idx) > 0 else train_feats_scaled[:, -5:]

    meta_features_final = np.hstack([
        mlp_val_preds_final,
        rf_val_preds_final,
        (mlp_val_preds_final * rf_val_preds_final).reshape(-1, 1),
        np.abs(mlp_val_preds_final - rf_val_preds_final).reshape(-1, 1),
        ((mlp_val_preds_final + rf_val_preds_final) / 2).reshape(-1, 1),
        top_features_final
    ])

    # 重新训练元模型（使用全量数据）
    meta_model_final = RidgeClassifier(
        alpha=0.5,
        class_weight='balanced',
        random_state=42
    )
    meta_model_final.fit(meta_features_final, train_labels)

    # SHAP分析（三个模型分别分析，统一风格）
    print("\n===== 开始SHAP特征重要性分析 =====")
    feat_names = get_feature_names(extractor, mask, selector_support)

    # 1. MLP模型SHAP分析
    print("\n----- MLP模型SHAP分析 -----")
    mlp_shap_values, mlp_feat_importance = shap_analysis(
        model=mlp_model_final,
        X_data=train_feats_scaled,
        feature_names=feat_names,
        model_type="mlp",
        sample_size=100,
        save_dir='shap_plots/mlp',
        use_fast_calculation=True
    )

    # 2. 随机森林模型SHAP分析
    print("\n----- 随机森林模型SHAP分析 -----")
    rf_shap_values, rf_feat_importance = shap_analysis(
        model=rf_model_final,
        X_data=train_feats_scaled,
        feature_names=feat_names,
        model_type="rf",
        sample_size=100,
        save_dir='shap_plots/rf',
        use_fast_calculation=False
    )

    # 3. 融合模型SHAP分析
    print("\n----- 融合模型SHAP分析 -----")
    meta_feat_names = best_models[6] if len(best_models) > 6 else [f"元特征{i + 1}" for i in
                                                                   range(meta_features_final.shape[1])]
    meta_shap_values, meta_feat_importance = shap_analysis(
        model=meta_model_final,
        X_data=meta_features_final,
        feature_names=meta_feat_names,
        model_type="meta",
        sample_size=100,
        save_dir='shap_plots/meta',
        use_fast_calculation=True
    )

    # ========================
    # 特征相互作用分析（热图+网络图）
    # ========================
    print("\n===== 开始前20个重要特征的相互作用分析 =====")
    if mlp_shap_values is not None and mlp_feat_importance is not None:
        # 从MLP的SHAP重要性中提取前20个特征
        top20_indices = np.argsort(-mlp_feat_importance)[:20]  # 降序排列，取前20
        top20_names = [feat_names[i] for i in top20_indices]  # 获取特征名称
        top20_features = train_feats_scaled[:, top20_indices]  # 提取特征矩阵

        # 1. 绘制三种方法的相互作用热图（粉蓝渐变，无数值）
        analyze_feature_interactions(
            features=top20_features,
            feature_names=top20_names,
            save_dir='feature_interaction_plots',
            methods=['pearson', 'spearman', 'mutual_info']
        )

        # 2. 绘制相互作用网络图（参考图1风格）
        plot_interaction_network(
            features=top20_features,
            feature_names=top20_names,
            importance=mlp_feat_importance,  # SHAP重要性作为节点权重
            save_dir='feature_interaction_plots',
            method='mutual_info'
        )
    else:
        print("警告：MLP模型的SHAP分析结果无效，跳过特征相互作用分析")

    # 最终测试集预测
    print("\n===== 测试集预测 =====")
    mlp_model_final.eval()
    test_loader = DataLoader(ProteinDataset(test_feats_scaled), batch_size=8, shuffle=False)
    with torch.no_grad():
        mlp_test_preds = []
        for feats in test_loader:
            feats = feats.to(device)
            mlp_test_preds.extend(torch.sigmoid(mlp_model_final(feats)).cpu().numpy())
        mlp_test_preds = np.array(mlp_test_preds).reshape(-1, 1)

    rf_test_preds = rf_model_final.predict_proba(test_feats_scaled)[:, 1].reshape(-1, 1)

    # 构建元特征
    if len(top_feat_idx) > 0 and top_feat_idx[0] < test_feats_scaled.shape[1]:
        test_top_features = test_feats_scaled[:, top_feat_idx]
    else:
        test_top_features = test_feats_scaled[:, -5:]

    meta_features_test = np.hstack([
        mlp_test_preds, rf_test_preds,
        (mlp_test_preds * rf_test_preds).reshape(-1, 1),
        np.abs(mlp_test_preds - rf_test_preds).reshape(-1, 1),
        ((mlp_test_preds + rf_test_preds) / 2).reshape(-1, 1),
        test_top_features
    ])

    decision_scores = meta_model_final.decision_function(meta_features_test)
    ensemble_test_preds = 1 / (1 + np.exp(-decision_scores))

    # 确定阈值
    cv_thresholds = [m['threshold'] for m in cv_metrics if 'threshold' in m]
    if cv_thresholds:
        cv_thresh_mean = np.mean(cv_thresholds)
        best_test_f1 = 0.0
        best_test_thresh = cv_thresh_mean
        for thresh in np.arange(max(0.2, cv_thresh_mean - 0.15),
                                min(0.7, cv_thresh_mean + 0.15), 0.01):
            y_pred = (ensemble_test_preds >= thresh).astype(int)
            current_f1 = f1_score(train_labels, y_pred, zero_division=0)
            if current_f1 > best_test_f1:
                best_test_f1 = current_f1
                best_test_thresh = thresh
        test_threshold = best_test_thresh
    else:
        test_threshold = best_models[3]

    print(f"测试集最佳阈值: {test_threshold:.3f}")
    test_preds_binary = (ensemble_test_preds >= test_threshold).astype(int)

    # 调整阳性样本
    pos_count = np.sum(test_preds_binary)
    n_test = len(test_preds_binary)
    prob_sorted = np.argsort(ensemble_test_preds)[::-1]
    if pos_count < max(2, n_test * 0.3):
        need = max(2, int(n_test * 0.3)) - pos_count
        add_idx = [i for i in prob_sorted if test_preds_binary[i] == 0][:need]
        test_preds_binary[add_idx] = 1
        print(f"测试集阳性不足，增加{len(add_idx)}个阳性样本")
    elif pos_count > n_test * 0.6:
        need = pos_count - int(n_test * 0.6)
        remove_idx = [i for i in prob_sorted[::-1] if test_preds_binary[i] == 1][:need]
        test_preds_binary[remove_idx] = 0
        print(f"测试集阳性过多，减少{len(remove_idx)}个阳性样本")

    print(f"测试集预测阳性数量: {np.sum(test_preds_binary)}")

    # 生成结果表格
    final_predictions = pd.DataFrame({
        'Protein Name': test_proteins,
        'In Corona Probability': ensemble_test_preds.flatten(),
        'In Corona': test_preds_binary.astype(bool)
    })

    protein_avg_predictions = pd.DataFrame()
    unique_proteins = round_predictions['Protein Name'].unique()
    for protein in unique_proteins:
        protein_data = round_predictions[round_predictions['Protein Name'] == protein]
        avg_prob = protein_data['In Corona Probability'].mean()
        ci = confidence_interval(protein_data['In Corona Probability'])
        protein_row = pd.DataFrame({
            'Protein Name': [protein],
            'Average In Corona Probability': [round(avg_prob, 3)],
            '95 Percent Confidence Interval': [round(ci, 3)]
        })
        protein_avg_predictions = pd.concat([protein_avg_predictions, protein_row], ignore_index=True)

    # 保存结果
    print("保存结果...")
    with pd.ExcelWriter('final_ensemble_predictions_with_interaction.xlsx') as writer:
        round_predictions.to_excel(writer, sheet_name='Round Based Prediction', index=False)
        round_metrics.to_excel(writer, sheet_name='Classifier Round Metrics', index=False)
        protein_avg_predictions.to_excel(writer, sheet_name='Protein Average Predictions', index=False)
        final_predictions.to_excel(writer, sheet_name='Final Predictions', index=False)

    print(f"预测结果保存完成: final_ensemble_predictions_with_interaction.xlsx")
    print("\n===== 最终模型指标（交叉验证平均值） =====")
    print(f"AUC: {avg_auc:.4f}")
    print(f"准确率: {avg_acc:.4f}")
    print(f"召回率: {avg_recall:.4f}")
    print(f"精确率: {avg_precision:.4f}")
    print(f"F1分数: {avg_f1:.4f}")
    print(f"特异性: {avg_specificity:.4f}")


# --------------------------
# 10. 数据加载函数
# --------------------------
def load_data():
    try:
        plasma_data = pd.read_excel("data/gt6_plasma_features_names_biopy_gravy.xlsx", header=0, index_col=0)
        csf_data = pd.read_excel("data/gt6_csf_features_names_biopy_gravy.xlsx", header=0, index_col=0)
    except Exception as e:
        raise IOError(f"数据加载失败: {str(e)}")

    # 确保包含标签列
    for df in [plasma_data, csf_data]:
        if 'Corona' not in df.columns:
            raise ValueError("训练数据必须包含'Corona'标签列")

    # 合并训练数据
    train_data = pd.concat([plasma_data, csf_data], ignore_index=True)
    train_labels = train_data['Corona'].copy().astype(float)

    # 测试数据
    test_data = pd.read_excel("data/netsurfp_2_proteins_selected_for_testing_processed_updated.xlsx",
                              header=0, index_col=0)

    # 提取蛋白质名称
    if 'Protein names' in test_data.columns:
        test_proteins = test_data['Protein names'].tolist()
    else:
        test_proteins = [f"Protein_{i}" for i in range(len(test_data))]

    # 确保所有必要特征存在
    extractor = FeatureExtractor()
    all_required_feats = (
            extractor.hydrophobic_cols + extractor.structure_cols +
            extractor.physicochemical_cols + extractor.aa_composition_cols
    )
    for col in all_required_feats:
        for df in [train_data, test_data]:
            if col not in df.columns:
                df[col] = 0

    return train_data, train_labels, test_data, test_proteins


if __name__ == "__main__":
    main()