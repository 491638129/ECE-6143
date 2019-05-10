# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import gc

from utils import lending_cate_cols, lending_num_cols, lending_fill_cols, lending_useless_cols, lending_lower_cols

from preprocessing import Preprocesser
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse


def lr_train(x, y):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    cate_feas = x[lending_cate_cols].values
    num_feas = x[lending_num_cols].values
    onehot_enc = OneHotEncoder(sparse=True, dtype=np.float32)
    cate_feas = onehot_enc.fit_transform(cate_feas)

    scaler = StandardScaler()
    num_feas = scaler.fit_transform(num_feas)
    num_feas = sparse.csr_matrix(num_feas)
    feas = sparse.hstack([cate_feas, num_feas]).tocsr()
    #feas = feas.toarray()
    del cate_feas, num_feas
    gc.collect()
    val_score = []
    for tr_idx, val_idx in kfold.split(y, y):
        tr_x, tr_y, val_x, val_y = feas[tr_idx,
                                        :], y[tr_idx], feas[val_idx, :], y[val_idx]
        lr = LogisticRegression(C=1)
        lr.fit(tr_x, tr_y)
        val_pred = lr.predict_proba(val_x)[:, 1]
        auc_score = roc_auc_score(val_y, val_pred)
        val_score.append(auc_score)
        print(auc_score)
    return np.mean(val_score)


def svm_train(x, y):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    cate_feas = x[lending_cate_cols].values
    num_feas = x[lending_num_cols].values
    onehot_enc = OneHotEncoder(sparse=True, dtype=np.float32)
    cate_feas = onehot_enc.fit_transform(cate_feas)

    scaler = StandardScaler()
    num_feas = scaler.fit_transform(num_feas)
    num_feas = sparse.csr_matrix(num_feas)
    feas = sparse.hstack([cate_feas, num_feas]).tocsr()

    del cate_feas, num_feas
    gc.collect()
    val_score = []
    for tr_idx, val_idx in kfold.split(y, y):
        tr_x, tr_y, val_x, val_y = feas[tr_idx,
                                        :], y[tr_idx], feas[val_idx, :], y[val_idx]
        svm = SVC(kernel='linear', verbose=True,
                  max_iter=1000, probability=True)
        svm.fit(tr_x, tr_y)
        val_pred = svm.predict_proba(val_x)[:, 1]
        auc_score = roc_auc_score(val_y, val_pred)
        val_score.append(auc_score)
        print(auc_score)
    return np.mean(val_score)


def lgb_train(x, y):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    lgb_paras = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'lambda_l1': 0.1,
        'lambda_l2': 1,
        'seed': 2019,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,

    }
    fea_names = lending_cate_cols+lending_num_cols
    val_score = []
    feature_importance_list = []
    for tr_idx, val_idx in kfold.split(x, y):
        train_x, train_y, val_x, val_y = x.iloc[tr_idx], y[tr_idx], x.iloc[val_idx], y[val_idx]
        train_data = lgb.Dataset(
            train_x, train_y, feature_name=fea_names, categorical_feature=lending_cate_cols)
        val_data = lgb.Dataset(
            val_x, val_y, feature_name=fea_names, categorical_feature=lending_cate_cols)

        lgb_model = lgb.train(lgb_paras, train_data, valid_sets=[train_data,
                                                                 val_data], early_stopping_rounds=50, num_boost_round=10000, verbose_eval=1)
        val_pred = lgb_model.predict(val_x)
        auc_score = roc_auc_score(val_y, val_pred)
        val_score.append(auc_score)
        feature_importance_list.append(
            lgb_model.feature_importance(importance_type='gain'))
    import matplotlib.pyplot as plt
    model = lgb.LGBMClassifier(n_estimators=1000, objective='binary',
                               subsample=0.8, subsample_freq=4, colsample_bytree=0.8, random_state=2019, reg_alpha=0.1, reg_lambda=1)
    model = model.fit(train_x, train_y, eval_set=[
                      (train_x, train_y), (val_x, val_y)], early_stopping_rounds=50, eval_metric=['binary_logloss', 'auc'])
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    lgb.plot_metric(model, metric='auc', dataset_names=None, ax=ax1, xlim=None, ylim=None,
                    title='AUC during training', xlabel='Iterations', ylabel='AUC', figsize=None, grid=True)
    ax2 = fig.add_subplot(2, 1, 2)
    lgb.plot_metric(model, metric='binary_logloss', dataset_names=None, ax=ax2, xlim=None, ylim=None,
                    title='logloss during training', xlabel='Iterations', ylabel='logloss', figsize=None, grid=True)
    plt.tight_layout()
    fig.savefig('lending_club_log.png')

    fea_imp_df = pd.DataFrame({
        'featureName': fea_names,
        'importance': np.mean(feature_importance_list, axis=0)
    })
    fea_imp_df = fea_imp_df.sort_values('importance', ascending=False)
    fea_imp_df['importance'] = fea_imp_df['importance'].astype(int)
    fea_imp_df.to_csv('lending_club_importance.csv', index=False)
    return np.mean(val_score)


if __name__ == '__main__':
    data = pd.read_excel('./data/Lending Club.xlsx',
                         header=2, encoding='utf-8')
    processer = Preprocesser(data, categorical_cols=lending_cate_cols, numerical_cols=lending_num_cols,
                             fill_cols=lending_fill_cols, drop_cols=lending_useless_cols, lower_cols=lending_lower_cols)
    processer.drop_useless_cols()
    processer.to_lower()
    processer.fill_na()
    processer.replace_low_freq_value()
    processer.label_encoder()
    data = processer.data
    print(data.shape)
    data['delinq_2yrs'] = data['delinq_2yrs'].apply(
        lambda x: 1 if x > 0 else 0)
    y = data['delinq_2yrs'].values
    x = data.drop(['delinq_2yrs'], axis=1)
    fea_names = lending_cate_cols+lending_num_cols
    x = x[fea_names]
    lgb_score = lgb_train(x, y)
    lr_score = lr_train(x, y)
    svm_score = svm_train(x, y)

    print('logistic regression auc:', lr_score)
    print('svm auc:', svm_score)
    print('gbdt auc:', lgb_score)
