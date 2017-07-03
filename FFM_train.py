import pandas as pd
import lightgbm as lgb
import numpy as np
import gc


def load_all_data():
    X_train = pd.read_csv('train.csv')
    X_val = pd.read_csv('val.csv')
    X_test = pd.read_csv('test.csv')
    return X_train, X_val, X_test


def load_cate_data():
    X_train = pd.read_csv('cate_train.csv')
    X_val = pd.read_csv('cate_val.csv')
    X_test = pd.read_csv('cate_test.csv')
    return X_train, X_val, X_test


def predict(model, data):
    preds_lgb = model._Booster.predict(data, num_iteration=model.best_iteration, pred_leaf=True)
    df = pd.DataFrame(preds_lgb)
    return df


def convert(X_train, y_train, ntrain, nval):
    rank_col = np.array(X_train.columns)
    # category feature
    train_rank = X_train.loc[:, rank_col].rank(method='dense').astype(int)
    addcount = 0
    for c in rank_col:
        train_rank[c] = train_rank[c] + addcount
        addcount = train_rank[c].max()

    count_feat = 0
    for col in rank_col:
        f = lambda x: str(count_feat) + ':' + str(x) + ':1'
        X_train[col] = train_rank[col].apply(f)
        count_feat = count_feat + 1

    X_train.insert(0, 'label', y_train)

    X_train.loc[:ntrain-1].to_csv('ffm_train.csv', sep=' ', header=False, index=False)
    X_train.loc[ntrain:ntrain+nval-1].to_csv('ffm_val.csv', sep=' ', header=False, index=False)
    X_train.loc[ntrain+nval:].to_csv('ffm_test.csv', sep=' ', header=False, index=False)


# load all data
X_train, X_val, X_test = load_all_data()
# get label
y_train = X_train.pop('label')
y_val = X_val.pop('label')
# handle
X_train.fillna(value=0, inplace=True)
X_val.fillna(value=0, inplace=True)
X_test.fillna(value=0, inplace=True)
# train
model = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.5, n_estimators=10000, objective='binary')
model.fit(X_train, y_train, eval_metric='binary_logloss', eval_set=[(X_train, y_train), (X_val, y_val)],early_stopping_rounds=100)
# predict
lgb_train = predict(model,X_train)
lgb_val = predict(model,X_val)
lgb_test = predict(model,X_test)
#load categorical data
cate_train, cate_val, cate_test = load_cate_data()
train =pd.concat([cate_train,lgb_train],axis=1).reset_index()
train.drop(['index'],axis=1,inplace=True)
val =pd.concat([cate_val,lgb_val],axis=1).reset_index()
val.drop(['index'],axis=1,inplace=True)
test =pd.concat([cate_test,lgb_test],axis=1).reset_index()
test.drop(['index'],axis=1,inplace=True)
del cate_train
del cate_val
del cate_test
del lgb_train
del lgb_val
del lgb_test
gc.collect()

# get label
y_train = train.pop('label')
y_val = val.pop('label')
y_test = test.pop('label')
test_instanceID = test.pop('instanceID')
# concat all data & label
all_train =pd.concat([train,val,test]).reset_index()
all_train.drop(['index'],axis=1,inplace=True)
all_label =pd.concat([y_train,y_val,y_test]).reset_index()
all_label.drop(['index'],axis=1,inplace=True)

ntrain = X_train.shape[0]
nval = X_val.shape[0]
del train
del val
del test
del y_train
del y_val
del y_test
gc.collect()
convert(all_train,all_label,ntrain,nval)