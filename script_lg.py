import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from collections import defaultdict

usecols = ['ncodpers', 'fecha_dato' , 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
       
df_train = pd.read_csv('train_ver2.csv', usecols=usecols)
sample = pd.read_csv('sample_submission.csv')
df_train_old= df_train[df_train["fecha_dato"]=="2016-04-28"]

# df_train = df_train.drop_duplicates(['ncodpers'], keep='last')
df_train= df_train[df_train["fecha_dato"]=="2016-05-28"]
df_train.fillna(0, inplace=True)
# df_train=df_train[df_train['ncodpers'].isin(df_train_old["ncodpers"].tolist())].drop(['fecha_dato'],1)
# df_train_olclf.fit(x_train, y_train)d=df_train_old[~df_train_old['ncodpers'].isin(df_train["ncodpers"].tolist())].drop(['fecha_dato'],1)
df_train_old=df_train_old.append(df_train[~df_train['ncodpers'].isin(df_train_old["ncodpers"].tolist())])
df_train_old=df_train_old.drop(['fecha_dato'],1)

df_train= df_train.drop(['fecha_dato'],1)
df_train_old=df_train_old[df_train_old['ncodpers'].isin(df_train["ncodpers"].tolist())]
print("df_old_count"+ str(len(df_train_old)))
print("df_new_count"+ str(len(df_train)))

models = {}
model_preds = {}
id_preds = defaultdict(list)
ids = df_train_old['ncodpers'].values
for c in df_train.iloc[:,1:].columns:
    if c != 'ncodpers':
        print(c)
        y_train = df_train[c]
        x_train = df_train_old.drop([c, 'ncodpers'], 1)
        x_train_n = df_train.drop([c, 'ncodpers'], 1)
        clf = LogisticRegression()
        clf.fit(x_train.append(x_train_n), y_train.append(y_train))
        p_train = clf.predict_proba(x_train_n)[:,1]
        models[c] = clf
        model_preds[c] = p_train
        for id, p in zip(ids, p_train):
            id_preds[id].append(p)
        print(roc_auc_score(y_train, p_train))

already_active = {}
for row in df_train.values:
    row = list(row)
    id = row.pop(0)
    active = [c[0] for c in zip(df_train.columns[1:], row) if c[1] > 0]
    already_active[id] = active
    
train_preds = {}
for id, p in  id_preds.items():
    # Here be dragons
    preds = [i[0] for i in sorted([i for i in zip(df_train.columns[1:], p) if i[0] not in already_active[id]], key=lambda i:i [1], reverse=True)[:7]]
    train_preds[id] = preds
    
test_preds = []
for row in sample.values:
    id = row[0]
    p = train_preds[id]
    test_preds.append(' '.join(p))

sample['added_products'] = test_preds
sample.to_csv('collab_sub5.csv', index=False)