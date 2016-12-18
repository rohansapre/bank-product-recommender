import sys
import numpy as np
import pandas as pd
import csv
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense, Activation

categorical_features = ['ind_empleado', 'pais_residencia', 'sexo', 'tiprel_1mes', 'indresi', 'indext', 'conyuemp', 'canal_entrada', 'indfall']

map_dict = {
'ind_empleado'  : {'N':0, -99:1, 'B':2, 'F':3, 'A':4, 'S':5},
'sexo'          : {'V':0, 'H':1, -99:2},
'ind_nuevo'     : {0.0:0, 1.0:1, -99.0:2},
'indrel'        : {1.0:0, 99.0:1, -99.0:2},
'indrel_1mes'   : {-99:0, 1.0:1, 1:1, 2.0:2, 2:2, 3.0:3, 3:3, 4.0:4, 4:4, 'P':5},
'tiprel_1mes'   : {-99:0, 'I':1, 'A':2, 'P':3, 'R':4, 'N':5},
'indresi'       : {-99:0, 'S':1, 'N':2},
'indext'        : {-99:0, 'S':1, 'N':2},
'conyuemp'      : {-99:0, 'S':1, 'N':2},
'indfall'       : {-99:0, 'S':1, 'N':2},
'tipodom'       : {-99.0:0, 1.0:1},
'ind_actividad_cliente' : {0.0:0, 1.0:1, -99.0:2},
'segmento'      : {'02 - PARTICULARES':0, '03 - UNIVERSITARIO':1, '01 - TOP':2, -99:2},
'pais_residencia' : {'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17, 'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73, 'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100, 'HR': 67, 'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20, 'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90, 'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 'MT': 118, 'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7, 'NO': 46, 'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4, 'CA': 2, 'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32, 'CU': 72, 'KE': 65, 'KH': 95, 'SV': 53, 'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66, 'SE': 24, 'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105, -99: 1, 'LB': 81, 'TW': 29, 'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37, 'VE': 14, 'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5, 'QA': 58, 'MZ': 27},
'canal_entrada' : {'013': 49, 'KHP': 160, 'KHQ': 157, 'KHR': 161, 'KHS': 162, 'KHK': 10, 'KHL': 0, 'KHM': 12, 'KHN': 21, 'KHO': 13, 'KHA': 22, 'KHC': 9, 'KHD': 2, 'KHE': 1, 'KHF': 19, '025': 159, 'KAC': 57, 'KAB': 28, 'KAA': 39, 'KAG': 26, 'KAF': 23, 'KAE': 30, 'KAD': 16, 'KAK': 51, 'KAJ': 41, 'KAI': 35, 'KAH': 31, 'KAO': 94, 'KAN': 110, 'KAM': 107, 'KAL': 74, 'KAS': 70, 'KAR': 32, 'KAQ': 37, 'KAP': 46, 'KAW': 76, 'KAV': 139, 'KAU': 142, 'KAT': 5, 'KAZ': 7, 'KAY': 54, 'KBJ': 133, 'KBH': 90, 'KBN': 122, 'KBO': 64, 'KBL': 88, 'KBM': 135, 'KBB': 131, 'KBF': 102, 'KBG': 17, 'KBD': 109, 'KBE': 119, 'KBZ': 67, 'KBX': 116, 'KBY': 111, 'KBR': 101, 'KBS': 118, 'KBP': 121, 'KBQ': 62, 'KBV': 100, 'KBW': 114, 'KBU': 55, 'KCE': 86, 'KCD': 85, 'KCG': 59, 'KCF': 105, 'KCA': 73, 'KCC': 29, 'KCB': 78, 'KCM': 82, 'KCL': 53, 'KCO': 104, 'KCN': 81, 'KCI': 65, 'KCH': 84, 'KCK': 52, 'KCJ': 156, 'KCU': 115, 'KCT': 112, 'KCV': 106, 'KCQ': 154, 'KCP': 129, 'KCS': 77, 'KCR': 153, 'KCX': 120, 'RED': 8, 'KDL': 158, 'KDM': 130, 'KDN': 151, 'KDO': 60, 'KDH': 14, 'KDI': 150, 'KDD': 113, 'KDE': 47, 'KDF': 127, 'KDG': 126, 'KDA': 63, 'KDB': 117, 'KDC': 75, 'KDX': 69, 'KDY': 61, 'KDZ': 99, 'KDT': 58, 'KDU': 79, 'KDV': 91, 'KDW': 132, 'KDP': 103, 'KDQ': 80, 'KDR': 56, 'KDS': 124, 'K00': 50, 'KEO': 96, 'KEN': 137, 'KEM': 155, 'KEL': 125, 'KEK': 145, 'KEJ': 95, 'KEI': 97, 'KEH': 15, 'KEG': 136, 'KEF': 128, 'KEE': 152, 'KED': 143, 'KEC': 66, 'KEB': 123, 'KEA': 89, 'KEZ': 108, 'KEY': 93, 'KEW': 98, 'KEV': 87, 'KEU': 72, 'KES': 68, 'KEQ': 138, -99: 6, 'KFV': 48, 'KFT': 92, 'KFU': 36, 'KFR': 144, 'KFS': 38, 'KFP': 40, 'KFF': 45, 'KFG': 27, 'KFD': 25, 'KFE': 148, 'KFB': 146, 'KFC': 4, 'KFA': 3, 'KFN': 42, 'KFL': 34, 'KFM': 141, 'KFJ': 33, 'KFK': 20, 'KFH': 140, 'KFI': 134, '007': 71, '004': 83, 'KGU': 149, 'KGW': 147, 'KGV': 43, 'KGY': 44, 'KGX': 24, 'KGC': 18, 'KGN': 11},
}

# index = 0
# with open(featureFile) as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
#     for line in reader:
#         map_dict[categorical_features[index]] = {}
#         flag = False
#         if categorical_features[index] in ['sexo', 'segmento']:
#             k = 0
#         else:
#             k = 1
#             flag = True
#         for key in line:
#             try:
#                 key = float(key)
#             except (ValueError, TypeError):
#                 pass
#             map_dict[categorical_features[index]][key] = k
#             k += 1
#         if flag:
#             map_dict[categorical_features[index]][-99] = 0
#         else:
#             map_dict[categorical_features[index]][-99] = k
#         index += 1
# map_dict['segmento'] = {'02 - PARTICULARES':0, '03 - UNIVERSITARIO':1, '01 - TOP':2, -99:2}
# map_dict['ind_nuevo'] = {0.0 : 0, 1.0 : 1, -99.0 : 2}
# map_dict['ind_actividad_cliente'] = {0.0 : 1, 1.0 : 1, -99.0 : 2}
# map_dict['indrel_1mes'] = {-99:0, 1.0:1, 1:1, 2.0:2, 2:2, 3.0:3, 3:3, 4.0:4, 4:4, 'P':5}
# map_dict['indrel'] = {1.0:0, 99.0:1, -99.0:2}
# map_dict['tipodom'] = {-99.0:0, 1.0:1}

# for key,value in map_dict.iteritems():
#     print key, value

categorical_features = list(map_dict.keys())

dtype_list = {'ind_cco_fin_ult1': 'float16', 'ind_deme_fin_ult1': 'float16', 'ind_aval_fin_ult1': 'float16', 'ind_valo_fin_ult1': 'float16', 'ind_reca_fin_ult1': 'float16', 'ind_ctju_fin_ult1': 'float16', 'ind_cder_fin_ult1': 'float16', 'ind_plan_fin_ult1': 'float16', 'ind_fond_fin_ult1': 'float16', 'ind_hip_fin_ult1': 'float16', 'ind_pres_fin_ult1': 'float16', 'ind_nomina_ult1': 'float16', 'ind_cno_fin_ult1': 'float16', 'ncodpers': 'int64', 'ind_ctpp_fin_ult1': 'float16', 'ind_ahor_fin_ult1': 'float16', 'ind_dela_fin_ult1': 'float16', 'ind_ecue_fin_ult1': 'float16', 'ind_nom_pens_ult1': 'float16', 'ind_recibo_ult1': 'float16', 'ind_deco_fin_ult1': 'float16', 'ind_tjcr_fin_ult1': 'float16', 'ind_ctop_fin_ult1': 'float16', 'ind_viv_fin_ult1': 'float16', 'ind_ctma_fin_ult1': 'float16'}

numerical_features = ['age', 'antiguedad', 'renta']
min_vals = np.array([-1., -999999., -1.])
range_vals = np.array([1.65000000e+02, 1.00025500e+06, 2.88943965e+07])

product_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

encs = []
total_features = 0

for feat in categorical_features:
    ohe = preprocessing.OneHotEncoder()
    values = list(map_dict[feat].values())
    ohe.fit(np.array(values).reshape(-1,1))
    total_features += ohe.n_values_[0]
    encs.append(ohe)
total_features += len(numerical_features)

#print map_dict
meanVals = {'age': 40.0, 'antiguedad': 0.0, 'renta': 101850.0}

def batch_generator(csvfile, size, train):
    while(True):
        columns = ['ncodpers']
        if train:
            columns += categorical_features + numerical_features + product_cols
        else:
            columns += categorical_features + numerical_features
        batch = pd.read_csv(csvfile, usecols=columns, chunksize=size)
        rows = 0
        for item in batch:
            itemX = item[categorical_features]
            itemX = itemX.fillna(-99)
            for ci, col in enumerate(categorical_features):
                itemX[col] = itemX[col].apply(lambda x: map_dict[col][x])
                ohe = encs[ci]
                tempX = ohe.transform(np.array(itemX[col]).reshape(-1,1))
                X = tempX.todense().copy() if ci == 0 else np.hstack((X, tempX.todense()))

            itemX = item[numerical_features]

            for i, col in enumerate(numerical_features):
                if itemX[col].dtype == 'object':
                    itemX[col] = itemX[col].map(str.strip).replace(['NA'], value=meanVals[col]).fillna(meanVals[col]).astype('float64')
                else:
                    itemX[col] = itemX[col].fillna(meanVals[col]).astype('float64')
                itemX[col] = (itemX[col] - min_vals[i]) / range_vals[i]

            itemX = np.array(itemX).astype('float64')
            X = np.hstack((X, itemX))

            if train:
                y = np.array(item[product_cols].fillna(0))

            if train:
                yield X, y
            else:
                yield X

            rows += size
            if train and rows >= size:
                break

def init_model():
    model = Sequential()
    model.add(Dense(50, input_dim=total_features, init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Dense(len(product_cols), init='zero'))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

if __name__ == '__main__':
    train_data = 'input/train_ver3.csv'
    test_data = 'input/test_ver2.csv'
    train_size = 10000
    test_size = 931454
    model = init_model()
    fit = model.fit_generator(generator=batch_generator(train_data, 500, True), nb_epoch=1, samples_per_epoch=train_size)
    preds = model.predict_generator(generator=batch_generator(test_data, 10000, False), val_samples=test_size)
    print("Predictions : ", preds.shape)

    last_data = pd.read_csv(train_data, usecols=['ncodpers']+product_cols, dtype=dtype_list)
    last_data = last_data.drop_duplicates('ncodpers', keep='last')
    last_data = last_data.fillna(0).astype('int')

    cust_dict = {}
    product_cols = np.array(product_cols)
    for i, row in last_data.iterrows():
        cust = row['ncodpers']
        used_products = set(product_cols[np.array(row[1:])==1])
        cust_dict[cust] = used_products
    del last_data

    product_cols = np.array(product_cols)
    preds = np.argsort(preds, axis=1)
    preds = np.fliplr(preds)
    test_id = np.array(pd.read_csv(test_data, usecols=['ncodpers'])['ncodpers'])
    final_preds = []
    for i, pred in enumerate(preds):
        cust = test_id[i]
        top_products = product_cols[pred]
        used_products = cust_dict.get(cust,[])
        new_top_products = []
        for product in top_products:
            if product not in used_products:
                new_top_products.append(product)
                if len(new_top_products) == 7:
                    break
        final_preds.append(" ".join(new_top_products))
    out_data = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
    out_data.to_csv('may_2016.csv', index=False)
