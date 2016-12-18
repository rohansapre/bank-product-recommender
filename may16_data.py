import csv

mayFile = 'data/date17train.csv'
aprilFile = 'data/date16train.csv'

product_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

def getTarget(row):
    tlist = []
    for col in product_cols:
        if row[col].strip() in ['', 'NA']:
            target = 0
        else:
            target = int(float(row[col]))
        tlist.append(target)
    return tlist

def processData(in_file_name,cust_dict):
    x_vars_list = []
    y_vars_list = []
    customer_dict = {}

    for row in csv.DictReader(in_file_name):

        if row['fecha_dato'] not in ['2016-05-28', '2016-04-28']:
            continue

        cust_id = int(row['ncodpers'])
        if row['fecha_dato'] in ['2016-04-28', '2016-04-28']: 
            target_list = getTarget(row)
            cust_dict[cust_id] =  target_list[:]
            continue

        x_vars = []
        prod_list = []
        if row['fecha_dato'] == '2016-05-28':
            prev_target_list = cust_dict.get(cust_id, [0]*24)
            target_list = getTarget(row)
            new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list)]
            for i, prod in enumerate(product_cols):
                if new_products[i] == 1:
                    prod_list.append(prod)
            customer_dict[cust_id] = prod_list
    return customer_dict

def processPredictionData(in_file_name):
    customer_dict = {}

    for row in csv.DictReader(in_file_name):
        customer_dict[row['ncodpers']] = row['added_products'].split(' ')

    return customer_dict

def apk(actual, predicted, k=7):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted):
    total = 0
    count = 0.0
    for key in predicted:
        total += apk(actual[key], predicted[key])
        count += 1.0
    return total / count 

trueValues = processData(open('input/april_may_data.csv'), {})

predFile = open('sub_keras_v2.csv')

predValules = processPredictionData(predFile)

print mapk(trueValues, predValules)