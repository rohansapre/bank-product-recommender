import pandas as pd
import csv

inputFile = '../input/june_filter_data.csv'
# dtype_list = {'ind_cco_fin_ult1': 'float16', 'ind_deme_fin_ult1': 'float16', 'ind_aval_fin_ult1': 'float16', 'ind_valo_fin_ult1': 'float16', 'ind_reca_fin_ult1': 'float16', 'ind_ctju_fin_ult1': 'float16', 'ind_cder_fin_ult1': 'float16', 'ind_plan_fin_ult1': 'float16', 'ind_fond_fin_ult1': 'float16', 'ind_hip_fin_ult1': 'float16', 'ind_pres_fin_ult1': 'float16', 'ind_nomina_ult1': 'float16', 'ind_cno_fin_ult1': 'float16', 'ncodpers': 'int64', 'ind_ctpp_fin_ult1': 'float16', 'ind_ahor_fin_ult1': 'float16', 'ind_dela_fin_ult1': 'float16', 'ind_ecue_fin_ult1': 'float16', 'ind_nom_pens_ult1': 'float16', 'ind_recibo_ult1': 'float16', 'ind_deco_fin_ult1': 'float16', 'ind_tjcr_fin_ult1': 'float16', 'ind_ctop_fin_ult1': 'float16', 'ind_viv_fin_ult1': 'float16', 'ind_ctma_fin_ult1': 'float16'}
product_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

# data = pd.read_csv(inputFile, usecols=['fecha_dato', 'ncodpers']+product_cols, dtype=dtype_list)

mayData = {}
juneData = {}
customers = {}

with open(inputFile) as csvFile:
	reader = csv.DictReader(csvFile)
	for row in reader:
		total = 0.0
		if row['fecha_dato'] == '2015-05-28':
			for product in product_cols:
				try:
					total += float(row[product])
				except ValueError:
					pass
			mayData[row['ncodpers']] = total
		elif row['fecha_dato'] == '2015-06-28':
			for product in product_cols:
				try:
					total += float(row[product])
				except ValueError:
					pass
			juneData[row['ncodpers']] = total

for key in mayData:
	if key in juneData and juneData[key] > mayData[key]:
		customers[key] = juneData[key] - mayData[key]
print len(customers)

outputFile = open('../input/adding_customer.csv', 'wb')
with open(inputFile) as csvFile:
	reader = csv.reader(csvFile)
	writer = csv.writer(outputFile, delimiter=',')
        writer.writerow(next(reader))
	for row in reader:
		if row[0] == '2015-06-28' and row[1] in customers:
			for i in xrange(int(customers[row[1]])):
				writer.writerow(row)
