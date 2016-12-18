import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file = pd.read_csv(str(sys.argv[1]) + ".csv",dtype={ "indrel_1mes": 'category',"segmento" :'category', "ind_ahor_fin_ult1": int,"ind_aval_fin_ult1": int,"ind_cco_fin_ult1": int,"ind_cder_fin_ult1": int,"ind_cno_fin_ult1": int,"ind_ctju_fin_ult1": int,"ind_ctma_fin_ult1": int,"ind_ctop_fin_ult1": int,"ind_ctpp_fin_ult1": int,"ind_deco_fin_ult1": int,"ind_deme_fin_ult1": int,"ind_dela_fin_ult1": int,"ind_ecue_fin_ult1": int,"ind_fond_fin_ult1": int,"ind_hip_fin_ult1": int,"ind_plan_fin_ult1": int,"ind_pres_fin_ult1": int,"ind_reca_fin_ult1": int,"ind_tjcr_fin_ult1": int,"ind_valo_fin_ult1": int,"ind_viv_fin_ult1": int,"ind_nomina_ult1": int,"ind_nom_pens_ult1": int,"ind_recibo_ult1": int}, low_memory=False)

#creating a temporary copy of the read file
modifiedData=file[file["fecha_dato"]==str(sys.argv[2])] 

# modifiedData["sex"]=float('Nan')
#factorising the sex into two categories
# modifiedData["sex"][modifiedData["sexo"]=="V"]=1
# modifiedData["sex"][modifiedData["sexo"]=="H"]=0

# indrel_1mes which tells customer type, assigning a value of 5 to P, a potential customer
# modifiedData["indrel_1mes"][modifiedData["indrel_1mes"]=="P"]=5

print(modifiedData.head())
corr_file = modifiedData.corr(method='pearson')
print corr_file
	
plt.matshow(corr_file)
# plt.savefig('date1train.png')
plt.savefig("corr/" + str(sys.argv[1]) + '.png')
# plt.show()
plt.clf()

plt.figure(figsize=(15,10))
rowCount=modifiedData[modifiedData.columns[24]]

# productCount= [x * rowCount for x in modifiedData.iloc[:, 24:].mean()] 
(modifiedData.iloc[:, 24:].mean()).plot('barh')
# productCount.plot('barh')
plt.savefig("barGraph/" + str(sys.argv[1]) +'_bar'+ '.png', dpi=100)

# alternatively we can get just the mean which could also be considered as a probability distribution across all the products.
# we can find it using modifiedData[modifiedData.columns[24:]].mean()
# a fixed x-axis (currently it changes dynamically depending on the value of x axis) has to be defined for either of the two measures
# the count bar graph would help us understand the chnge in the number of products, a difference between subseuent 
#    bar graphs would give us an estimate which products are in demand

#  Also, if subsequent bar graphs are only considered for the customers the first monnth (current month) , we would understand whether
#       our current customers are interested in some other products
