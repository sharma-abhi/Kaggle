# A model for prediction survival on the Titanic based on where an 
# individual Embarked, their gender, or the class they traveled in. 
# AGC 2013
# 
# 
# Here Will will run generate predictions of who survived and who did not
# from our basic Least Squares Regression model.
# Our Formula is :
# survived_prediction = C + pclass + sex + age + sibsp  + embarked

# Import Utilities
import csv as csv
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
#import statsmodels.api as sm
#import predict as ka
from patsy import dmatrices
from sklearn import datasets, svm


df = pd.read_csv("../data/train.csv") 
df = df.drop(['Ticket','Cabin'], axis=1)
# Remove NaN values
df = df.dropna() 

# here the ~ sign is an = sign, and the features of our dataset
# are written as a formula to predict survived. The C() lets our 
# regression know that those variables are categorical.
formula_ml = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp  + Parch + C(Embarked)' 

# create a results dictionary to hold our regression results for easy analysis later        
results = {} 

# Create a regression friendly version of our data using dmatrices
y, x = dmatrices(formula_ml, data=df, return_type='matrix')

feature_1 = 2
feature_2 = 3

X = np.asarray(x)
X = X[:,[feature_1, feature_2]]  
y = np.asarray(y)
y = y.flatten()  

n_sample = len(X)
np.random.seed(0)
order = np.random.permutation(n_sample)

X = X[order]
y = y[order].astype(np.float)

# do a cross validation
ninety_precent_of_sample = int(.9 * n_sample)
X_train = X[:ninety_precent_of_sample]
y_train = y[:ninety_precent_of_sample]
X_test = X[ninety_precent_of_sample:]
y_test = y[ninety_precent_of_sample:]

# create a list of the types of kernels we will use for your analysis
'''
#types_of_kernels = ['linear', 'rbf', 'poly']
# specify our color map for plotting the results
color_map = plt.cm.RdBu_r

# fit the model
for fig_num, kernel in enumerate(types_of_kernels):
    clf = svm.SVC(kernel=kernel, gamma=3)
    clf.fit(X_train, y_train)

    plt.figure(fig_num)
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=color_map)

    # circle out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)
    
    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=color_map)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
               levels=[-.5, 0, .5])

    plt.title(kernel)
    plt.show()'''

clf = svm.SVC(kernel='poly', gamma=3).fit(X_train, y_train) 
###################################################################################################
test_data = pd.read_csv("../data/test.csv") 
t2 = [i for i in test_data['PassengerId']]
test_data = test_data.drop(['Ticket','Cabin'], axis=1)
# Remove NaN values
test_data = test_data.dropna() 
test_data['Survived'] = 1.23

y,x = dmatrices(formula_ml, data=test_data, return_type='dataframe')                                                           

# Change the interger values within x.ix[:,[6,3]].dropna() explore the relationships between other 
# features. the ints are column postions. ie. [6,3] 6th column and the third column are evaluated. 
res_svm = clf.predict(x.ix[:,[2,3]].dropna()) 
res_svm = DataFrame(res_svm,columns=['Survived'])
#res_svm.to_csv("../data/output/svm_poly_63_g10.csv") # saves the results for you, change the name as you please. 
#print res_svm


d = {}
t1 = [i for i in test_data['PassengerId']]
p = [i for i in res_svm['Survived']]
for i in range(len(t1)):
	if p[i] > 0.5:
		d[t1[i]] = 1
	else:
		d[t1[i]] = 0
	#d[t1[i]] = p[i]

for j in t2:
	if j not in t1:
		d[j] = 0

df2 = pd.DataFrame(d.items())

df2.to_csv("../data/output/svm_poly_23_g10.csv")

print "Analysis ended"
