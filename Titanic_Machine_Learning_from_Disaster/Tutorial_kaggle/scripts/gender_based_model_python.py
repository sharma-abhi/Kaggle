import csv as csv
import numpy as np

csv_file_object = csv.reader(open('data/train.csv','rb'))
header = csv_file_object.next()


data = []
for row in csv_file_object:
	data.append(row)

data = np.array(data)

number_passengers = np.size(data[::,1].astype(np.float))
number_survived = np.sum(data[::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

women_only_stats = data[0::,4] == "female"
men_only_stats = data[0::,4] != "female"
# Using the index from above we select the females and males separately
women_onboard = data[women_only_stats,1].astype(np.float)     
men_onboard = data[men_only_stats,1].astype(np.float)

# Then we finds the proportions of them that survived
proportion_women_survived = \
                       np.sum(women_onboard) / np.size(women_onboard)  
proportion_men_survived = \
                       np.sum(men_onboard) / np.size(men_onboard) 

# and then print it out
print 'Proportion of women who survived is %s' % proportion_women_survived
print 'Proportion of men who survived is %s' % proportion_men_survived


test_file = open('data/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

prediction_file = open("data/output/genderbasedmodel_python.csv", "wb")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])

for row in test_file_object:       # For each row in test.csv
    if row[3] == 'female':         # is it a female, if yes then                                       
        prediction_file_object.writerow([row[0],'1'])    # predict 1
    else:                              # or else if male,       
        prediction_file_object.writerow([row[0],'0'])    # predict 0
test_file.close()
prediction_file.close()
