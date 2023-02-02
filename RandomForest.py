def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# A kaggle project I initially completed on codecademy
# https://www.kaggle.com/datasets/farhanmd29/position-salaries


income_data = pd.read_csv("income.csv", header = 0, delimiter = ", ")


income_data["sex-int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)

income_data["country-int"] = income_data["native-country"].apply(lambda row: 0 if row == "United-States" else 1)


labels = income_data[["income"]]
data = income_data[["age", "capital-gain","capital-loss","hours-per-week","sex-int"]]
train_data,test_data,train_labels,test_labels = train_test_split(data,labels,random_state =1)

forest = RandomForestClassifier(random_state =1)
tree1 = tree.DecisionTreeClassifier(random_state =1)
forest.fit(train_data,train_labels)
tree1.fit(train_data,train_labels)

print(forest.feature_importances_)
print(tree1.feature_importances_)

print(forest.score(test_data,test_labels))
print(tree1.score(test_data,test_labels))