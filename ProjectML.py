import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dataset = pd.read_csv('StudentsPerformance.csv')
np.set_printoptions(linewidth=3278)
pd.set_option('display.width', 1278)

passingmarks = 55

print(dataset.shape)
print(dataset.describe())
print(dataset.dtypes)
attribute_names = list(dataset.columns.values)

print(dataset.isnull().sum())

#menentukan standar kelulusan
dataset['Math_PassStatus'] = np.where(dataset['math score']<passingmarks, 'F', 'P')
print(dataset.Math_PassStatus.value_counts())

dataset['Reading_PassStatus'] = np.where(dataset['reading score']<passingmarks, 'F', 'P')
print(dataset.Reading_PassStatus.value_counts())

dataset['Writing_PassStatus'] = np.where(dataset['writing score']<passingmarks, 'F', 'P')
print(dataset.Writing_PassStatus.value_counts())

dataset['OverAll_PassStatus'] = dataset.apply(lambda x : 'F' if x['Math_PassStatus'] == 'F' or
                                    x['Reading_PassStatus'] == 'F' or x['Writing_PassStatus'] == 'F' else 'P', axis =1)
print(dataset.OverAll_PassStatus.value_counts())
sns.countplot(x='parental level of education', data = dataset, hue='OverAll_PassStatus', palette='bright')
plt.show()
sns.heatmap(data=dataset.corr(),annot=True)

plt.title("Correlation Matrix of Students Performance")
plt.show()

sns.pairplot(data=dataset[['math score', 'reading score', 'writing score', 'OverAll_PassStatus']],
                 hue='OverAll_PassStatus')
plt.show()



dataset.drop(columns=['math score', 'reading score', 'writing score', 'Math_PassStatus', 'Reading_PassStatus', 'Writing_PassStatus'], inplace=True)
print(dataset.head())



one_hot = pd.get_dummies(dataset['gender'], prefix='gender', drop_first=True)
dataset = dataset.join(one_hot)
one_hot = pd.get_dummies(dataset['race/ethnicity'], prefix='race/ethnicity', drop_first=True)
dataset = dataset.join(one_hot)
one_hot = pd.get_dummies(dataset['parental level of education'], prefix='parental level of education', drop_first=True)
dataset = dataset.join(one_hot)
one_hot = pd.get_dummies(dataset['lunch'], prefix='lunch', drop_first=True)
dataset = dataset.join(one_hot)
one_hot = pd.get_dummies(dataset['test preparation course'], prefix='test preparation course', drop_first=True)
dataset = dataset.join(one_hot)
dataset.head()


#split dataset

data_train, data_test_hold = train_test_split(dataset, test_size=0.30, random_state=21)
data_test, data_hold = train_test_split(data_test_hold, test_size=0.33, random_state=21)

columns_move = ["gender", "race/ethnicity", "parental level of education", "lunch",
                "test preparation course", "gender_male", "race/ethnicity_group B",
                "race/ethnicity_group C", "race/ethnicity_group D", "race/ethnicity_group E",
                "parental level of education_bachelor's degree", "parental level of education_high school",
                "parental level of education_master's degree", "parental level of education_some college",
                "parental level of education_some high school", "lunch_standard", "test preparation course_none"]

y_train = data_train["OverAll_PassStatus"].values
X_train = data_train[columns_move].values
y_test = data_test["OverAll_PassStatus"].values
X_test = data_test[columns_move].values

model = DecisionTreeClassifier(criterion='gini', splitter='best',
                               max_depth=None, min_samples_split=2,
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                               max_features=None, random_state=None,
                               max_leaf_nodes=None, min_impurity_decrease=0.0,
                               min_impurity_split=None, class_weight=None,
                               presort=False)

#training model
model.fit(X_train[:,5:], y_train)

#evaluasi
y_pred = model.predict(X_test[:,5:])
print("Prediksi akurasi testing set: %.2f" % (accuracy_score(y_test,y_pred)*100), "%")
y_pred_train = model.predict(X_train[:,5:])
print("Prediksi akurasi training set: %.2f" % (accuracy_score(y_train,y_pred_train)*100), "%")

#confusion matrix
a = pd.DataFrame(confusion_matrix(y_test,y_pred), columns=['prediction/f', 'prediction/p'], index=['actual/f', 'actual/p'])
print("Confusion Matrix:")
print(a)

#classification report
print("Classification Report:")
print("")
print(classification_report(y_test,y_pred))

