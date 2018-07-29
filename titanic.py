import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, adam
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
# read in data
data_train = pd.read_csv('train.csv')
data_val = pd.read_csv('test.csv')

names_train = data_train['Name']
names_val = data_val['Name']
# drop columns that do not correlate with survival / have significant gaps
data_train = data_train.drop(['Cabin', 'Name', 'PassengerId', 'Fare', 'Embarked'], axis=1)
data_val = data_val.drop(['Cabin', 'Name', 'PassengerId', 'Fare', 'Embarked'], axis=1)
data = [data_train, data_val]

classes = ['Deceased', 'Survived']

def vec_to_description(person):
	description = ""
	alone = None
	ticket = None
	if person[0] == 1:
		sex = "Male"
	else:
		sex = "Female"
	if person[1] == 1:
		alone = "with large family"
	if person[2] == 1:
		alone = "alone"
	if person[3] == 1:
		age = "Child"
	if person[4] == 1:
		age = "Young adult"
	if person[5] == 1:
		age =  "Adult"
	if person[6] == 1:
		age = "Elderly"
	if person[7] == 1:
		ticket = "travelling in first class"
	if person[8] == 1:
		ticket = "travelling in second class"
	if person[9] == 1:
		ticket = "travelling in third class"

	if(ticket is None):
		ticket = "travelling in UNKNOWN"
	if(alone is None):
		alone = "with companion / or family with 4 members or less"
	description = ("{0}, {1}, {2} {3}").format(age, sex, ticket, alone)

	return description

for d in data:
	'''
	Preprocessing
	1. Map columns containing strings to binary numerical values
	   1 = male, 0 = female
	2. Identify and flag large families, families with 5 or more
	   members had significantly lower survival rates
	3. Identify and flag solo travellers, they had lower survival rates
	4. If age is not known for a person then populate with median
	5. Create age groups for child, young adult, adult and old
	6. Group by ticket class
	'''
	# Create new columns for relevant groups
	d['largeFam'] = 0
	d['Alone'] = 0
	d['Child'] = 0
	d['YoungA'] = 0
	d['Adult'] = 0
	d['Old'] = 0
	d['First_class'] = 0
	d['Second_class'] = 0
	d['Third_class'] =0

	# 1. Convert sex to  binary numerical values
	d['Sex'] = d['Sex'].map({ 'male': 1, 'female': 0}).astype(int)

	# 2. Identify + flag large groups/families
	d.loc[d['SibSp'] + d['Parch'] + 1 > 4, 'largeFam'] = 1
	
	# 3. Identify + flag solo travellers
	d.loc[d['SibSp'] + d['Parch'] + 1 == 1, 'Alone'] =1

	# 4. Set blank ages to median age
	d.loc[d.Age.isnull(), 'Age'] = d.Age.median()
	
	# 5. Group passengers by age
	d.loc[d['Age'] <= 16, 'Child'] = 1
	d.loc[(d['Age'] > 16) & (d['Age'] <= 30), 'YoungA'] = 1
	d.loc[(d['Age'] > 30) & (d['Age'] <= 60), 'Adult'] = 1
	d.loc[d['Age'] > 60, 'Old'] = 1

	#6. Group passengers by ticket class
	d.loc[d['Pclass'] == 1, 'First_class'] = 1
	d.loc[d['Pclass'] == 2, 'Second_class'] = 1
	d.loc[d['Pclass'] == 3, 'Third_class'] = 1


# Drop columns we don't need any more from training/validation set
# We've moved these into groups above
data_train = data_train.drop(['Ticket', 'Parch', 'SibSp', 'Age', 'Pclass'], axis=1)
data_val = data_val.drop(['Ticket', 'Parch', 'SibSp', 'Age', 'Pclass'], axis=1)

# Set target column to predict (Survived)
targets = np.array(to_categorical(data_train['Survived'], 2))

# Set features we are trying to correlate to target
features = np.array(data_train.drop('Survived', axis=1))
features_val = np.array(data_val)

# Set neural net architecture
# 2 simple fully connected layers with relu activation
# output layer will have 2 nodes (len classes) with softmax activation
# to provide probability score
model = Sequential()
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(.2))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(.2))
model.add(Dense(len(classes), activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])


model.fit(features, targets, epochs=100, batch_size=32, verbose=1)

p = model.predict(features_val)
for i, x in enumerate(p):
	n = names_val[i]
	d = features_val[i]
	d = vec_to_description(d)
	mortality = classes[np.argmax(x)]
	if (mortality == 'Deceased'):
		prob = x[0]
	else:
		prob = x[1]
	
	print("\n\n\n\n{0}\n________________________________________\nDescription: {1}".format(n, d)) 
	print("Prediction: {0} ({1:.1%} probability)".format(mortality, prob))

score = model.evaluate(features, targets)
print ("\nEstimated accuracy: {:.0%}".format(score[1]))
	
