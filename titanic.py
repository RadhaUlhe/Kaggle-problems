import pandas as pd
import sys
import io
import os
import scipy.stats as stats
from scipy.stats import chi2_contingency
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import re
import tensorflow as tf
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import xgboost as xgb


class chiSquare :
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None
        self.chi_test_stat = None
        self.dof = None

        self.dfObserved = None
        self.dfExpected = None

    def print_result(self, colX, alpha):

        if self.p < alpha:
            print("{0} is important for prediction".format(colX))
        else:
            print("discard {0}".format(colX))




    def testIndependence(self, colX, colY, alpha= 0.05):

        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)

        self.dfObserved = pd.crosstab(Y, X)
        ch2, p, dof, expected = chi2_contingency(self.dfObserved.values)
        self.chi_test_stat = ch2
        self.p = p
        self.dof = dof
        self.dfExpected = expected

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index=self.dfObserved.index)

        self.print_result(colX, alpha)


class decision_tree:

    def __init__(self, x_dataframe, y_dataframe):
        self.xDf = x_dataframe
        self.yDf = y_dataframe
        #self.y_predict = None
        self.classifier_gini = None
        self.classifier_entropy = None
        self.accuracy = None
        self.confusionMat = None
    def train_using_gini(self):

        self.classifier_gini = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=100)

        self.classifier_gini.fit(self.xDf, self.yDf)

        return self.classifier_gini

    def train_using_entropy(self):

        self.classifier_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=100)

        self.classifier_entropy.fit(self.xDf, self.yDf)

        return self.classifier_entropy

    def prediction(self, df):


        y_predict_gini = self.classifier_gini.predict(df)
        y_predict_entropy = self.classifier_entropy.predict(df)

        return y_predict_gini, y_predict_entropy


    def cal_accuracy(self, df, y_predict_gini, y_predict_entropy):

        #self.confusionMat = confusion_matrix(df, y_predict_gini)
        #print(self.confusionMat)

        accuracy = accuracy_score(df, y_predict_gini)
        print("gini accuracy ::{0}".format(accuracy))
        accuracy = accuracy_score(df, y_predict_entropy)
        print("entropy accuracy ::{0}".format(accuracy))


class rF:

    def __init__(self, xDf, yDf):
        self.x = xDf
        self.y = yDf
        self.model = None

    def build_randomForest(self):

        self.model = RandomForestClassifier(n_estimators=50, max_features=3, max_depth=3)
        self.model.fit(self.x, self.y)
        self.x = pd.DataFrame(self.x)
        fi = pd.DataFrame({'feature': list(self.x.columns),
                           'importance': self.model.feature_importances_}).\
            sort_values('importance', ascending=False)
        print(fi)

    def predict_randomForest(self, df):

        predicted_y = self.model.predict(df)
        return predicted_y

    def cal_accuracy(self, y_predicted, y_true):

        accuracy = accuracy_score(y_true, y_predicted)
        print("accuracy {0}:".format(accuracy))


class NeuralNet:

    def __init__(self, seed):

        # set variables
        self.input_num_units = 6
        self.hidden_num_units = 3
        self.output_num_units = 2

        self.x = tf.placeholder(tf.float32, shape=[None, self.input_num_units])
        self.y = tf.placeholder(tf.float32, shape=[None, self.output_num_units])

        #self.y_one_hot = tf.placeholder(tf.float32, shape=[None, 2])
        self.epochs = 50
        self.batch_size = 150
        self.learning_rate = 0.1

        # set weights and biases
        self.weights = {'hidden': tf.Variable(tf.random_normal([self.input_num_units, self.hidden_num_units], seed=seed)),
                        'output': tf.Variable(tf.random_normal([self.hidden_num_units, self.output_num_units], seed=seed))}

        self.biases = {'hidden': tf.Variable(tf.random_normal([self.hidden_num_units], seed=seed)),
                       'output': tf.Variable(tf.random_normal([self.output_num_units], seed=seed))}

        # create neural net computational graph
        self.hidden_layer = tf.add(tf.matmul(self.x, self.weights['hidden']), self.biases["hidden"])
        self.hidden_layer = tf.nn.relu(self.hidden_layer)
        self.output_layer = tf.add(tf.matmul(self.hidden_layer, self.weights['output']), self.biases['output'])

        # define cost of NN
        #self.cost = tf.reduce_mean(tf.squared_difference(self.y, self.output_layer))
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.output_layer))

        # define optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def dense_to_one_hot_encoder(self, yDf):

        # binary encode
        label_encoder = preprocessing.LabelEncoder()
        integer_encoded = label_encoder.fit_transform(yDf)

        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


        return onehot_encoded

    def dense_to_one_hot(seif, labels_dense, num_classes=2):
        """Convert class labels from scalars to one-hot vectors"""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    def batch_creator(self, dataset_len, xDf, yDf, rng):

        batch_mask = rng.choice(dataset_len, self.batch_size)

        batch_x = xDf[[batch_mask]].reshape(-1, self.input_num_units)
        batch_y = yDf[[batch_mask]]
        #batch_y = yDf[[batch_mask]].reshape(-1, self.output_num_units)
        #batch_y = self.dense_to_one_hot_encoder(batch_y)
        batch_y = self.dense_to_one_hot(batch_y)
        return batch_x, batch_y


    def run_session(self, xDf, yDf, xDf_valid, yDf_valid, rng):

        # initialize NN
        self.initialise = tf.global_variables_initializer()

        with tf.Session() as self.sess:
            self.sess.run(self.initialise)

            for epoch in range(self.epochs):

                avg_cost = 0
                total_batch = int(len(xDf)/self.batch_size)

                for i in range(total_batch):
                    batch_x, batch_y = self.batch_creator(len(xDf), xDf, yDf, rng)
                    _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x: batch_x, self.y: batch_y})

                    avg_cost += c/total_batch

                print("epoch :", epoch+1, "cost :", "{:.5f}".format(avg_cost))


            print("Training completed!")

            pred_temp = tf.equal(tf.argmax(self.output_layer, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))

            #print(len(xDf.reshape(-1,self.input_num_units)))
            #print(len(yDf.reshape(-1, self.output_num_units)))

            acc = self.sess.run(accuracy, feed_dict={self.x:xDf.reshape(-1,self.input_num_units), self.y:self.dense_to_one_hot(yDf)})
            #acc = self.sess.run(accuracy, feed_dict={self.x: xDf.reshape(-1, self.input_num_units), self.y: yDf.reshape(-1, self.output_num_units)})

            #acc = accuracy.eval(
            #    {self.x: xDf.reshape(-1, self.input_num_units), self.y: yDf.reshape(-1, self.output_num_units)})


            print("accuracy on traiing set :", acc)

            y_temp = self.dense_to_one_hot(yDf_valid)
            #y_temp = y_temp.reshape(-1,self.output_num_units)
            acc = self.sess.run(accuracy, feed_dict={self.x: xDf_valid.reshape(-1, self.input_num_units),
                                                     self.y: y_temp})


            print("accuracy on validation set :", acc)


class boost:
    def __init__(self, xDf, yDf):
        self.x = xDf
        self.y = yDf
        self.abc = None

    def build_model(self):
        #svc = SVC(probability=True, kernel='linear')
        self.abc = AdaBoostClassifier()
        self.abc.fit(self.x, self.y)

    def predict_model(self, xDf):
        y_pred = self.abc.predict(xDf)
        return y_pred

    def cal_accuracy(self, y_hat, yDf):

        print("Accuracy :", accuracy_score(yDf, y_hat))

class xgboost:
    def __init__(self, xDf, yDf):
        self.x = xDf
        self.y = yDf
        self.xgboost_model = None

    def build_model(self):

        self.xgboost_model = xgb.DMatrix(data=self.x, label=self.y)

        self.xgboost_model = xgb.XGBClassifier(colsample_bytree=0.3, subsample=0.6, objective="reg:logistic", gamma=0.4)

        self.xgboost_model.fit(self.x, self.y)

    def predict_model(self, xDf):
        y_predict = self.xgboost_model.predict(xDf)
        return y_predict

    def cal_accuracy(self, y_hat, yDf):

        print("accuracy:", accuracy_score(yDf, y_hat))




baseDir = os.path.dirname(os.path.realpath(__file__))
trainDatafile = os.path.join(baseDir, "train.csv")
testDatafile = os.path.join(baseDir, "test.csv")

train_data_original = pd.read_csv(trainDatafile)
test_data_original = pd.read_csv(testDatafile)

print(train_data_original.dtypes)
#print(train_data.shape)
#print(test_data.shape)

######  Finding the relationship between independent and target variables  #######

train_data = train_data_original.copy()
train_data["dummy_var"] = np.random.choice([0,1], size=len(train_data), p=[0.5, 0.5])

# Inintialise chi-square class
cT = chiSquare(train_data)

# run through columes to find the relationship
test_columns = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
train_data.Cabin.fillna(value="0", inplace=True)

for idx, row in train_data.iterrows():
    train_data.at[idx, 'Cabin'] = re.sub("[^a-zA-Z]+", "", train_data.at[idx, 'Cabin'])


for col in test_columns:
   cT.testIndependence(colX=col, colY="Survived")


####### split the data set ##################
y_train = train_data_original["Survived"].values
x_train = train_data_original
# removing target attribute
x_train = x_train.drop(columns="Survived")

x_train = x_train.fillna({"Embarked": "S", "Fare": 0.0, "Cabin": "0"})

for idx, row in x_train.iterrows():
    x_train.at[idx, 'Cabin'] = re.sub("[^a-zA-Z]+", "", x_train.at[idx, 'Cabin'])
    x_train.at[idx, 'Cabin'] = x_train.at[idx, 'Cabin'][0:1]

#embarked_num = {"Embarked": {"S": 1, "Q": 2, "C": 3}}
#x_train.replace(embarked_num, inplace=True)

# removing attributes which are redundant found in chi-square test
x_train = x_train.drop(columns={"PassengerId", "Name", "Age", "Ticket", "Fare"})
le = preprocessing.LabelEncoder()
col = ["Sex", "Embarked", "Cabin"]
Sex_cat = le.fit_transform(x_train.Sex.values)
x_train["Sex_cat"] = Sex_cat
Embarked_cat = le.fit_transform(x_train.Embarked.values)
x_train['Embarked_cat'] = Embarked_cat
Cabin_cat = le.fit_transform(x_train.Cabin.values)
x_train['Cabin_cat'] = Cabin_cat

x_train.drop(columns=["Sex", "Embarked", "Cabin"], inplace=True)


#x_train[:, i] = le.fit_transform(x_train[:, i])
x_train = x_train.values

#print(x_train.shape)
#print(y_train.shape)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=100)

x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5, random_state=100)

#print(x_test.shape)
#print(y_test.shape)
#print(x_valid.shape)
#print(y_valid.shape)


############# Building a model DT model ###################


dT = decision_tree(x_train, y_train)
gini_classifier = dT.train_using_gini()
entropy_classifier = dT.train_using_entropy()
print("############ DECISION TREE #############")
############ Testing a model on training set #########
print("*******************************")
y_predict_gini, y_predict_entropy = dT.prediction(x_train)
print("accuracy for train set:")
dT.cal_accuracy(y_train, y_predict_gini, y_predict_entropy)

######### Testing a model on vaidation set ########

y_predict_gini, y_predict_entropy = dT.prediction(x_valid)
print("accuracy for validation set:")
dT.cal_accuracy(y_valid, y_predict_gini, y_predict_entropy)

print("*******************************")

############# Building a Random forest model ########################

randomForest = rF(x_train, y_train)
randomForest.build_randomForest()

# predicting a model on train data
y_predicted = randomForest.predict_randomForest(x_train)

print("########## RANDOM FOREST #############")
# calculate acuracy
print("accuracy on train data :")
randomForest.cal_accuracy(y_predicted, y_train)

# predicting model o validation set
y_predicted = randomForest.predict_randomForest(x_valid)
# calculate acuracy
print("accuracy on validation data :")
randomForest.cal_accuracy(y_predicted, y_valid)


############# building a neural net mode ###################

print("########## NEURAL NET #############")

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)

nn = NeuralNet(seed)
nn.run_session(x_train, y_train, x_valid, y_valid, rng)


########### building a boosting model using random forest ###########


boosting = boost(x_train, y_train)
boosting.build_model()

y_predicted = boosting.predict_model(x_train)

print("######## BOOSTING algorithm #############")
## Accuracy on training set #######
print("Training set")
boosting.cal_accuracy(y_predicted, y_train)

### Accuracy on validation set ######
y_predicted = boosting.predict_model(x_valid)
print("validation set")
boosting.cal_accuracy(y_predicted, y_valid)


########### building a boosting model using xgbbost ###########

xgboosting = xgboost(x_train, y_train)
xgboosting.build_model()

print("########## XGBOOST ###########")
### Accuracy on training set ######
y_predicted = xgboosting.predict_model(x_train)

print("Training set")
xgboosting.cal_accuracy(y_predicted, y_train)

print("validation set")
y_predicted = xgboosting.predict_model(x_valid)
xgboosting.cal_accuracy(y_predicted, y_valid)
