import findspark
findspark.init()
findspark.find()
import pyspark
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext, SparkConf
from pyspark.ml import Pipeline
import numpy as np

conf = pyspark.SparkConf().setAppName('winequality').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

hi = spark.read.format("csv").load("TrainingDataset.csv" , header = True ,sep =";")
hi.show()

for col_name in hi.columns[1:-1]+['""""quality"""""']:
    hi = hi.withColumn(col_name, col(col_name).cast('float'))
hi = hi.withColumnRenamed('""""quality"""""', "label")

features =np.array(hi.select(hi.columns[1:-1]).collect())
label = np.array(hi.select('label').collect())

VectorAssembler = VectorAssembler(inputCols = hi.columns[1:-1] , outputCol = 'features')
hi_tr = VectorAssembler.transform(hi)
hi_tr = hi_tr.select(['features','label'])

def to_labeled_point(sc, features, labels, categorical=False):
    labeled_points = []
    for x, y in zip(features, labels):        
        lp = LabeledPoint(y, x)
        labeled_points.append(lp)
    return sc.parallelize(labeled_points)

dataset = to_labeled_point(sc, features, label)

training, test = dataset.randomSplit([0.9, 0.1],seed =11)

RFmodel = RandomForest.trainClassifier(training, numClasses=10, categoricalFeaturesInfo={}, numTrees=21, featureSubsetStrategy="auto", impurity='gini', maxDepth=30, maxBins=32)

predictions = RFmodel.predict(test.map(lambda x: x.features))

labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)

labelsAndPredictions_hi = labelsAndPredictions.toDF()

pred = labelsAndPredictions.toDF(["label", "Prediction"])
pred.show()
pred_hi = pred.toPandas()

score = f1_score(pred_hi['label'], pred_hi['Prediction'], average='micro')
print("F1 score of the model is : ")
print(score)
print("Accuracy of the model is: - " , accuracy_score(pred_hi['label'], pred_hi['Prediction']))
print(confusion_matrix(pred_hi['label'],pred_hi['Prediction']))
print(classification_report(pred_hi['label'],pred_hi['Prediction']))

error = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test.count())
print('Error = ' + str(error))

RFmodel.save(sc, 's3://aws-logs-766621730595-us-east-1/elasticmapreduce/j-3NKDKZSBHWW3/trainingmodel.model')