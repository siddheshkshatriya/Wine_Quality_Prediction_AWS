import pyspark
import pandas as pd
import sys
from pyspark.mllib.linalg import Vectors
from pyspark.sql.session import SparkSession
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark import SparkContext, SparkConf
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import VectorAssembler
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

configuration = pyspark.SparkConf().setAppName('winequality').setMaster('local')
spark_context = pyspark.SparkContext(configuration=configuration)
spark = SparkSession(spark_context)
validation = spark.read.format("csv").load("ValidationDataset.csv" , header = True , sep=";")
validation.printSchema()
validation.show()

for col_name in validation.columns[1:-1]+['""""quality"""""']:
    validation = validation.withColumn(col_name, col(col_name).cast('float'))
validation = validation.withColumnRenamed('""""quality"""""', "label")

features =np.array(validation.select(validation.columns[1:-1]).collect())
label = np.array(validation.select('label').collect())

VectorAssembler = VectorAssembler(inputCols = validation.columns[1:-1] , outputCol = 'features')
df_tr = VectorAssembler.transform(validation)
df_tr = df_tr.select(['features','label'])

def labeledpoint(spark_context, features, labels, categorical=False):
    labeled_points = []
    for x, y in zip(features, labels):        
        lp = LabeledPoint(y, x)
        labeled_points.append(lp)
    return spark_context.parallelize(labeled_points)

dataset = labeledpoint(spark_context, features, label)

RFModel = RandomForestModel.load(spark_context, "/home/ec2-user/model/")

print("model loaded successfully")
predictions = RFModel.predict(dataset.map(lambda x: x.features))

labelsAndPredictions = dataset.map(lambda lp: lp.label).zip(predictions)
 
labelsAndPredictions_df = labelsAndPredictions.toDF()
pred = labelsAndPredictions.toDF(["label", "Prediction"])
pred.show()
pred_df = pred.toPandas()

score = f1_score(pred_df['label'], pred_df['Prediction'], average='micro')
print("F1 score of the model is : -", score)
print("Accuracy of the model is : -" , accuracy_score(pred_df['label'], pred_df['Prediction']))
print(confusion_matrix(pred_df['label'],pred_df['Prediction']))
print(classification_report(pred_df['label'],pred_df['Prediction']))

error = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(dataset.count())    
print('Error = ' + str(error))

