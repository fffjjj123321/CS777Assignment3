import sys
import re
import numpy as np
from numpy import dot
from numpy.linalg import norm
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext

# spark = SparkSession.builder.master("yarn").getOrCreate()
sc = SparkContext.getOrCreate()
# sqlContext = SQLContext(sc)

lines = sc.textFile(sys.argv[1])

taxilines = lines.map(lambda x:x.split(','))

def is_float(x):
    try:
        float(x)
        return True
    except:
        return False

def correct_rows(x):
    if len(x) == 17:
        if is_float(x[5]) and is_float(x[11]):
            amount = float(x[11])
            if float(x[5]) != 0 and amount >= 1 and amount <= 600:
                return x

taxilinesCorrected = taxilines.filter(correct_rows)


taxi_points = taxilinesCorrected.map(lambda x:(float(x[5]),float(x[11])))

taxi_points.persist()



learningRate = 0.001
num_iteration = 50

beta = [0.1, 0.1]

size = taxi_points.count()

for i in range(num_iteration):
    # sample = myRDD.sample(False, 100)
    
    gradientCost = taxi_points.map(lambda x: (x[1], (x[0] - np.dot(x[1] , beta) )))\
                           .map(lambda x: (x[0]*x[1], x[1]**2 )).reduce(lambda x, y: (x[0] +y[0], x[1]+y[1] ))
    
    cost = gradientCost[1]
    
    gradient = (-1/float(size))* gradientCost[0]
    
    print(i, "Beta", beta, " Cost", cost)
    beta = beta - learningRate * gradient


sc.stop()

