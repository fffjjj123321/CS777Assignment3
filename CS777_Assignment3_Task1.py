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

n = taxi_points.count()
print("This is n: {}".format(n))
sumOfX = taxi_points.keys().sum()
print("This is x: {}".format(sumOfX))
sumOfY = taxi_points.values().sum()
print("This is y: {}".format(sumOfY))
sumOfProductXY = taxi_points.map(lambda x: x[0] * x[1]).sum()
print("This is ProductXY: {}".format(sumOfProductXY))
sumOfXSqr = taxi_points.map(lambda x: x[0] * x[0]).sum()
print("This is XSqr: {}".format(sumOfXSqr))




m = (n*sumOfProductXY - sumOfX * sumOfY)/(n*sumOfXSqr - sumOfX * sumOfX)

b = (sumOfXSqr*sumOfY - sumOfX*sumOfProductXY)/(n*sumOfXSqr - sumOfX * sumOfX)

print("m = {:.8f}, b = {:.8f}".format(m,b))





