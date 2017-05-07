import findspark
findspark.init('/home/dave/spark-2.1.1-bin-hadoop2.7',edit_rc=True)
from pyspark import SparkContext
sc=SparkContext()
sc
