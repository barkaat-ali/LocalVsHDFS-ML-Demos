from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col

# Create a Spark session
spark = SparkSession.builder.appName("StrokePrediction").getOrCreate()

# Load your data from HDFS
data = spark.read.csv("hdfs://0.0.0.0:9000/user/hadoop/1.csv", header=True, inferSchema=True)
data = data.withColumn("bmi", col("bmi").cast("double"))

# Convert categorical variables to numerical using StringIndexer
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index").fit(data) for col in categorical_cols]
pipeline = Pipeline(stages=indexers)
indexed_data = pipeline.fit(data).transform(data)

# Impute null values with the mean of the "bmi" column
imputer = Imputer(inputCols=["bmi"], outputCols=["bmi_imputed"]).setStrategy("mean")
imputed_data = imputer.fit(indexed_data).transform(indexed_data)

# Combine feature columns into a single vector column
feature_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi_imputed"] + [col + "_index" for col in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_data = assembler.transform(imputed_data)

# Split the data into training and testing sets
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(featuresCol="features", labelCol="stroke", numTrees=100, maxDepth=5)

# Train the model
model = rf_classifier.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="stroke")
accuracy = evaluator.evaluate(predictions)

print("Accuracy:", accuracy)

# Stop the Spark session
spark.stop()

