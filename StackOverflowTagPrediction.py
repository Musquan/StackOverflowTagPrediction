from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, count, desc
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create Spark session
spark = SparkSession.builder.appName("LogisticRegression").getOrCreate()

# Define path to the CSV file
bucket_name = "dataproc-staging-us-central1-609145668490-9oco1tdd"
path = f"gs://{bucket_name}/train.csv"

# Read CSV file into a Spark DataFrame
df = spark.read.csv(path, header=True, inferSchema=True)

df.count()

# Extract tags column from the dataframe
tags_df = df.select("Tags")

# Split the tags column into individual tags
tags_df = tags_df.withColumn("Tag", explode(split("Tags", "<")))

# Count the occurrence of each tag and sort by descending order
tags_count_df = tags_df.groupBy("Tag").agg(count("*").alias("Count")).sort(desc("Count"))

# Select the top 9 tags
top_tags_df = tags_count_df.limit(9)

# Show the top 9 tags
top_tags_df.show()

top_9_tags = ["python", "java", "javascript", "c#", "php", "android", "ios", "c", "c++"]
df = df.filter(df.Tags.isin(top_9_tags))

df.count()

# Extract relevant columns for modeling
df = df.select("Title", "Tags")

# Clean and tokenize text data
tokenizer = Tokenizer(inputCol="Title", outputCol="Title_words")
df = tokenizer.transform(df)

# Remove stop words
stopwords = StopWordsRemover.loadDefaultStopWords("english")
stopwordsRemover = StopWordsRemover(inputCol="Title_words", outputCol="Title_filtered1", stopWords=stopwords)
df = stopwordsRemover.transform(df)

# Hash and transform text data into vectors
hashingTF = HashingTF(inputCol="Title_filtered1", outputCol="Title_features", numFeatures=1000)
df = hashingTF.transform(df)

# Fit an IDF model on the output of HashingTF
idf = IDF(inputCol="Title_features", outputCol="Title_idf_features")
idfModel = idf.fit(df)
df = idfModel.transform(df)

df.show()

# Convert tags to label indices
labelIndexer = StringIndexer(inputCol="Tags", outputCol="label")
df = labelIndexer.fit(df).transform(df)

# Split the data into training and testing sets
(trainingData, testData) = df.randomSplit([0.8, 0.2], seed=42)

# Create Logistic Regression model
lr = LogisticRegression(maxIter=100, regParam=0.01, elasticNetParam=0.1, featuresCol="Title_idf_features", labelCol="label")

# Train the model using the training data
lrModel = lr.fit(trainingData)

# Make predictions on the test data
predictions = lrModel.transform(testData)

# Evaluate model accuracy
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Print the accuracy score
print(f"Accuracy: {accuracy}")
