#!/usr/bin/env python
# coding: utf-8

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (RandomForestClassifier, 
                                       LogisticRegression, 
                                       GBTClassifier)
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import (BinaryClassificationEvaluator, 
                                   MulticlassClassificationEvaluator)
from pyspark.sql.types import DoubleType, TimestampType
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt

import os
import shutil

try:
    # 1. Initialisation Spark
    spark = SparkSession.builder \
        .appName("FraudDetection") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")

    # 2. Chargement des données
    df = spark.read.csv(
        "/home/aya/Streaming-Fraud-Detection/data/raw/fraudTrain.csv",
        header=True,
        inferSchema=True,
        ignoreLeadingWhiteSpace=True,
        ignoreTrailingWhiteSpace=True
    )
    
    if '_c0' in df.columns:
        df = df.drop('_c0')
    
    # 3. Feature Engineering
    def calculate_distance(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 6371 * 2 * atan2(sqrt(a), sqrt(1-a))
    
    distance_udf = F.udf(calculate_distance, DoubleType())
    
    df = (df
          .withColumn("trans_date", F.to_timestamp("trans_date_trans_time"))
          .withColumn("hour", F.hour("trans_date"))
          .withColumn("day_of_week", F.dayofweek("trans_date"))
          .withColumn("month", F.month("trans_date"))
          .withColumn("age", F.year("trans_date") - F.year("dob"))
          .withColumn("distance", distance_udf("lat", "long", "merch_lat", "merch_long"))
          .drop("first", "last", "street", "city", "state", "zip", "trans_num", "dob", "trans_date_trans_time","cc_num", "merchant", "category", "job", "gender"))  # ⬅️ suppression des colonnes catégorielles
    
    windowSpec = Window.partitionBy('amt')
    df = df.withColumn('amt_vs_category_avg', F.col('amt')/F.avg('amt').over(windowSpec))

    # 4. Conversion timestamp -> numérique
    timestamp_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, TimestampType)]
    for col in timestamp_cols:
        df = df.withColumn(col, F.unix_timestamp(col))
    
    # 5. Split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # 6. Poids pour gestion des classes
    fraud_counts = train_df.groupBy("is_fraud").count().collect()
    counts = {row['is_fraud']: row['count'] for row in fraud_counts}
    total = counts[0] + counts[1]
    weight_for_0 = total / (2 * counts[0])
    weight_for_1 = total / (2 * counts[1])
    
    train_df = train_df.withColumn(
        "class_weight",
        F.when(F.col("is_fraud") == 1, weight_for_1).otherwise(weight_for_0)
    )
    
    # 7. Modèles
    feature_cols = [col for col in train_df.columns if col not in ["is_fraud", "class_weight"]]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    models = {
        "RandomForest": RandomForestClassifier(
            featuresCol="features",
            labelCol="is_fraud",
            weightCol="class_weight",
            maxBins=64,
            seed=42
        ),
        "LogisticRegression": LogisticRegression(
            featuresCol="features",
            labelCol="is_fraud",
            weightCol="class_weight",
            maxIter=100
        ),
        "GBTClassifier": GBTClassifier(
            featuresCol="features",
            labelCol="is_fraud",
            weightCol="class_weight",
            maxBins=64,
            seed=42
        )
    }
    
    resultats_dir = "/home/aya/Streaming-Fraud-Detection/resultats"
    os.makedirs(resultats_dir, exist_ok=True)
    
    results = []

    # 8. Entraînement et Évaluation
    for name, model in models.items():
        print(f"=== Training {name} ===")
        
        model_dir = os.path.join(resultats_dir, f"{name}_model")
        pred_dir = os.path.join(resultats_dir, f"predictions_{name}")
        
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        if os.path.exists(pred_dir):
            shutil.rmtree(pred_dir)
        
        pipeline = Pipeline(stages=[assembler, model])
        trained_model = pipeline.fit(train_df)
        
        trained_model.save(model_dir)
        
        predictions = trained_model.transform(test_df)
        (predictions
         .select("is_fraud", "prediction", "probability")
         .write
         .mode("overwrite")
         .parquet(pred_dir))
        
        evaluators = {
            "AUC": BinaryClassificationEvaluator(labelCol="is_fraud", metricName="areaUnderROC"),
            "Accuracy": MulticlassClassificationEvaluator(labelCol="is_fraud", metricName="accuracy"),
            "F1": MulticlassClassificationEvaluator(labelCol="is_fraud", metricName="f1")
        }
        metrics = {metric: evaluator.evaluate(predictions) for metric, evaluator in evaluators.items()}
        results.append({"Model": name, **metrics})


 
except Exception as e:
    print(f"ERREUR: {str(e)}")
    import traceback
    traceback.print_exc()

finally:
    if 'spark' in locals():
        spark.stop()

