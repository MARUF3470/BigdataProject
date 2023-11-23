from flask import Flask, request, jsonify
from flask_cors import CORS
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator
from functools import reduce
import findspark
findspark.init()
spark = SparkSession.builder.appName("CarRentPrediction").getOrCreate()
app = Flask(__name__)
CORS(app)
print('Hello world')

@app.route('/predict', methods=['POST'])
def providePrediction():
    try:
        # Get data from the request
        form_data = request.form.to_dict()
        
        print('working')
        data = spark.read.csv("./CarRentalData.csv", header=True, inferSchema=True)
        data = data.dropna(how="any")
        data.show(5)
        null_counts = [data.where(col(column).isNull()).count() for column in data.columns]
        print(null_counts)

        categorical_cols = ["fuelType", "locationCity", "locationCountry", "locationState", "vehicleMake", "vehicleModel", "vehicleType"]
        indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index").fit(data) for col in categorical_cols]

        # Step 5: Create a Pipeline to execute the StringIndexer stages
        pipeline = Pipeline(stages=indexers)
        data = pipeline.fit(data).transform(data)
        data = data.drop(*categorical_cols)
        data = data.drop("ownerId")
        data = data.drop("vehicleYear")
        data.show()

        feature_columns = data.columns
        feature_columns.remove("vehicleMake_index")
        feature_columns.remove("vehicleModel_index")

        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        data = assembler.transform(data)
        data.show(5)   

        (training_data, testing_data) = data.randomSplit([0.8, 0.2], seed=1234)

        rf_make = RandomForestRegressor(featuresCol="features", labelCol="vehicleMake_index", maxBins=1000)
        model_make = rf_make.fit(training_data)    

        predictions_make = model_make.transform(testing_data)

        evaluator_model = RegressionEvaluator(labelCol="vehicleMake_index", predictionCol="prediction", metricName="mse")
        mse_model = evaluator_model.evaluate(predictions_make)
        print(f"MSE for vehicleMake: {mse_model}") 

        rf_model = RandomForestRegressor(featuresCol="features", labelCol="vehicleModel_index", maxBins=1000)
        model_model = rf_model.fit(training_data)

        predictions_model = model_model.transform(testing_data)      

        evaluator_model = RegressionEvaluator(labelCol="vehicleModel_index", predictionCol="prediction", metricName="mse")
        mse_model = evaluator_model.evaluate(predictions_model)
        print(f"MSE for vehicleModel: {mse_model}")  

        single_test_data = data.select(feature_columns).limit(1)
        single_test_data.show()

        single_test_data = assembler.transform(single_test_data)

        prediction_make = model_make.transform(single_test_data).select("prediction").collect()[0][0]

        prediction_model = model_model.transform(single_test_data).select("prediction").collect()[0][0]

        print(f"Predicted vehicleMake: {prediction_make}")
        print(f"Predicted vehicleModel: {prediction_model}")

        
        ##############################################################################
        
        data2 = spark.read.csv("./CarRentalData.csv", header=True, inferSchema=True)
        data2 = data2.dropna(how="any")
        data2.show(2)
        
        conditions = [col(column) == value for column, value in form_data.items()]
        filtered_data = data2.filter(reduce(lambda x, y: x & y, conditions))

        filtered_data.show()
        feature_columns = filtered_data.columns
        feature_columns.remove("vehicleMake")
        feature_columns.remove("vehicleModel")
        indexers_filtered = [StringIndexer(inputCol=col, outputCol=f"{col}_index").fit(data2) for col in categorical_cols]

# Create a pipeline for indexing
        pipeline_filtered = Pipeline(stages=indexers_filtered)

# Transform the filtered data using the pipeline
        filtered_data_indexed = pipeline_filtered.fit(data2).transform(filtered_data)

# Display the indexed data
        categorical_cols = ["fuelType", "locationCity", "locationCountry", "locationState", "vehicleMake", "vehicleModel", "vehicleType","vehicleMake_index","vehicleModel_index"]
        filtered_data_indexed = filtered_data_indexed.drop(*categorical_cols)
        filtered_data_indexed = assembler.transform(filtered_data_indexed)
        filtered_data_indexed.show()

        # Make predictions for vehicleMake
        prediction_make = model_make.transform(filtered_data_indexed).select("prediction").collect()[0][0]
        prediction_model = model_model.transform(filtered_data_indexed).select("prediction").collect()[0][0]

        distinct_make_values = data2.select("vehicleMake").distinct().collect()
        distinct_model_values = data2.select("vehicleModel").distinct().collect()

        make_index_to_value = {index: value for index, value in enumerate(distinct_make_values)}
        model_index_to_value = {index: value for index, value in enumerate(distinct_model_values)}

        predicted_make_index = round(prediction_make)
        predicted_model_index = round(prediction_model)

        predicted_make = make_index_to_value.get(predicted_make_index)
        predicted_model = model_index_to_value.get(predicted_model_index) 
    
        
        print(f"Predicted vehicleMake: {predicted_make} (original index: {predicted_make_index}, original: {prediction_make})")
        print(f"Predicted vehicleModel: {predicted_model} (original index: {predicted_model_index}, original: {prediction_model})")


        return jsonify({
            'vehicleMake': predicted_make[0],
            'vehicleModel': predicted_model[0]
        })
    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(debug=True)