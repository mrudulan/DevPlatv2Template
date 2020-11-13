from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import numpy as np
import pandas
import mlflow
import mlflow.onnx
import train_utils as utils

(sklearn_model_name, onnx_model_name) = utils.parse_job_arguments()

def main():

    from azureml.opendatasets import NycTlcYellow
    from datetime import datetime
    from dateutil import parser

    # Load data
    # Get a sample data of NYC yellow taxi from Azure Open Datasets
    start_date = parser.parse('2018-05-01')
    end_date = parser.parse('2018-05-07')
    nyc_tlc = NycTlcYellow(start_date=start_date, end_date=end_date)
    nyc_tlc_df = nyc_tlc.to_pandas_dataframe()
    nyc_tlc_df.info()

    # Prepare and featurize data
    # - There are extra dimensions that are not going to be useful in the model, taking the dimensions that are needed and add into the featurised dataframe. 
    # - There are also a bunch of outliers in the data to filter out.
    sampled_df = nyc_tlc_df.sample(n=10000, random_state=123)

    def get_pickup_time(df):
        pickupHour = df['pickupHour']
        if ((pickupHour >= 7) & (pickupHour <= 10)):
            return 'AMRush'
        elif ((pickupHour >= 11) & (pickupHour <= 15)):
            return 'Afternoon'
        elif ((pickupHour >= 16) & (pickupHour <= 19)):
            return 'PMRush'
        else:
            return 'Night'

    featurized_df = pandas.DataFrame()
    featurized_df['tipped'] = (sampled_df['tipAmount'] > 0).astype('int')
    featurized_df['fareAmount'] = sampled_df['fareAmount'].astype('float32')
    featurized_df['paymentType'] = sampled_df['paymentType'].astype('int')
    featurized_df['passengerCount'] = sampled_df['passengerCount'].astype('int')
    featurized_df['tripDistance'] = sampled_df['tripDistance'].astype('float32')
    featurized_df['pickupHour'] = sampled_df['tpepPickupDateTime'].dt.hour.astype('int')
    featurized_df['tripTimeSecs'] = ((sampled_df['tpepDropoffDateTime'] - sampled_df['tpepPickupDateTime']) / np.timedelta64(1, 's')).astype('int')

    featurized_df['pickupTimeBin'] = featurized_df.apply(get_pickup_time, axis=1)
    featurized_df = featurized_df.drop(columns='pickupHour')


    filtered_df = featurized_df[(featurized_df.tipped >= 0) & (featurized_df.tipped <= 1)\
        & (featurized_df.fareAmount >= 1) & (featurized_df.fareAmount <= 250)\
        & (featurized_df.paymentType >= 1) & (featurized_df.paymentType <= 2)\
        & (featurized_df.passengerCount > 0) & (featurized_df.passengerCount < 8)\
        & (featurized_df.tripDistance >= 0) & (featurized_df.tripDistance <= 100)\
        & (featurized_df.tripTimeSecs >= 30) & (featurized_df.tripTimeSecs <= 7200)]

    filtered_df.info()

    # Split training and testing data sets
    # - 70% of the data is used to train the model
    # - 30% of the data is used to test the model
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(filtered_df, test_size=0.3, random_state=123)

    x_train = pandas.DataFrame(train_df.drop(['tipped'], axis = 1))
    y_train = pandas.DataFrame(train_df.iloc[:,train_df.columns.tolist().index('tipped')])

    # For scoring
    x_test = pandas.DataFrame(test_df.drop(['tipped'], axis = 1))
    y_test = pandas.DataFrame(test_df.iloc[:,test_df.columns.tolist().index('tipped')])

    # Train a bi-classifier to predict whether a taxi trip will be tipped or not.
    mlflow.sklearn.autolog()
    
    # preprocessor and pipeline
    float_features = ['fareAmount', 'tripDistance']
    float_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    integer_features = ['paymentType', 'passengerCount', 'tripTimeSecs']
    integer_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = ['pickupTimeBin']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('float', float_transformer, float_features),
            ('integer', integer_transformer, integer_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    sk_model = Pipeline(steps=[('preprocessor', preprocessor),('classifier', LogisticRegression(solver='lbfgs'))])
    sk_model = sk_model.fit(x_train, np.ravel(y_train))

    # Infer signature
    (signature, input_sample) = utils.get_model_signature(x_train=x_train)

    print("====================== sklearn: model log and register ===================")
    mlflow.sklearn.log_model(sk_model=sk_model, artifact_path="sklearn_model", registered_model_name=sklearn_model_name, signature=signature, input_example=input_sample)
    
    print("====================== onnx: model conversion ============================")
    # Currently, T-SQL scoring only supports ONNX model format (https://onnx.ai/).
    onnx_model = utils.convert_to_onnx_model(sk_model=sk_model, x_train=x_train)

    print("====================== onnx: log and register ============================")
    mlflow.onnx.log_model(onnx_model=onnx_model, artifact_path='onnx_model', registered_model_name=onnx_model_name, signature=signature, input_example=input_sample)

if __name__ == "__main__":
    main()