from skl2onnx import convert_sklearn
from mlflow.models.signature import infer_signature
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, DoubleTensorType, StringTensorType

import pandas
import argparse

def parse_job_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sklearn_model_name', type=str, default=False)
    parser.add_argument('--onnx_model_name', type=str, default=False)

    args = parser.parse_args()

    sklearn_model_name = args.sklearn_model_name
    onnx_model_name = args.onnx_model_name
    print('sklearn model name: ' +sklearn_model_name)
    print('onnx model name: ' + onnx_model_name)

    return (sklearn_model_name, onnx_model_name)

def get_model_signature(x_train):
    input_sample = x_train.head(1)
    output_sample = pandas.DataFrame(columns=['output_label'], data=[1])
    signature = infer_signature(input_sample, output_sample)
    return (signature,input_sample)

def convert_to_onnx_model(sk_model, x_train):
    def convert_dataframe_schema(df, drop=None):
        inputs = []
        for k, v in zip(df.columns, df.dtypes):
            if drop is not None and k in drop:
                continue
            if v == 'int64':
                t = Int64TensorType([1, 1])
            elif v == 'float32':
                t = FloatTensorType([1, 1])
            elif v == 'float64':
                t = DoubleTensorType([1, 1])
            else:
                t = StringTensorType([1, 1])
            inputs.append((k, t))
        return inputs

    # convert to onnx
    model_inputs = convert_dataframe_schema(x_train)
    onnx_model = convert_sklearn(sk_model, 'nyc_taxi_tip_predict', model_inputs)
    return onnx_model

