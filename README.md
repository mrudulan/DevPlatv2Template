# Predict NYC Taxi Tips

This repo trains a model based on an Open Dataset that tracks NYC Yellow Taxi trips and various attributes around them. The goal is to for a given trip, predict whether there will be a tip or not. The model then will be converted to ONNX format and tracked by MLFlow.

# Azure Machine Learning Service 
[Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/) provides a cloud-based environment to prep data, train, test, deploy, manage, and track machine learning models. This service fully supports open-source technologies such as PyTorch, TensorFlow, and scikit-learn and can be used for any kind of machine learning, from classical ML to deep learning, supervised and unsupervised learning.

Learn how Azure Machine Learning can help you streamline the building, training, and deployment of machine learning models. Start free today.

# Getting Started

### Train a sklearn model w/ NYC Yellow Taxi trips

[![Train With Datasets On Azure](https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/1-CONTRIBUTION-GUIDE/images/deploytoazure.svg?sanitize=true)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fmrudulan%2FDevPlatv2Template%2Fmaster%2F.cloud%2FazuredeployCommandJobWithData.json)
[![Visualize](https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/1-CONTRIBUTION-GUIDE/images/visualizebutton.svg?sanitize=true)](http://armviz.io/#/?load=https://raw.githubusercontent.com/mrudulan/DevPlatv2Template/master/.cloud/azuredeployCommandJobWithData.json)


### Deploy a sklearn model as a batch endpoint

[![Train On Azure](https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/1-CONTRIBUTION-GUIDE/images/deploytoazure.svg?sanitize=true)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fmrudulan%2FDevPlatv2Template%2Fmaster%2F.cloud%2FazuredeployBatchEndpoint.json)


### Deploy a sklearn model as an online endpoint

[![Train On Azure](https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/1-CONTRIBUTION-GUIDE/images/deploytoazure.svg?sanitize=true)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fmrudulan%2FDevPlatv2Template%2Fmaster%2F.cloud%2FazuredeployOnlineEndpoint.json)


This ARM template creates a code job in Azure Machine Learning workspace.

If you are new to Azure Machine Learning, see:

- [Azure Machine Learning service](https://azure.microsoft.com/services/machine-learning-service/)
- [Azure Machine Learning documentation](https://docs.microsoft.com/azure/machine-learning/)
- [Azure Machine Learning template reference](https://docs.microsoft.com/azure/templates/microsoft.machinelearningservices/allversions)
- [Quickstart templates](https://azure.microsoft.com/resources/templates/)
- [MLflow](https://github.com/mlflow/mlflow)
- [MLflow in Azure ML](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow)
