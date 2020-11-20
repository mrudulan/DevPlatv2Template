# Light GBM on Azure Machine Learning Service 

<img src=https://github.com/microsoft/LightGBM/blob/master/docs/logo/LightGBM_logo_black_text.svg width=100 />


LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:

- Faster training speed and higher efficiency.
- Lower memory usage.
- Better accuracy.
- Support of parallel and GPU learning.
- Capable of handling large-scale data.

For further details, please refer to [Features](https://github.com/microsoft/LightGBM/blob/master/docs/Features.rst).

Benefitting from these advantages, LightGBM is being widely-used in many [winning solutions](https://github.com/microsoft/LightGBM/blob/master/examples/README.md#machine-learning-challenge-winning-solutions) of machine learning competitions.

[Comparison experiments](https://github.com/microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment) on public datasets show that LightGBM can outperform existing boosting frameworks on both efficiency and accuracy, with significantly lower memory consumption. What's more, [parallel experiments](https://github.com/microsoft/LightGBM/blob/master/docs/Experiments.rst#parallel-experiment) show that LightGBM can achieve a linear speed-up by using multiple machines for training in specific settings.

Get Started and Documentation
-----------------------------

Our primary documentation is at https://lightgbm.readthedocs.io/ and is generated from this repository. If you are new to LightGBM, follow [the installation instructions](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html) on that site.


# Azure Machine Learning Service 
[Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/) provides a cloud-based environment to prep data, train, test, deploy, manage, and track machine learning models. This service fully supports open-source technologies such as PyTorch, TensorFlow, and scikit-learn and can be used for any kind of machine learning, from classical ML to deep learning, supervised and unsupervised learning.

Learn how Azure Machine Learning can help you streamline the building, training, and deployment of machine learning models. Start free today.

# Getting Started

### Train a LightGBM model w/ Datasets

[![Train With Datasets On Azure](https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/1-CONTRIBUTION-GUIDE/images/deploytoazure.svg?sanitize=true)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fmrudulan%2FDevPlatv2Template%2Fmaster%2F.cloud%2FazuredeployCommandJobWithData.json)
[![Visualize](https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/1-CONTRIBUTION-GUIDE/images/visualizebutton.svg?sanitize=true)](http://armviz.io/#/?load=https://raw.githubusercontent.com/mrudulan/DevPlatv2Template/master/.cloud/azuredeployCommandJobWithData.json)

### Train a LightGBM model w/ Distributed Training

[![Train With Tensorflow Distributed Config On Azure](https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/1-CONTRIBUTION-GUIDE/images/deploytoazure.svg?sanitize=true)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fmrudulan%2FDevPlatv2Template%2Fmaster%2F.cloud%2FazuredeployCommandJobTensorflow.json)

### Sweep job sample

[![Train On Azure](https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/1-CONTRIBUTION-GUIDE/images/deploytoazure.svg?sanitize=true)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fmrudulan%2FDevPlatv2Template%2Fmaster%2F.cloud%2FazuredeployLabelingJobWithData.json)

### Labeling job sample

[![Train On Azure](https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/1-CONTRIBUTION-GUIDE/images/deploytoazure.svg?sanitize=true)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fmrudulan%2FDevPlatv2Template%2Fmaster%2F.cloud%2FazuredeploySweepJobWithData.json)

### Deploy a LightGBM model as an endpoint

[![Train On Azure](https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/1-CONTRIBUTION-GUIDE/images/deploytoazure.svg?sanitize=true)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fmrudulan%2FDevPlatv2Template%2Fmaster%2F.cloud%2FazuredeployOnlineEndpoint.json)


This ARM template creates a code job in Azure Machine Learning workspace.

If you are new to Azure Machine Learning, see:

- [Azure Machine Learning service](https://azure.microsoft.com/services/machine-learning-service/)
- [Azure Machine Learning documentation](https://docs.microsoft.com/azure/machine-learning/)
- [Azure Machine Learning template reference](https://docs.microsoft.com/azure/templates/microsoft.machinelearningservices/allversions)
- [Quickstart templates](https://azure.microsoft.com/resources/templates/)
