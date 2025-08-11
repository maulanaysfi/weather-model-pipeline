# KubeFlow Pipeline : Weather Prediction model

This is a KubeFlow ML Pipeline compiled using [python kfp SDK](https://www.kubeflow.org/docs/components/pipelines/legacy-v1/sdk/sdk-overview/). This project is intended to maintain end-to-end Machine Learning workflow. Powered by KubeFlow Pipeline and compiled using python.

## Why KubeFlow pipeline?

The main purpose is to utilize Kubernetes Cluster as the infrastructure. Among existing ML workflow tools, KubeFlow is optimized to run on a Kubernetes architecture.

## Why predicting weather?

There is no specific reason but to learn how to maintain a model lifecycle. The dataset used in this project is publicly available in [Kaggle](www.kaggle.com).

# How to create KubeFlow ML Pipeline
These are the steps to create a Machine Learning pipeline with KubeFlow.

## Deploying KubeFlow pipeline on Kubernetes
1. Deploy KubeFlow using kustomize. (Referring to [official installation guide](https://www.kubeflow.org/docs/components/pipelines/legacy-v1/installation/localcluster-deployment/#deploying-kubeflow-pipelines) on Local Kubernetes Cluster)
```shell
export PIPELINE_VERSION=2.4.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"
```
2. Port-forwarding the Kubernetes service.
```shell
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

## Compiling pipeline with python
1. Create new python virtual environment. 
```shell
python -m venv venv
```
2. Activate the virtual environment. (In this case, I'm using bash.)
```shell
source venv/bin/activate
```
3. Install KFP dependencies.
```shell
pip install kfp typing
```
4. Clone this repository.
5. Run python script.
```shell
python weather-predict-model.py
```
6. Access KubeFlow pipeline (port-forwarded) URL. `http://localhost:8080`
7. Upload the compiled KubeFlow pipeline YAML file.
