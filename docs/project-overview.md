# Project Overview

[(back to main README)](../README.md)

## ML pipeline structure
This project defines an ML pipeline for automated retraining and batch inference of an ML model
on tabular data.

See the full pipeline structure below. The [stacks README](https://github.com/databricks/mlops-stack/blob/main/Pipeline.md)
contains additional details on how ML pipelines are tested and deployed across each of the dev, staging, prod environments below.

![MLOps Stacks diagram](./images/mlops-stack-summary.png)


## Code structure
This project contains the following components:

| Component                  | Description                                                                                                                                     |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| ML Code                    | Example ML project code, with unit tested Python modules and notebooks using [MLflow recipes](https://mlflow.org/docs/latest/recipes.html)  |
| ML Resource Config as Code | ML pipeline resource config (training and batch inference job schedules, etc) defined through [Terraform](https://docs.databricks.com/dev-tools/terraform/index.html) |
| CI/CD                      | [GitHub Actions](https://github.com/actions) workflows to test and deploy ML code and resources                                                  |

contained in the following files:

```
├── features              <- Feature computation code (Python modules) that implements the feature transforms.
│                         The output of these transforms get persisted as Feature Store tables. Most development
│                         work happens here.
│
├── notebooks          <- Databricks notebooks that run the ML pipelines, i.e. run the features logic as part of an ETL
│                         job that publishes to a Feature Store table. Used to drive code execution on Databricks for CI/CD.
│                         In most cases, you do not need to modify these notebooks.
│
├── requirements.txt   <- Specifies Python dependencies for ML code (model training, batch inference, etc).
│
├── tests              <- Unit tests for the modules under `features`.
│
├── .github            <- Configuration folder for CI/CD using GitHub Actions. The CI/CD workflows run the notebooks
│                         under `notebooks` to test and deploy model training code
│
├── databricks-config  <- ML resource (ML jobs, MLflow models) config definitions expressed as code, across staging/prod.
│   ├── staging
│   ├── prod
```

## Next Steps
See the [main README](../README.md#using-this-repo) for additional links on how to work with this repo.
