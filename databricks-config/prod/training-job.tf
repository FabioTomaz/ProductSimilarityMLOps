resource "databricks_job" "model_training_job" {
  name = "${local.env_prefix}my-mlops-project-model-training-job"

  # Optional validation: we include it here for convenience, to help ensure that the job references a notebook
  # that exists in the current repo. Note that Terraform >= 1.2 is required to use these validations
  lifecycle {
    postcondition {
      condition     = alltrue([for task in self.task : fileexists("../../${task.notebook_task[0].notebook_path}.py")])
      error_message = "Databricks job must reference a notebook at a relative path from the root of the repo, with file extension omitted. Could not find one or more notebooks in repo"
    }
  }

  task {
    task_key = "Train"

    notebook_task {
      notebook_path = "notebooks/TrainWithFeatureStore"
      base_parameters = {
        training_data_path = "/databricks-datasets/nyctaxi-with-zipcodes/subsampled"
        experiment_name    = databricks_mlflow_experiment.experiment.name
        model_name         = "${local.env_prefix}my-mlops-project-model"
        fs_stage           = ""
      }
    }

    new_cluster {
      num_workers   = 3
      spark_version = "11.0.x-cpu-ml-scala2.12"
      node_type_id  = "i3.xlarge"
      # We set the job cluster to single user mode to enable your training job to access
      # the Unity Catalog.
      single_user_name   = data.databricks_current_user.service_principal.user_name
      data_security_mode = "SINGLE_USER"
      custom_tags        = { "clusterSource" = "mlops-stack/0.0" }
    }
  }

  task {
    task_key = "ModelValidation"
    depends_on {
      task_key = "Train"
    }

    notebook_task {
      notebook_path = "notebooks/ModelValidation"
      base_parameters = {
        env             = local.env
        experiment_name = databricks_mlflow_experiment.experiment.name
        # Run mode for model validation. Possible values are :
        #   disabled : Do not run the model validation notebook.
        #   dry_run  : Run the model validation notebook. Ignore failed model validation rules and proceed to move model to Production stage.
        #   enabled  : Run the model validation notebook. Move model to Production stage only if all model validation rules are passing.
        # Please complete the TODO sessions in notebooks/ModelValidation before enabling model validation
        run_mode = "disabled"
      }
    }

    new_cluster {
      num_workers   = 3
      spark_version = "11.0.x-cpu-ml-scala2.12"
      node_type_id  = "i3.xlarge"
      custom_tags   = { "clusterSource" = "mlops-stack/0.0" }
    }
  }

  task {
    task_key = "TriggerModelDeploy"
    depends_on {
      task_key = "ModelValidation"
    }

    notebook_task {
      notebook_path = "notebooks/TriggerModelDeploy"
      base_parameters = {
        env = local.env
      }
    }

    new_cluster {
      num_workers   = 3
      spark_version = "11.0.x-cpu-ml-scala2.12"
      node_type_id  = "i3.xlarge"
      custom_tags   = { "clusterSource" = "mlops-stack/0.0" }
    }
  }

  git_source {
    url      = var.git_repo_url
    provider = "gitHub"
    branch   = "release"
  }

  schedule {
    quartz_cron_expression = "0 0 9 * * ?" # daily at 9am
    timezone_id            = "UTC"
  }

  # If you want to turn on notifications for this job, please uncomment the below code,
  # and provide a list of emails to the on_failure argument.
  #
  #  email_notifications {
  #    on_failure: []
  #  }
}
