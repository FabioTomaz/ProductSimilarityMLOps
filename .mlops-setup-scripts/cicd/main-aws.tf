module "aws_create_sp" {
  source = "databricks/mlops-aws-project/databricks"
  providers = {
    databricks.staging = databricks.staging
    databricks.prod    = databricks.prod
  }
  service_principal_name       = "my-mlops-project-cicd"
  project_directory_path       = "/my-mlops-project"
  service_principal_group_name = "my-mlops-project-service-principals"
}

data "databricks_current_user" "staging_user" {
  provider = databricks.staging
}

provider "databricks" {
  alias = "staging_sp"
  host  = "https://dbc-38ce632c-4934.cloud.databricks.com"
  token = module.aws_create_sp.staging_service_principal_token
}

provider "databricks" {
  alias = "prod_sp"
  host  = "https://dbc-96a52355-626c.cloud.databricks.com"
  token = module.aws_create_sp.prod_service_principal_token
}

module "staging_workspace_cicd" {
  source = "./common"
  providers = {
    databricks = databricks.staging_sp
  }
  git_provider      = var.git_provider
  git_token         = var.git_token
  env               = "staging"
  github_repo_url   = var.github_repo_url
  github_server_url = var.github_server_url
}

module "prod_workspace_cicd" {
  source = "./common"
  providers = {
    databricks = databricks.prod_sp
  }
  git_provider      = var.git_provider
  git_token         = var.git_token
  env               = "prod"
  github_repo_url   = var.github_repo_url
  github_server_url = var.github_server_url
}

// We produce the service principal API tokens as output, to enable
// extracting their values and storing them as secrets in your CI system
//
// If using GitHub Actions, you can create new repo secrets through Terraform as well
// e.g. using https://registry.terraform.io/providers/integrations/github/latest/docs/resources/actions_secret
output "STAGING_WORKSPACE_TOKEN" {
  value     = module.aws_create_sp.staging_service_principal_token
  sensitive = true
}

output "PROD_WORKSPACE_TOKEN" {
  value     = module.aws_create_sp.prod_service_principal_token
  sensitive = true
}
