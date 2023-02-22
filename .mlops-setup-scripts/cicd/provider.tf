terraform {
  required_providers {
    databricks = {
      source  = "databricks/databricks"
      version = ">= 0.5.8"
    }
  }
  // The `backend` block below configures the s3 backend
  // (docs: https://www.terraform.io/language/settings/backends/s3)
  // for storing Terraform state in an AWS S3 bucket. The targeted S3 bucket and DynamoDB table are
  // provisioned by the Terraform config under .mlops-setup-scripts/terraform
  // Note: AWS region must be specified via environment variable or via the `region` field
  // in the provider block below, as described
  // in https://registry.terraform.io/providers/hashicorp/aws/latest/docs#environment-variables
  backend "s3" {
    bucket         = "my-mlops-project-cicd-setup-tfstate"
    dynamodb_table = "my-mlops-project-cicd-setup-tfstate-lock"
    key            = "cicd-setup.terraform.tfstate"
  }
}

provider "databricks" {
  alias   = "staging"
  profile = var.staging_profile
}

provider "databricks" {
  alias   = "prod"
  profile = var.prod_profile
}

