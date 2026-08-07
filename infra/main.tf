# MedGuard infrastructure as code (Azure).
# Describes the full stack: resource group, container registry, Log Analytics,
# Application Insights, Container Apps environment, and the Container App itself.

terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

resource "azurerm_resource_group" "this" {
  name     = "medguard-rg"
  location = var.location
}

# Private registry for the app image
resource "azurerm_container_registry" "this" {
  name                = var.acr_name # globally unique
  resource_group_name = azurerm_resource_group.this.name
  location            = var.acr_location
  sku                 = "Basic"
  admin_enabled       = true
}

# Log Analytics workspace backs both the Container Apps env and App Insights
resource "azurerm_log_analytics_workspace" "this" {
  name                = "medguard-logs"
  resource_group_name = azurerm_resource_group.this.name
  location            = var.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_application_insights" "this" {
  name                = "medguard-insights"
  resource_group_name = azurerm_resource_group.this.name
  location            = var.location
  application_type    = "web"
  workspace_id        = azurerm_log_analytics_workspace.this.id
}

resource "azurerm_container_app_environment" "this" {
  name                       = "medguard-env"
  resource_group_name        = azurerm_resource_group.this.name
  location                   = var.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.this.id
}

resource "azurerm_container_app" "this" {
  name                         = "medguard-api"
  resource_group_name          = azurerm_resource_group.this.name
  container_app_environment_id = azurerm_container_app_environment.this.id
  revision_mode                = "Single"

  # Pull auth for the private registry
  registry {
    server               = azurerm_container_registry.this.login_server
    username             = azurerm_container_registry.this.admin_username
    password_secret_name = "acr-password"
  }

  # Secrets (never inline in env vars)
  secret {
    name  = "acr-password"
    value = azurerm_container_registry.this.admin_password
  }
  secret {
    name  = "groq-key"
    value = var.groq_api_key
  }
  secret {
    name  = "qdrant-key"
    value = var.qdrant_api_key
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = 0
    max_replicas = 1

    container {
      name   = "medguard-api"
      image  = "${azurerm_container_registry.this.login_server}/medguard-api:${var.image_tag}"
      cpu    = 2.0
      memory = "4Gi"

      env {
        name  = "LLM_PROVIDER"
        value = "groq"
      }
      env {
        name  = "LLM_MODEL"
        value = "llama-3.1-8b-instant"
      }
      env {
        name  = "QDRANT_URL"
        value = var.qdrant_url
      }
      env {
        name  = "VERIFIER"
        value = "nli_judge"
      }
      env {
        name  = "ENABLE_GUARDS"
        value = "true"
      }
      env {
        name  = "GATE_ACTIVE"
        value = "true"
      }
      env {
        name        = "GROQ_API_KEY"
        secret_name = "groq-key"
      }
      env {
        name        = "QDRANT_API_KEY"
        secret_name = "qdrant-key"
      }
      env {
        name  = "APPLICATIONINSIGHTS_CONNECTION_STRING"
        value = azurerm_application_insights.this.connection_string
      }
    }
  }
  lifecycle {
        ignore_changes = [template[0].container[0].image]
  }
}
