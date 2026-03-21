terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
  required_version = ">= 1.5"
}

provider "azurerm" {
  features {}
}

# ── Resource Group ───────────────────────────────────────────────────────────

resource "azurerm_resource_group" "main" {
  name     = "${var.app_name}-rg"
  location = var.location
}

# ── App Service Plan (Linux) ─────────────────────────────────────────────────

resource "azurerm_service_plan" "main" {
  name                = "${var.app_name}-plan"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = var.sku_name
}

# ── Linux Web App (Docker container) ─────────────────────────────────────────

resource "azurerm_linux_web_app" "main" {
  name                = var.app_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  service_plan_id     = azurerm_service_plan.main.id

  https_only = true

  site_config {
    application_stack {
      docker_image_name   = "${var.docker_image}:latest"
      docker_registry_url = "https://index.docker.io"
    }

    # Keep the connection alive for streaming (SSE) responses
    http2_enabled = true

    # Prevent the app from sleeping when idle — required for 24/7 availability.
    always_on = true
  }

  app_settings = {
    "ANTHROPIC_API_KEY" = var.anthropic_api_key
    "WEBSITES_PORT"     = "8000"
  }

  logs {
    http_logs {
      file_system {
        retention_in_days = 7
        retention_in_mb   = 35
      }
    }
  }
}
