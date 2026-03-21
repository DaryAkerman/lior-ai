variable "anthropic_api_key" {
  description = "Your Anthropic API key (from console.anthropic.com)"
  type        = string
  sensitive   = true
}

variable "app_name" {
  description = "Base name used for all Azure resources. The web app URL will be <app_name>.azurewebsites.net — must be globally unique."
  type        = string
  default     = "psych101-ai"
}

variable "location" {
  description = "Azure region to deploy into"
  type        = string
  default     = "West Europe"
}

variable "docker_image" {
  description = "Docker Hub image to deploy, in the form username/repo (e.g. winterzone/psych101-ai)"
  type        = string
  default     = "winterzone2/psych101-ai"
}

variable "sku_name" {
  description = "App Service Plan SKU. B1 is the cheapest tier with 24/7 uptime (Always On). Upgrade to B2 if you hit memory errors. Options: B1 (~$13/mo), B2 (~$26/mo), B3 (~$52/mo)."
  type        = string
  default     = "B2"
}
