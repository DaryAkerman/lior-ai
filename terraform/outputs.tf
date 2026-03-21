output "app_url" {
  description = "Public URL of the deployed web app"
  value       = "https://${azurerm_linux_web_app.main.default_hostname}"
}

output "resource_group" {
  description = "Azure Resource Group containing all resources"
  value       = azurerm_resource_group.main.name
}

output "app_name" {
  description = "Azure Web App name (also used in deploy.sh)"
  value       = azurerm_linux_web_app.main.name
}

output "deploy_command" {
  description = "Command to run after terraform apply to push your code"
  value       = "bash deploy.sh"
}
