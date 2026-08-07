output "app_url" {
  description = "Public HTTPS endpoint of the MedGuard API"
  value       = "https://${azurerm_container_app.this.ingress[0].fqdn}"
}

output "acr_login_server" {
  description = "Container registry login server"
  value       = azurerm_container_registry.this.login_server
}

output "app_insights_connection_string" {
  description = "App Insights connection string (sensitive)"
  value       = azurerm_application_insights.this.connection_string
  sensitive   = true
}
