variable "subscription_id" {
  type        = string
  description = "Azure subscription ID"
}

variable "location" {
  type        = string
  description = "Region for the app, environment, logs, and App Insights"
  default     = "swedencentral"
}

variable "acr_name" {
  type        = string
  description = "Globally unique container registry name"
  default     = "medguardacr"
}

variable "acr_location" {
  type        = string
  description = "Region for the container registry"
  default     = "germanywestcentral"
}

variable "image_tag" {
  type        = string
  description = "Image tag to deploy (e.g. the git commit SHA)"
}

variable "qdrant_url" {
  type        = string
  description = "Qdrant Cloud endpoint, e.g. https://<cluster>:6333"
}

variable "groq_api_key" {
  type      = string
  sensitive = true
}

variable "qdrant_api_key" {
  type      = string
  sensitive = true
}
