variable "BACKEND" {
  default = "cpu"
  validation {
    condition     = contains(["cpu", "cuda", "vulkan"], BACKEND)
    error_message = "BACKEND must be cpu, cuda, or vulkan."
  }
}

variable "BUILD_JOBS" {
  default = ""
}

variable "IMAGE" {
  default = "clap"
}

# No platform override means the host platform. Use platform= followed by
# platform+= overrides to request multiple platforms.
group "default" {
  targets = ["clap"]
}

target "clap" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "runtime"
  args = {
    BACKEND    = BACKEND
    BUILD_JOBS = BUILD_JOBS
  }
  tags = ["${IMAGE}:${BACKEND}"]
}

target "cpu" {
  inherits = ["clap"]
  args = { BACKEND = "cpu" }
  tags = ["${IMAGE}:cpu"]
}

target "cuda" {
  inherits = ["clap"]
  args = { BACKEND = "cuda" }
  tags = ["${IMAGE}:cuda"]
}

target "vulkan" {
  inherits = ["clap"]
  args = { BACKEND = "vulkan" }
  tags = ["${IMAGE}:vulkan"]
}
