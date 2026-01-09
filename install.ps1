# Ensure errors stop the script
$ErrorActionPreference = "Stop"

Write-Host "Checking for uv..."

# Check if uv is available
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue

if (-not $uvCommand) {
    Write-Host "uv not found. Installing uv..."

    # Install uv using the official installer
    Invoke-WebRequest https://astral.sh/uv/install.ps1 -UseBasicParsing | Invoke-Expression

    # Refresh PATH for the current session
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "User") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "Machine")

    # Verify installation
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv installation failed or uv is not on PATH."
    }

    Write-Host "uv installed successfully."
} else {
    Write-Host "uv is already installed."
}

Write-Host "Installing datapipes using uv..."

# Install numpy
uv tool install git+https://github.com/Bulow/datapipes

# Refresh PATH for the current session
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "User") + ";" +
            [System.Environment]::GetEnvironmentVariable("PATH", "Machine")

Write-Host "datapipes installation complete."
Write-Host "To start a new project, open a terminal in a empty folder and run `datapipes init`."
Write-Host "To see available commands, run `datapipes help`."

datapipes help
