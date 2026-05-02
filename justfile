# Justfile for rapi project
# Designed to work across different machines and directories

# Start echo server for debugging forwarded requests
# Listens on port 18081 and prints all request details
start-echo-server:
    @echo "=== Starting Echo Server on port 18081 ==="
    @echo ""
    @echo "[1/2] Checking prerequisites..."
    @command -v cargo >/dev/null 2>&1 || { echo "ERROR: 'cargo' is not installed. Please install Rust first:"; echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"; exit 1; }
    @echo "  ✓ cargo found: $(cargo --version)"
    @echo "[2/2] Starting Echo Server..."
    @cd {{justfile_directory()}} && PORT=18081 cargo run --bin echo_server

# Default recipe
default:
    @just --list

# Start Rust API service (development mode)
# Uses cargo run for faster compilation
start-rust-dev:
    @echo "=== Starting Rust API (dev mode) ==="
    @echo ""
    @echo "[1/2] Checking prerequisites..."
    @command -v cargo >/dev/null 2>&1 || { echo "ERROR: 'cargo' is not installed. Please install Rust first:"; echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"; exit 1; }
    @echo "  ✓ cargo found: $(cargo --version)"
    @echo "[2/2] Starting Rust API service..."
    @cd {{justfile_directory()}} && cargo run

# Start Rust API service (release mode)
# Uses cargo run --release for optimized performance
start-rust:
    @echo "=== Starting Rust API (release mode) ==="
    @echo ""
    @echo "[1/2] Checking prerequisites..."
    @command -v cargo >/dev/null 2>&1 || { echo "ERROR: 'cargo' is not installed. Please install Rust first:"; echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"; exit 1; }
    @echo "  ✓ cargo found: $(cargo --version)"
    @echo "[2/2] Starting Rust API service..."
    @cd {{justfile_directory()}} && cargo run --release

# Start openai-privacy service
# Performs: uv sync, pull model, start API server
start-openai-privacy:
    @echo "=== Starting OpenAI Privacy Filter API ==="
    @echo ""
    @echo "[1/4] Checking prerequisites..."
    @command -v uv >/dev/null 2>&1 || { echo "ERROR: 'uv' is not installed. Please install it first:"; echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
    @echo "  ✓ uv found: $(uv --version)"
    @echo "[2/4] Syncing dependencies..."
    @cd {{justfile_directory()}}/openai-privacy && uv sync --quiet
    @echo "  ✓ Dependencies synced"
    @echo "[3/4] Checking OPF model..."
    @cd {{justfile_directory()}}/openai-privacy && uv run python {{justfile_directory()}}/just/check_opf_model.py
    @echo "[4/4] Starting API service on http://0.0.0.0:8000"
    @echo "  Press Ctrl+C to stop"
    @echo ""
    @cd {{justfile_directory()}}/openai-privacy && uv run openai-privacy

# Run functional tests against the openai-privacy service
# Requires the service to be running on the target URL
test-openai-privacy base_url="http://localhost:8000":
    @echo "=== Running OpenAI Privacy Filter API Tests ==="
    @echo "Target: {{base_url}}"
    @echo ""
    @python3 {{justfile_directory()}}/just/test_openai_privacy.py --base-url {{base_url}}

# Run performance benchmarks against the openai-privacy service
# Requires the service to be running on the target URL
# Usage: just perf-openai-privacy [base_url] [concurrency] [requests]
perf-openai-privacy base_url="http://localhost:8000" concurrency="10" requests="50":
    @echo "=== Running OpenAI Privacy Filter API Performance Tests ==="
    @echo "Target: {{base_url}}"
    @echo "Concurrency: {{concurrency}}"
    @echo "Requests per benchmark: {{requests}}"
    @echo ""
    @python3 {{justfile_directory()}}/just/perf_test_openai_privacy.py --base-url "{{base_url}}" --concurrency {{concurrency}} --requests {{requests}}
