#!/bin/bash
set -euo pipefail

PROJECT_DIR="/opt/nous-proxy"

echo "=== NousResearch Proxy Setup ==="

# 1. Check uv
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Create project venv and install
echo "Setting up Python environment with uv..."
cd "$PROJECT_DIR"
uv venv .venv
uv pip install -e .

# 3. Create service user (no login shell, home dir for data)
if ! id -u nous-proxy &>/dev/null; then
    echo "Creating system user 'nous-proxy'..."
    useradd --system --no-create-home --shell /usr/sbin/nologin nous-proxy
fi

# 4. Set permissions
chown -R nous-proxy:nous-proxy "$PROJECT_DIR/data"
chmod 750 "$PROJECT_DIR/data"

# 5. Create .env if not exists
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "Creating .env from .env.example..."
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo ""
    echo ">>> Edit $PROJECT_DIR/.env to customize settings"
fi

# 6. Install systemd service
echo "Installing systemd service..."
cp "$PROJECT_DIR/scripts/nous-proxy.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable nous-proxy

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit /opt/nous-proxy/.env if needed"
echo "  2. Start the service:"
echo "     systemctl start nous-proxy"
echo "  3. Check status:"
echo "     systemctl status nous-proxy"
echo "  4. Authenticate (first time only):"
echo "     curl -X POST http://localhost:8090/auth/device-code -H 'Authorization: Bearer <key>'"
echo "     # Open URL in browser, enter code, then:"
echo "     curl -X POST http://localhost:8090/auth/poll -H 'Authorization: Bearer <key>'"
echo ""
echo "API key is auto-generated on first run. Check:"
echo "     cat $PROJECT_DIR/data/api_keys.json"
echo ""
