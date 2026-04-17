#!/bin/sh
set -e

# Fix data dir ownership at startup (needed for bind mounts)
# This runs as root before we drop privileges
if [ -d /app/data ]; then
  chown -R app:app /app/data 2>/dev/null || true
fi

# Drop root and run as app user
# Note: PATH is inherited, no --reset-env to preserve it
exec setpriv --reuid=app --regid=app --init-groups "$@"
