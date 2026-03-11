#!/bin/bash
set -e

# Run Alembic migrations if using PostgreSQL
if [[ "$DATABASE_URL" == postgresql://* ]]; then
    echo "Running database migrations..."
    cd /app
    alembic upgrade head
    echo "Migrations complete."
fi

# Execute the main command
exec "$@"
