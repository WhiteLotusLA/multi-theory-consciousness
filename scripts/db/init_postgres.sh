#!/usr/bin/env bash
# Create the MTC application role and database on a Homebrew or apt
# PostgreSQL (14 or later), then apply migrations/postgres/*.sql as that role
# so it owns its tables. The Compose image creates the role and database from
# its own environment and does not need this script.
#
# Usage: scripts/db/init_postgres.sh [--env-file PATH] [--dry-run] [--no-migrate]
#
#   --env-file PATH  environment file to read (default: ./.env)
#   --dry-run        print the statements and the migration files; run nothing
#   --no-migrate     create the role and database only
#   --help           show this text
#
# Settings: only MTC_DB_PREFIX and the MTC_POSTGRES_* lines are read from the
# environment file, with a line parser (the file is never sourced or
# evaluated); an exported variable of the same name wins, as in Settings.
# Empty names derive exactly as Settings derives them: the database is
# <prefix>_consciousness and the role is <prefix>.
#
# The administrator connects through the standard PG* variables (PGHOST,
# PGPORT, PGUSER, PGPASSWORD, PGDATABASE). The role's password reaches psql
# through the environment only, never the command line. An existing role's
# password is never changed, and nothing is ever dropped.
#
# Exit codes: 0 done; 1 a psql step failed; 2 usage or configuration error.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MIGRATIONS_DIR="${REPO_ROOT}/migrations/postgres"
# Mirrors the default of Settings.DB_PREFIX; a test compares the two.
DEFAULT_DB_PREFIX="mtc"

usage() {
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

env_file=".env"
dry_run=0
migrate=1
while [ "$#" -gt 0 ]; do
    case "$1" in
        --env-file)
            if [ "$#" -lt 2 ]; then
                echo "init_postgres: --env-file needs a path" >&2
                exit 2
            fi
            env_file="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --no-migrate)
            migrate=0
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "init_postgres: unknown option $1 (see --help)" >&2
            exit 2
            ;;
    esac
done

file_prefix=""
file_db=""
file_user=""
file_password=""
file_host=""
file_port=""

strip_quotes() {
    local value="$1"
    case "${value}" in
        \"*\")
            value="${value#\"}"
            value="${value%\"}"
            ;;
        \'*\')
            value="${value#\'}"
            value="${value%\'}"
            ;;
    esac
    printf '%s' "${value}"
}

if [ -f "${env_file}" ]; then
    pattern='^[[:space:]]*(export[[:space:]]+)?(MTC_DB_PREFIX|MTC_POSTGRES_[A-Z_]+)[[:space:]]*=(.*)$'
    while IFS= read -r line || [ -n "${line}" ]; do
        line="${line%$'\r'}"
        if [[ "${line}" =~ ${pattern} ]]; then
            key="${BASH_REMATCH[2]}"
            value="$(strip_quotes "${BASH_REMATCH[3]}")"
            case "${key}" in
                MTC_DB_PREFIX) file_prefix="${value}" ;;
                MTC_POSTGRES_DB) file_db="${value}" ;;
                MTC_POSTGRES_USER) file_user="${value}" ;;
                MTC_POSTGRES_PASSWORD) file_password="${value}" ;;
                MTC_POSTGRES_HOST) file_host="${value}" ;;
                MTC_POSTGRES_PORT) file_port="${value}" ;;
            esac
        fi
    done < "${env_file}"
elif [ "${env_file}" != ".env" ]; then
    echo "init_postgres: environment file ${env_file} not found" >&2
    exit 2
fi

prefix="${MTC_DB_PREFIX:-${file_prefix}}"
prefix="${prefix:-${DEFAULT_DB_PREFIX}}"
database="${MTC_POSTGRES_DB:-${file_db}}"
database="${database:-${prefix}_consciousness}"
role="${MTC_POSTGRES_USER:-${file_user}}"
role="${role:-${prefix}}"
role_password="${MTC_POSTGRES_PASSWORD:-${file_password}}"
app_host="${MTC_POSTGRES_HOST:-${file_host}}"
app_port="${MTC_POSTGRES_PORT:-${file_port}}"

name_pattern='^[A-Za-z_][A-Za-z0-9_]*$'
for name in "${role}" "${database}"; do
    if ! [[ "${name}" =~ ${name_pattern} ]]; then
        echo "init_postgres: ${name} is not a plain identifier (letters, digits, underscores)" >&2
        exit 2
    fi
done

migration_files=()
if [ "${migrate}" -eq 1 ]; then
    for path in "${MIGRATIONS_DIR}"/*.sql; do
        if [ -f "${path}" ]; then
            migration_files+=("${path}")
        fi
    done
    if [ "${#migration_files[@]}" -eq 0 ]; then
        echo "init_postgres: no migrations found under ${MIGRATIONS_DIR}" >&2
        exit 2
    fi
fi

if [ -n "${role_password}" ]; then
    role_template="CREATE ROLE %I LOGIN PASSWORD %L"
else
    role_template="CREATE ROLE %I LOGIN"
fi

role_sql="\\set role \`printenv MTC_INIT_ROLE\`
\\set pw \`printenv MTC_INIT_PASSWORD\`
SELECT format('${role_template}', :'role', :'pw')
WHERE NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = :'role') \\gexec"

database_sql="\\set role \`printenv MTC_INIT_ROLE\`
\\set db \`printenv MTC_INIT_DATABASE\`
SELECT format('CREATE DATABASE %I OWNER %I', :'db', :'role')
WHERE NOT EXISTS (SELECT 1 FROM pg_catalog.pg_database WHERE datname = :'db') \\gexec"

role_exists_sql="\\set role \`printenv MTC_INIT_ROLE\`
SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = :'role';"

if [ "${dry_run}" -eq 1 ]; then
    echo "# dry run: nothing is executed"
    echo "# role ${role}, database ${database}"
    if [ -z "${role_password}" ]; then
        echo "# MTC_POSTGRES_PASSWORD is empty: the role is created without a password (trust authentication)"
    fi
    echo "# as the administrator (PG* variables): psql -X -v ON_ERROR_STOP=1 -f -"
    printf '%s\n' "${role_sql}"
    printf '%s\n' "${database_sql}"
    if [ "${migrate}" -eq 1 ]; then
        echo "# as ${role} on ${database} (password through PGPASSWORD):"
        for path in "${migration_files[@]}"; do
            echo "psql -X -v ON_ERROR_STOP=1 -f ${path}"
        done
    fi
    exit 0
fi

if ! command -v psql > /dev/null 2>&1; then
    echo "init_postgres: psql not found; install the PostgreSQL client tools" >&2
    exit 2
fi

export MTC_INIT_ROLE="${role}"
export MTC_INIT_DATABASE="${database}"
export MTC_INIT_PASSWORD="${role_password}"

if [ -z "${role_password}" ]; then
    echo "init_postgres: warning: MTC_POSTGRES_PASSWORD is empty; the role gets no password and needs trust authentication" >&2
fi

existing="$(printf '%s\n' "${role_exists_sql}" | psql -X -q -t -A -v ON_ERROR_STOP=1 -f -)" || {
    echo "init_postgres: the administrator connection failed (check the PG* variables)" >&2
    exit 1
}
if [ -n "${existing}" ]; then
    echo "init_postgres: note: role ${role} already exists; its password was not changed"
fi

printf '%s\n' "${role_sql}" | psql -X -q -v ON_ERROR_STOP=1 -f - || {
    echo "init_postgres: creating the role failed" >&2
    exit 1
}
printf '%s\n' "${database_sql}" | psql -X -q -v ON_ERROR_STOP=1 -f - || {
    echo "init_postgres: creating the database failed" >&2
    exit 1
}
unset MTC_INIT_PASSWORD
echo "init_postgres: role ${role} and database ${database} are present"

if [ "${migrate}" -eq 0 ]; then
    exit 0
fi

# The role connects to the server the application uses; the administrator's
# password is not passed on.
if [ -n "${app_host}" ]; then
    export PGHOST="${app_host}"
fi
if [ -n "${app_port}" ]; then
    export PGPORT="${app_port}"
fi
unset PGPASSWORD
for path in "${migration_files[@]}"; do
    status=0
    if [ -n "${role_password}" ]; then
        PGPASSWORD="${role_password}" PGUSER="${role}" PGDATABASE="${database}" \
            psql -X -q -v ON_ERROR_STOP=1 -f "${path}" || status=$?
    else
        PGUSER="${role}" PGDATABASE="${database}" \
            psql -X -q -v ON_ERROR_STOP=1 -f "${path}" || status=$?
    fi
    if [ "${status}" -ne 0 ]; then
        echo "init_postgres: applying $(basename "${path}") as ${role} failed" >&2
        exit 1
    fi
    echo "init_postgres: applied $(basename "${path}")"
done
