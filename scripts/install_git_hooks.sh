#!/usr/bin/env bash
# Install the pre-commit privacy audit hook into this clone.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Resolve the hooks directory through git so worktrees and core.hooksPath work.
HOOKS_DIR="$(git -C "$ROOT" rev-parse --path-format=absolute --git-path hooks)"
HOOK="$HOOKS_DIR/pre-commit"
mkdir -p "$HOOKS_DIR"
cat > "$HOOK" <<'HOOKEOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
python3 scripts/privacy_audit.py --staged
HOOKEOF
chmod +x "$HOOK"
echo "installed $HOOK"
