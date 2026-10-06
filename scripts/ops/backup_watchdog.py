#!/usr/bin/env python3
"""Check that the newest backup is fresh and complete.

Usage: python3 scripts/ops/backup_watchdog.py [--max-age-hours N]
           [--min-bytes N] [--json] [--alert]

Wraps mtc.monitoring.backup_status.check_backups for a timer (cron, a launch
agent or a systemd timer). It reads file names and sizes under
BACKUP_LOCATION and the manifest's status lines; it never opens a dump, and
it never makes, repairs or deletes a backup.

  --max-age-hours N  the newest backup may be this old (default 26: a daily
                     backup with two hours of slack)
  --min-bytes N      a store's file smaller than this counts as missing
                     (default 1)
  --json             print the status as JSON
  --alert            when the check does not pass, also raise a backup alert
                     through the notifiers ALERT_NOTIFIERS names

Exit codes: 0 the newest backup is fresh and complete; 1 it is stale,
incomplete or absent, or its manifest records a failed dump; 2 usage or
settings that do not load; 3 BACKUP_LOCATION does not exist.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_UNREACHABLE = 3


async def raise_alert(settings: Any, report: dict) -> bool:
    """Send one backup alert; True when the alert manager accepted it."""
    from mtc.monitoring.alert_manager import AlertManager, AlertSeverity, AlertType

    manager = AlertManager.from_settings(settings)
    alert = await manager.send_alert(
        AlertType.BACKUP,
        AlertSeverity.URGENT,
        "Backup check did not pass",
        str(report.get("detail") or report.get("status")),
        data=report,
    )
    return alert is not None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--max-age-hours", type=float, default=None, metavar="N")
    parser.add_argument("--min-bytes", type=int, default=1, metavar="N")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--alert", action="store_true")
    args = parser.parse_args(argv)

    from mtc.core.checks import CheckStatus
    from mtc.core.config import Settings
    from mtc.monitoring.backup_status import DEFAULT_MAX_AGE_HOURS, check_backups

    max_age = (
        DEFAULT_MAX_AGE_HOURS if args.max_age_hours is None else args.max_age_hours
    )
    if max_age <= 0 or args.min_bytes < 0:
        print("backup_watchdog: the limits must be positive", file=sys.stderr)
        return EXIT_USAGE
    try:
        settings = Settings()
    except Exception as exc:  # noqa: BLE001 - named by type, never by value
        print(
            f"backup_watchdog: the settings do not load ({type(exc).__name__})",
            file=sys.stderr,
        )
        return EXIT_USAGE

    status = check_backups(settings, max_age_hours=max_age, min_bytes=args.min_bytes)
    report = status.as_dict()
    if status.status is CheckStatus.OK:
        code = EXIT_OK
    elif status.status is CheckStatus.UNREACHABLE:
        code = EXIT_UNREACHABLE
    else:
        code = EXIT_FAILED
    report["exit_code"] = code
    if args.alert and code != EXIT_OK:
        report["alerted"] = bool(asyncio.run(raise_alert(settings, dict(report))))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"backup: {status.status.value}: {status.detail}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
