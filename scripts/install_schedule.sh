#!/usr/bin/env bash
# Install (or reinstall) the macOS launchd job that runs the pipeline after each US close.
#
#   scripts/install_schedule.sh               # 11:00 Tue-Sat local time (suits New Zealand)
#   HOUR=7 MINUTE=30 DAYS="2 3 4 5 6" scripts/install_schedule.sh
#   DRY_RUN=1 PLIST_OUT=/tmp/test.plist scripts/install_schedule.sh   # write the file only
#
# Pick a local time after 16:30 New York time on US trading days. If the job can't run, for example
# because macOS blocks background access to ~/Documents, you get a desktop notification instead of a
# silent failure (see README: "Running it automatically").
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
HOUR="${HOUR:-11}"
MINUTE="${MINUTE:-0}"
DAYS="${DAYS:-2 3 4 5 6}"
LABEL="com.smp.pipeline"
PLIST="${PLIST_OUT:-$HOME/Library/LaunchAgents/$LABEL.plist}"   # PLIST_OUT: write elsewhere (testing)
LOGDIR="$HOME/Library/Logs/smp"   # outside ~/Documents, so launchd can always write it
mkdir -p "$LOGDIR" "$(dirname "$PLIST")"

# The job's command is inline (not a script file in the project) so that a permission problem with the
# project folder still produces a notification instead of failing before anything runs.
read -r -d '' COMMAND <<CMD || true
cd "$PROJECT" && ./.venv/bin/smp run-all
status=\$?
if [ \$status -ne 0 ]; then
  /usr/bin/osascript -e "display notification \"The daily run failed (exit \$status). Log: $LOGDIR/pipeline.err\" with title \"Stock Market Predictor\""
fi
exit \$status
CMD

intervals=""
for d in $DAYS; do
  intervals+="    <dict><key>Weekday</key><integer>$d</integer><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>$MINUTE</integer></dict>"$'\n'
done

escaped=$(printf '%s' "$COMMAND" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g')
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-c</string>
    <string>$escaped</string>
  </array>
  <key>StartCalendarInterval</key>
  <array>
$intervals  </array>
  <key>StandardOutPath</key><string>$LOGDIR/pipeline.out</string>
  <key>StandardErrorPath</key><string>$LOGDIR/pipeline.err</string>
</dict>
</plist>
PLIST

plutil -lint "$PLIST" >/dev/null
if [ "${DRY_RUN:-0}" = "1" ]; then echo "Wrote $PLIST (dry run, not loaded)"; exit 0; fi
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "Installed $PLIST: weekdays $DAYS at $(printf '%02d:%02d' "$HOUR" "$MINUTE") local time."
echo "Logs: $LOGDIR. Test now with: launchctl kickstart gui/\$(id -u)/$LABEL"
