#!/bin/sh

set -u

. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/privacy_filter_common.sh"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/privacy-filter-invalid.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT INT HUP TERM

ensure_cli
log_section "privacy filter invalid input and injection rejection"

name="rejects malformed json"
out="$TMP_DIR/malformed.json"
run_cli '{"text":"alice@example.com"' "$out"
status=$?
assert_status "$name" "$status" 1 "$out"
assert_json_error_code "$name" "$out" "invalid_json"

name="rejects unknown model path injection field"
out="$TMP_DIR/model-path-injection.json"
request='{"text":"alice@example.com","model_path":"/tmp/evil.onnx"}'
run_cli "$request" "$out"
status=$?
assert_status "$name" "$status" 1 "$out"
assert_json_error_code "$name" "$out" "invalid_json"
assert_contains "$name" "$out" 'unknown field'

name="rejects unknown endpoint injection field"
out="$TMP_DIR/endpoint-injection.json"
request='{"text":"alice@example.com","endpoint":"https://attacker.invalid/model"}'
run_cli "$request" "$out"
status=$?
assert_status "$name" "$status" 1 "$out"
assert_json_error_code "$name" "$out" "invalid_json"
assert_contains "$name" "$out" 'unknown field'

name="rejects empty text"
out="$TMP_DIR/empty-text.json"
request='{"text":""}'
run_cli "$request" "$out"
status=$?
assert_status "$name" "$status" 1 "$out"
assert_json_error_code "$name" "$out" "invalid_input"
assert_contains "$name" "$out" 'text must not be empty'

name="rejects invalid output mode"
out="$TMP_DIR/output-mode.json"
request='{"text":"alice@example.com","output_mode":"../../evil"}'
run_cli "$request" "$out"
status=$?
assert_status "$name" "$status" 1 "$out"
assert_json_error_code "$name" "$out" "invalid_json"

name="rejects invalid context window"
out="$TMP_DIR/context-window.json"
request='{"text":"alice@example.com","context_window_length":0}'
run_cli "$request" "$out"
status=$?
assert_status "$name" "$status" 1 "$out"
assert_json_error_code "$name" "$out" "invalid_input"
assert_contains "$name" "$out" 'context_window_length must be greater than 0'

finish_tests
