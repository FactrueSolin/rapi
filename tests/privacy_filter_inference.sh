#!/bin/sh

set -u

. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/privacy_filter_common.sh"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/privacy-filter-inference.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT INT HUP TERM

ensure_cli
log_section "privacy filter q4 inference"

name="redacts email with typed placeholder"
out="$TMP_DIR/email.json"
request='{"text":"Contact Alice at alice@example.com for the deployment key."}'
run_cli "$request" "$out"
status=$?
assert_status "$name" "$status" 0 "$out"
if [ "$status" -eq 0 ]; then
    assert_json_ok "$name" "$out"
    assert_contains "$name" "$out" '"variant":"q4"'
    assert_contains "$name" "$out" 'alice@example.com'
    assert_contains "$name" "$out" '"span_count":'
    assert_not_contains "$name redacted_text" "$out" '"redacted_text":"Contact Alice at alice@example.com for the deployment key."'
fi

name="supports generic redacted output mode"
out="$TMP_DIR/redacted.json"
request='{"text":"Send the receipt to bob@example.com before Friday.","output_mode":"redacted"}'
run_cli "$request" "$out"
status=$?
assert_status "$name" "$status" 0 "$out"
if [ "$status" -eq 0 ]; then
    assert_json_ok "$name" "$out"
    assert_contains "$name" "$out" '"variant":"q4"'
    assert_contains "$name" "$out" '<REDACTED>'
    assert_not_contains "$name redacted_text" "$out" '"redacted_text":"Send the receipt to bob@example.com before Friday."'
fi

name="leaves non sensitive text callable"
out="$TMP_DIR/plain.json"
request='{"text":"The build finished successfully and no customer data is present."}'
run_cli "$request" "$out"
status=$?
assert_status "$name" "$status" 0 "$out"
if [ "$status" -eq 0 ]; then
    assert_json_ok "$name" "$out"
    assert_contains "$name" "$out" '"variant":"q4"'
    assert_contains "$name" "$out" '"redacted_text":"The build finished successfully and no customer data is present."'
fi

finish_tests
