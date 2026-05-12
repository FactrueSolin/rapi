#!/bin/sh

set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
CLI_BIN="$REPO_ROOT/target/debug/privacy_filter_test_cli"

PASS_COUNT=0
FAIL_COUNT=0

log_section() {
    printf '\n== %s ==\n' "$1"
}

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    printf 'PASS: %s\n' "$1"
}

fail() {
    FAIL_COUNT=$((FAIL_COUNT + 1))
    printf 'FAIL: %s -- %s\n' "$1" "$2"
}

ensure_cli() {
    printf '[setup] building privacy_filter_test_cli\n'
    if ! (cd "$REPO_ROOT" && cargo build --quiet --bin privacy_filter_test_cli); then
        printf 'FAIL: setup -- failed to build privacy_filter_test_cli\n'
        exit 1
    fi
}

run_cli() {
    request=$1
    output_file=$2
    printf '%s' "$request" | "$CLI_BIN" >"$output_file" 2>&1
}

assert_status() {
    name=$1
    actual=$2
    expected=$3
    output_file=$4
    if [ "$actual" -eq "$expected" ]; then
        pass "$name status"
    else
        fail "$name status" "expected exit $expected, got $actual, output=$(cat "$output_file")"
    fi
}

assert_contains() {
    name=$1
    output_file=$2
    expected=$3
    if grep -F "$expected" "$output_file" >/dev/null 2>&1; then
        pass "$name contains '$expected'"
    else
        fail "$name contains '$expected'" "output=$(cat "$output_file")"
    fi
}

assert_not_contains() {
    name=$1
    output_file=$2
    unexpected=$3
    if grep -F "$unexpected" "$output_file" >/dev/null 2>&1; then
        fail "$name does not contain '$unexpected'" "output=$(cat "$output_file")"
    else
        pass "$name does not contain '$unexpected'"
    fi
}

assert_json_ok() {
    name=$1
    output_file=$2
    assert_contains "$name" "$output_file" '"ok":true'
}

assert_json_error_code() {
    name=$1
    output_file=$2
    code=$3
    assert_contains "$name" "$output_file" '"ok":false'
    assert_contains "$name" "$output_file" '"code":"'"$code"'"'
}

finish_tests() {
    printf '\nSummary: %s passed, %s failed\n' "$PASS_COUNT" "$FAIL_COUNT"
    if [ "$FAIL_COUNT" -ne 0 ]; then
        exit 1
    fi
}
