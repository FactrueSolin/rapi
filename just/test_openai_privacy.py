#!/usr/bin/env python3
"""
Test script for OpenAI Privacy Filter API service.

Tests cover:
- Health endpoint (correct response, status code)
- Redact endpoint with various PII types
- Edge cases (empty text, long text, special characters)
- Error cases (missing body, invalid JSON, wrong method)

Usage:
    python test_openai_privacy.py [--base-url URL]

Exit codes:
    0 - All tests passed
    1 - One or more tests failed
"""

import argparse
import json
import sys
import urllib.request
import urllib.error


DEFAULT_BASE_URL = "http://localhost:8000"


class TestRunner:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.passed = 0
        self.failed = 0
        self.results = []

    def record(self, name, ok, detail=""):
        if ok:
            self.passed += 1
            self.results.append(f"  PASS: {name}")
        else:
            self.failed += 1
            msg = f"  FAIL: {name}"
            if detail:
                msg += f" -- {detail}"
            self.results.append(msg)

    def get(self, path):
        url = f"{self.base_url}{path}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
        except urllib.error.URLError as e:
            raise ConnectionError(f"Cannot connect to {url}: {e.reason}")

    def post(self, path, data=None, raw_body=None, content_type="application/json"):
        url = f"{self.base_url}{path}"
        if raw_body is not None:
            body = raw_body.encode("utf-8") if isinstance(raw_body, str) else raw_body
        elif data is not None:
            body = json.dumps(data).encode("utf-8")
        else:
            body = b""
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_body = resp.read()
                try:
                    return resp.status, json.loads(resp_body)
                except json.JSONDecodeError:
                    return resp.status, resp_body.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            resp_body = e.read()
            try:
                return e.code, json.loads(resp_body)
            except json.JSONDecodeError:
                return e.code, resp_body.decode("utf-8", errors="replace")

    # ============================================================
    # Health endpoint tests
    # ============================================================

    def test_health_returns_200(self):
        name = "GET /health returns 200"
        try:
            status, _ = self.get("/health")
            self.record(name, status == 200, f"expected 200, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_health_response_structure(self):
        name = "GET /health has correct response structure"
        try:
            _, body = self.get("/health")
            has_fields = all(k in body for k in ("status", "model_loaded", "device"))
            self.record(name, has_fields, f"missing fields in: {body}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_health_status_value(self):
        name = "GET /health status field is 'healthy'"
        try:
            _, body = self.get("/health")
            self.record(name, body.get("status") == "healthy", f"got status={body.get('status')}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_health_model_loaded(self):
        name = "GET /health model_loaded is true"
        try:
            _, body = self.get("/health")
            self.record(name, body.get("model_loaded") is True, f"got model_loaded={body.get('model_loaded')}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_health_device_field(self):
        name = "GET /health device is cuda or cpu"
        try:
            _, body = self.get("/health")
            device = body.get("device", "")
            self.record(name, device in ("cuda", "cpu"), f"got device={device}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    # ============================================================
    # Redact endpoint - PII detection tests
    # ============================================================

    def test_redact_person_name(self):
        name = "POST /redact detects person name"
        try:
            status, body = self.post("/redact", {"text": "My name is John Smith"})
            if status != 200:
                self.record(name, False, f"status={status}, body={body}")
                return
            has_person = any("person" in p["replacement"].lower() for p in body.get("pairs", []))
            self.record(name, has_person, f"no person detection in pairs: {body.get('pairs')}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_email(self):
        name = "POST /redact detects email address"
        try:
            status, body = self.post("/redact", {"text": "Contact me at john.doe@example.com"})
            if status != 200:
                self.record(name, False, f"status={status}, body={body}")
                return
            has_email = any("email" in p["replacement"].lower() for p in body.get("pairs", []))
            self.record(name, has_email, f"no email detection in pairs: {body.get('pairs')}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_address(self):
        name = "POST /redact detects address"
        try:
            status, body = self.post("/redact", {"text": "I live at 123 Main Street, New York"})
            if status != 200:
                self.record(name, False, f"status={status}, body={body}")
                return
            has_address = any("address" in p["replacement"].lower() for p in body.get("pairs", []))
            self.record(name, has_address, f"no address detection in pairs: {body.get('pairs')}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_phone(self):
        name = "POST /redact detects phone number"
        try:
            status, body = self.post("/redact", {"text": "Call me at +1-555-123-4567"})
            if status != 200:
                self.record(name, False, f"status={status}, body={body}")
                return
            has_phone = any("phone" in p["replacement"].lower() for p in body.get("pairs", []))
            self.record(name, has_phone, f"no phone detection in pairs: {body.get('pairs')}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_multiple_pii(self):
        name = "POST /redact detects multiple PII types in one text"
        try:
            status, body = self.post("/redact", {"text": "John Smith lives at 123 Main St and his email is john@example.com"})
            if status != 200:
                self.record(name, False, f"status={status}, body={body}")
                return
            pair_count = len(body.get("pairs", []))
            self.record(name, pair_count >= 2, f"expected >= 2 pairs, got {pair_count}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_returns_pairs(self):
        name = "POST /redact response has 'pairs' field"
        try:
            status, body = self.post("/redact", {"text": "Hello World"})
            if status != 200:
                self.record(name, False, f"status={status}, body={body}")
                return
            self.record(name, "pairs" in body, f"missing 'pairs' in response: {body}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_returns_redacted_text(self):
        name = "POST /redact response has 'redacted_text' field"
        try:
            status, body = self.post("/redact", {"text": "Hello World"})
            if status != 200:
                self.record(name, False, f"status={status}, body={body}")
                return
            self.record(name, "redacted_text" in body, f"missing 'redacted_text' in response: {body}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_pair_structure(self):
        name = "POST /redact each pair has 'original' and 'replacement' fields"
        try:
            status, body = self.post("/redact", {"text": "My name is John Smith and email is john@test.com"})
            if status != 200:
                self.record(name, False, f"status={status}, body={body}")
                return
            pairs = body.get("pairs", [])
            all_valid = all("original" in p and "replacement" in p for p in pairs) if pairs else True
            self.record(name, all_valid, f"invalid pair structure: {pairs}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_text_contains_placeholders(self):
        name = "POST /redact redacted_text contains placeholder tokens"
        try:
            status, body = self.post("/redact", {"text": "My name is John Smith"})
            if status != 200:
                self.record(name, False, f"status={status}, body={body}")
                return
            redacted = body.get("redacted_text", "")
            has_placeholder = "<PRIVATE_" in redacted and ">" in redacted
            self.record(name, has_placeholder, f"no placeholder in redacted_text: {redacted}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_original_not_in_redacted_text(self):
        name = "POST /redact original PII text is NOT in redacted_text"
        try:
            status, body = self.post("/redact", {"text": "My name is John Smith"})
            if status != 200:
                self.record(name, False, f"status={status}, body={body}")
                return
            redacted = body.get("redacted_text", "")
            pairs = body.get("pairs", [])
            originals_removed = all(p["original"] not in redacted for p in pairs)
            self.record(name, originals_removed, f"original text still in redacted: {redacted}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    # ============================================================
    # Redact endpoint - Edge case tests
    # ============================================================

    def test_redact_empty_string(self):
        name = "POST /redact with empty string returns success"
        try:
            status, _ = self.post("/redact", {"text": ""})
            self.record(name, status == 200, f"expected 200, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_empty_string_no_pairs(self):
        name = "POST /redact with empty string returns empty pairs"
        try:
            status, body = self.post("/redact", {"text": ""})
            if status != 200:
                self.record(name, False, f"status={status}")
                return
            self.record(name, body.get("pairs") == [], f"expected empty pairs, got {body.get('pairs')}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_no_pii_text(self):
        name = "POST /redact with text containing no PII returns empty pairs"
        try:
            status, body = self.post("/redact", {"text": "The quick brown fox jumps over the lazy dog"})
            if status != 200:
                self.record(name, False, f"status={status}")
                return
            self.record(name, body.get("pairs") == [], f"expected empty pairs, got {body.get('pairs')}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_no_pii_returns_original_text(self):
        name = "POST /redact with no PII returns original text as redacted_text"
        try:
            original = "The quick brown fox jumps over the lazy dog"
            status, body = self.post("/redact", {"text": original})
            if status != 200:
                self.record(name, False, f"status={status}")
                return
            self.record(name, body.get("redacted_text") == original, f"redacted_text differs: {body.get('redacted_text')}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_special_characters(self):
        name = "POST /redact handles special characters"
        try:
            status, _ = self.post("/redact", {"text": "Hello! @#$%^&*()_+-=[]{}|;':\",./<>?"})
            self.record(name, status == 200, f"expected 200, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_unicode_characters(self):
        name = "POST /redact handles unicode characters"
        try:
            status, _ = self.post("/redact", {"text": "你好世界 Hello 世界"})
            self.record(name, status == 200, f"expected 200, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_emoji(self):
        name = "POST /redact handles emoji characters"
        try:
            status, _ = self.post("/redact", {"text": "Hello World!"})
            self.record(name, status == 200, f"expected 200, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_very_long_text(self):
        name = "POST /redact handles long text (50000 chars)"
        try:
            long_text = "This is a test sentence. " * 2000
            status, _ = self.post("/redact", {"text": long_text})
            self.record(name, status == 200, f"expected 200, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_whitespace_only(self):
        name = "POST /redact handles whitespace-only text"
        try:
            status, _ = self.post("/redact", {"text": "   \t\n   "})
            self.record(name, status == 200, f"expected 200, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_newlines_only(self):
        name = "POST /redact handles newlines-only text"
        try:
            status, _ = self.post("/redact", {"text": "\n\n\n"})
            self.record(name, status == 200, f"expected 200, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_html_content(self):
        name = "POST /redact handles HTML content with email"
        try:
            status, body = self.post("/redact", {"text": "<div><p>Contact: john@example.com</p></div>"})
            if status != 200:
                self.record(name, False, f"status={status}")
                return
            has_email = any("email" in p["replacement"].lower() for p in body.get("pairs", []))
            self.record(name, has_email, f"no email detection in HTML: {body.get('pairs')}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_json_in_text(self):
        name = "POST /redact handles JSON-like content in text"
        try:
            status, _ = self.post("/redact", {"text": '{"name": "John Smith", "email": "john@example.com"}'})
            if status != 200:
                self.record(name, False, f"status={status}")
                return
            self.record(name, True)
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_url_in_text(self):
        name = "POST /redact detects URL in text"
        try:
            status, body = self.post("/redact", {"text": "Visit https://example.com/users/john for more info"})
            if status != 200:
                self.record(name, False, f"status={status}")
                return
            has_url = any("url" in p["replacement"].lower() for p in body.get("pairs", []))
            self.record(name, has_url, f"no URL detection in pairs: {body.get('pairs')}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    # ============================================================
    # Redact endpoint - Error case tests
    # ============================================================

    def test_redact_missing_text_field(self):
        name = "POST /redact with missing 'text' field returns 422"
        try:
            status, _ = self.post("/redact", {"wrong_field": "value"})
            self.record(name, status == 422, f"expected 422, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_invalid_json_body(self):
        name = "POST /redact with invalid JSON returns 422"
        try:
            status, _ = self.post("/redact", raw_body="not valid json{{{")
            self.record(name, status == 422, f"expected 422, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_text_not_string(self):
        name = "POST /redact with non-string text field returns 422"
        try:
            status, _ = self.post("/redact", {"text": 12345})
            self.record(name, status == 422, f"expected 422, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_text_is_null(self):
        name = "POST /redact with null text field returns 422"
        try:
            status, _ = self.post("/redact", {"text": None})
            self.record(name, status == 422, f"expected 422, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_empty_body(self):
        name = "POST /redact with empty body returns 422"
        try:
            status, _ = self.post("/redact", raw_body="")
            self.record(name, status == 422, f"expected 422, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_get_on_redact_returns_405(self):
        name = "GET /redact returns 405 Method Not Allowed"
        try:
            status, _ = self.get("/redact")
            self.record(name, status == 405, f"expected 405, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_unknown_endpoint_returns_404(self):
        name = "GET /unknown returns 404"
        try:
            status, _ = self.get("/unknown")
            self.record(name, status == 404, f"expected 404, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_post_to_unknown_endpoint_returns_404(self):
        name = "POST /unknown returns 404"
        try:
            status, _ = self.post("/unknown", {"text": "test"})
            self.record(name, status == 404, f"expected 404, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_text_is_list(self):
        name = "POST /redact with text as list returns 422"
        try:
            status, _ = self.post("/redact", {"text": ["hello", "world"]})
            self.record(name, status == 422, f"expected 422, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    def test_redact_text_is_object(self):
        name = "POST /redact with text as object returns 422"
        try:
            status, _ = self.post("/redact", {"text": {"key": "value"}})
            self.record(name, status == 422, f"expected 422, got {status}")
        except ConnectionError as e:
            self.record(name, False, str(e))

    # ============================================================
    # Run all
    # ============================================================

    def run_all(self):
        tests = [
            # Health endpoint
            self.test_health_returns_200,
            self.test_health_response_structure,
            self.test_health_status_value,
            self.test_health_model_loaded,
            self.test_health_device_field,

            # Redact - PII detection
            self.test_redact_person_name,
            self.test_redact_email,
            self.test_redact_address,
            self.test_redact_phone,
            self.test_redact_multiple_pii,
            self.test_redact_returns_pairs,
            self.test_redact_returns_redacted_text,
            self.test_redact_pair_structure,
            self.test_redact_text_contains_placeholders,
            self.test_redact_original_not_in_redacted_text,

            # Redact - Edge cases
            self.test_redact_empty_string,
            self.test_redact_empty_string_no_pairs,
            self.test_redact_no_pii_text,
            self.test_redact_no_pii_returns_original_text,
            self.test_redact_special_characters,
            self.test_redact_unicode_characters,
            self.test_redact_emoji,
            self.test_redact_very_long_text,
            self.test_redact_whitespace_only,
            self.test_redact_newlines_only,
            self.test_redact_html_content,
            self.test_redact_json_in_text,
            self.test_redact_url_in_text,

            # Redact - Error cases
            self.test_redact_missing_text_field,
            self.test_redact_invalid_json_body,
            self.test_redact_text_not_string,
            self.test_redact_text_is_null,
            self.test_redact_empty_body,
            self.test_get_on_redact_returns_405,
            self.test_unknown_endpoint_returns_404,
            self.test_post_to_unknown_endpoint_returns_404,
            self.test_redact_text_is_list,
            self.test_redact_text_is_object,
        ]

        for test_fn in tests:
            test_fn()

        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        for r in self.results:
            print(r)
        print("=" * 60)
        print(f"Total: {self.passed + self.failed} | Passed: {self.passed} | Failed: {self.failed}")
        print("=" * 60)

        if self.failed > 0:
            print(f"\n{self.failed} test(s) FAILED!")
            return 1
        else:
            print(f"\nAll {self.passed} test(s) PASSED!")
            return 0


def main():
    parser = argparse.ArgumentParser(description="Test OpenAI Privacy Filter API")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Base URL of the API service (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()

    print(f"Testing API at: {args.base_url}")
    runner = TestRunner(args.base_url)
    sys.exit(runner.run_all())


if __name__ == "__main__":
    main()
