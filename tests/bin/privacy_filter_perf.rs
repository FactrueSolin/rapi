use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use rapi::privacy_filter::{
    DecodeMode, ExecutionProviderProbeAttempt, OnnxExecutionProvider,
    OnnxExecutionProviderPreference, OnnxSessionError, OnnxSessionOptions, OutputMode,
    PrivacyFilterModelManager, PrivacyFilterOnnxVariant, PrivacyFilterPipeline,
    PrivacyFilterPipelineError, PrivacyFilterPipelineMetrics, PrivacyFilterPipelineOptions,
    ResolvedModelPaths, TokenizerOptions,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

#[derive(Debug, Clone)]
struct Args {
    cache_dir: PathBuf,
    input_path: Option<PathBuf>,
    variant: PrivacyFilterOnnxVariant,
    requests: usize,
    warmup: usize,
    concurrency: usize,
    context_window_length: Option<usize>,
    provider: OnnxExecutionProviderPreference,
    intra_threads: Option<usize>,
    inter_threads: Option<usize>,
    decode_mode: DecodeMode,
    output_jsonl: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputRecord {
    text: String,
}

#[derive(Debug, Serialize)]
struct PerfRecord {
    request_index: usize,
    worker_index: usize,
    success: bool,
    error: Option<String>,
    text_bytes: usize,
    token_count: usize,
    window_count: usize,
    detected_span_count: usize,
    latency_ms: f64,
    tokenize_ms: f64,
    window_ms: f64,
    onnx_ms: f64,
    scoring_ms: f64,
    decode_ms: f64,
    span_ms: f64,
    redaction_ms: f64,
    total_ms: f64,
}

#[derive(Debug, Serialize)]
struct PerfSummary {
    variant: &'static str,
    requested_provider: &'static str,
    selected_provider: &'static str,
    requests: usize,
    warmup_per_worker: usize,
    concurrency: usize,
    successes: usize,
    failures: usize,
    elapsed_ms: f64,
    throughput_rps: f64,
    latency_ms: LatencySummary,
    onnx_ms: LatencySummary,
    tokenize_ms: LatencySummary,
    decode_ms: LatencySummary,
    total_text_bytes: usize,
    total_tokens: usize,
    total_windows: usize,
    avg_text_bytes_per_request: f64,
    avg_tokens_per_request: f64,
    avg_windows_per_request: f64,
    tokens_per_second: f64,
    windows_per_second: f64,
}

#[derive(Debug, Serialize)]
struct LatencySummary {
    min: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    max: f64,
    mean: f64,
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("privacy_filter_perf failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let args = Args::parse()?;
    let texts = load_texts(&args)?;
    let manager = PrivacyFilterModelManager::with_cache_dir(args.cache_dir.clone());
    let paths = manager
        .ensure_downloaded(args.variant, |_| {})
        .await
        .map_err(|err| err.to_string())?;

    let mut pipelines = Vec::with_capacity(args.concurrency);
    for _ in 0..args.concurrency {
        pipelines.push(Mutex::new(build_pipeline(&paths, &args)?));
    }
    let selected_provider = pipelines[0].lock().await.session().provider();
    let pipelines = Arc::new(pipelines);
    run_warmup(&pipelines, &texts, args.warmup).await?;

    let jobs = Arc::new(Mutex::new(build_jobs(&texts, args.requests)));
    let started = Instant::now();

    let mut handles = Vec::with_capacity(args.concurrency);
    for worker_index in 0..args.concurrency {
        let pipelines = pipelines.clone();
        let jobs = jobs.clone();
        handles.push(tokio::task::spawn_blocking(move || {
            let mut records = Vec::new();
            loop {
                let job = jobs.blocking_lock().pop_front();
                let Some((request_index, text)) = job else {
                    break;
                };

                let request_start = Instant::now();
                let result = pipelines[worker_index]
                    .blocking_lock()
                    .redact_with_metrics(&text);
                let latency_ms = elapsed_ms(request_start);
                records.push(match result {
                    Ok(run) => {
                        record_from_metrics(request_index, worker_index, latency_ms, &run.metrics)
                    }
                    Err(error) => PerfRecord {
                        request_index,
                        worker_index,
                        success: false,
                        error: Some(error.to_string()),
                        text_bytes: text.len(),
                        token_count: 0,
                        window_count: 0,
                        detected_span_count: 0,
                        latency_ms,
                        tokenize_ms: 0.0,
                        window_ms: 0.0,
                        onnx_ms: 0.0,
                        scoring_ms: 0.0,
                        decode_ms: 0.0,
                        span_ms: 0.0,
                        redaction_ms: 0.0,
                        total_ms: latency_ms,
                    },
                });
            }
            records
        }));
    }

    let mut records = Vec::new();
    for handle in handles {
        records.extend(handle.await.map_err(|err| err.to_string())?);
    }
    records.sort_by_key(|record| record.request_index);
    let elapsed_ms = elapsed_ms(started);

    if args.output_jsonl {
        for record in &records {
            println!(
                "{}",
                serde_json::to_string(record).map_err(|err| err.to_string())?
            );
        }
    } else {
        let summary = summarize(&args, &records, elapsed_ms, selected_provider);
        println!(
            "{}",
            serde_json::to_string_pretty(&summary).map_err(|err| err.to_string())?
        );
    }

    Ok(())
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            cache_dir: std::env::var("PRIVACY_FILTER_MODEL_CACHE_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from(".model")),
            input_path: None,
            variant: PrivacyFilterOnnxVariant::Q4,
            requests: 20,
            warmup: 0,
            concurrency: 1,
            context_window_length: None,
            provider: OnnxExecutionProviderPreference::Auto,
            intra_threads: None,
            inter_threads: None,
            decode_mode: DecodeMode::Viterbi,
            output_jsonl: false,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--cache-dir" => parsed.cache_dir = PathBuf::from(next_value(&mut args, &arg)?),
                "--input" => parsed.input_path = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                "--variant" => parsed.variant = parse_variant(&next_value(&mut args, &arg)?)?,
                "--requests" => parsed.requests = parse_usize(&next_value(&mut args, &arg)?, &arg)?,
                "--warmup" => parsed.warmup = parse_usize(&next_value(&mut args, &arg)?, &arg)?,
                "--concurrency" => {
                    parsed.concurrency = parse_usize(&next_value(&mut args, &arg)?, &arg)?
                }
                "--context-window-length" => {
                    parsed.context_window_length =
                        Some(parse_usize(&next_value(&mut args, &arg)?, &arg)?)
                }
                "--provider" => parsed.provider = parse_provider(&next_value(&mut args, &arg)?)?,
                "--intra-threads" => {
                    parsed.intra_threads = Some(parse_usize(&next_value(&mut args, &arg)?, &arg)?)
                }
                "--inter-threads" => {
                    parsed.inter_threads = Some(parse_usize(&next_value(&mut args, &arg)?, &arg)?)
                }
                "--decode-mode" => {
                    parsed.decode_mode = parse_decode_mode(&next_value(&mut args, &arg)?)?
                }
                "--jsonl" => parsed.output_jsonl = true,
                "--help" | "-h" => return Err(usage()),
                _ => return Err(format!("unknown argument: {arg}\n{}", usage())),
            }
        }

        if parsed.requests == 0 {
            return Err("--requests must be greater than 0".to_string());
        }
        if parsed.concurrency == 0 {
            return Err("--concurrency must be greater than 0".to_string());
        }
        if parsed.context_window_length == Some(0) {
            return Err("--context-window-length must be greater than 0".to_string());
        }

        Ok(parsed)
    }
}

fn build_pipeline(
    paths: &ResolvedModelPaths,
    args: &Args,
) -> Result<PrivacyFilterPipeline, String> {
    PrivacyFilterPipeline::from_paths(
        paths,
        TokenizerOptions {
            context_window_length: args.context_window_length,
        },
        OnnxSessionOptions {
            execution_provider_preference: args.provider,
            require_requested_provider: !matches!(
                args.provider,
                OnnxExecutionProviderPreference::Auto
            ),
            intra_threads: args.intra_threads,
            inter_threads: args.inter_threads,
        },
        PrivacyFilterPipelineOptions {
            decode_mode: args.decode_mode,
            output_mode: OutputMode::Typed,
            ..PrivacyFilterPipelineOptions::default()
        },
    )
    .map_err(format_pipeline_error)
}

fn load_texts(args: &Args) -> Result<Vec<String>, String> {
    let Some(input_path) = &args.input_path else {
        return Ok(default_texts());
    };

    let file =
        File::open(input_path).map_err(|err| format!("read {}: {err}", input_path.display()))?;
    let reader = BufReader::new(file);
    let mut texts = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line.map_err(|err| format!("read {}: {err}", input_path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: InputRecord = serde_json::from_str(&line)
            .map_err(|err| format!("invalid JSONL line {}: {err}", line_index + 1))?;
        texts.push(record.text);
    }
    if texts.is_empty() {
        return Err(format!(
            "{} contains no input records",
            input_path.display()
        ));
    }
    Ok(texts)
}

fn build_jobs(texts: &[String], requests: usize) -> VecDeque<(usize, String)> {
    (0..requests)
        .map(|index| (index, texts[index % texts.len()].clone()))
        .collect()
}

async fn run_warmup(
    pipelines: &[Mutex<PrivacyFilterPipeline>],
    texts: &[String],
    warmup: usize,
) -> Result<(), String> {
    if warmup == 0 {
        return Ok(());
    }

    for (pipeline_index, pipeline) in pipelines.iter().enumerate() {
        let mut pipeline = pipeline.lock().await;
        for iteration in 0..warmup {
            let text = &texts[iteration % texts.len()];
            pipeline.redact(text).map_err(|err| {
                format!("warmup failed for pipeline {pipeline_index}, iteration {iteration}: {err}")
            })?;
        }
    }

    Ok(())
}

fn record_from_metrics(
    request_index: usize,
    worker_index: usize,
    latency_ms: f64,
    metrics: &PrivacyFilterPipelineMetrics,
) -> PerfRecord {
    PerfRecord {
        request_index,
        worker_index,
        success: true,
        error: None,
        text_bytes: metrics.text_bytes,
        token_count: metrics.token_count,
        window_count: metrics.window_count,
        detected_span_count: metrics.detected_span_count,
        latency_ms,
        tokenize_ms: metrics.tokenize_ms,
        window_ms: metrics.window_ms,
        onnx_ms: metrics.onnx_ms,
        scoring_ms: metrics.scoring_ms,
        decode_ms: metrics.decode_ms,
        span_ms: metrics.span_ms,
        redaction_ms: metrics.redaction_ms,
        total_ms: metrics.total_ms,
    }
}

fn summarize(
    args: &Args,
    records: &[PerfRecord],
    elapsed_ms: f64,
    selected_provider: OnnxExecutionProvider,
) -> PerfSummary {
    let successes = records.iter().filter(|record| record.success).count();
    let failures = records.len().saturating_sub(successes);
    let total_text_bytes = records.iter().map(|record| record.text_bytes).sum();
    let total_tokens = records.iter().map(|record| record.token_count).sum();
    let total_windows = records.iter().map(|record| record.window_count).sum();
    let elapsed_seconds = elapsed_ms / 1000.0;

    PerfSummary {
        variant: variant_name(args.variant),
        requested_provider: provider_name(args.provider),
        selected_provider: selected_provider_name(selected_provider),
        requests: records.len(),
        warmup_per_worker: args.warmup,
        concurrency: args.concurrency,
        successes,
        failures,
        elapsed_ms,
        throughput_rps: safe_rate(records.len(), elapsed_seconds),
        latency_ms: summarize_values(records.iter().map(|record| record.latency_ms).collect()),
        onnx_ms: summarize_values(records.iter().map(|record| record.onnx_ms).collect()),
        tokenize_ms: summarize_values(records.iter().map(|record| record.tokenize_ms).collect()),
        decode_ms: summarize_values(records.iter().map(|record| record.decode_ms).collect()),
        total_text_bytes,
        total_tokens,
        total_windows,
        avg_text_bytes_per_request: safe_average(total_text_bytes, records.len()),
        avg_tokens_per_request: safe_average(total_tokens, records.len()),
        avg_windows_per_request: safe_average(total_windows, records.len()),
        tokens_per_second: safe_rate(total_tokens, elapsed_seconds),
        windows_per_second: safe_rate(total_windows, elapsed_seconds),
    }
}

fn format_pipeline_error(error: PrivacyFilterPipelineError) -> String {
    match error {
        PrivacyFilterPipelineError::Onnx(OnnxSessionError::NoUsableExecutionProvider {
            attempts,
        }) => format!(
            "no usable ONNX Runtime execution provider found: {}",
            format_probe_attempts(&attempts)
        ),
        PrivacyFilterPipelineError::Onnx(OnnxSessionError::SessionLoadFailed {
            provider,
            path,
            message,
        }) => format!(
            "ONNX session load failed for provider {provider:?} at {}: {message}",
            path.display()
        ),
        other => other.to_string(),
    }
}

fn format_probe_attempts(attempts: &[ExecutionProviderProbeAttempt]) -> String {
    if attempts.is_empty() {
        return "no provider attempts recorded".to_string();
    }
    attempts
        .iter()
        .map(|attempt| {
            let message = attempt.message.as_deref().unwrap_or("no message");
            format!("{:?}={:?} ({message})", attempt.provider, attempt.status)
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn summarize_values(mut values: Vec<f64>) -> LatencySummary {
    if values.is_empty() {
        return LatencySummary {
            min: 0.0,
            p50: 0.0,
            p95: 0.0,
            p99: 0.0,
            max: 0.0,
            mean: 0.0,
        };
    }

    values.sort_by(f64::total_cmp);
    let sum = values.iter().sum::<f64>();
    LatencySummary {
        min: values[0],
        p50: percentile(&values, 0.50),
        p95: percentile(&values, 0.95),
        p99: percentile(&values, 0.99),
        max: values[values.len() - 1],
        mean: sum / values.len() as f64,
    }
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[index]
}

fn parse_provider(value: &str) -> Result<OnnxExecutionProviderPreference, String> {
    match value {
        "auto" => Ok(OnnxExecutionProviderPreference::Auto),
        "cpu" => Ok(OnnxExecutionProviderPreference::Cpu),
        "cuda" => Ok(OnnxExecutionProviderPreference::CudaThenCpu),
        "coreml" => Ok(OnnxExecutionProviderPreference::CoreMlThenCpu),
        _ => Err(format!("unsupported provider: {value}")),
    }
}

fn parse_variant(value: &str) -> Result<PrivacyFilterOnnxVariant, String> {
    match value {
        "q4" => Ok(PrivacyFilterOnnxVariant::Q4),
        "q4f16" => Ok(PrivacyFilterOnnxVariant::Q4F16),
        "quantized" => Ok(PrivacyFilterOnnxVariant::Quantized),
        "fp16" => Ok(PrivacyFilterOnnxVariant::Fp16),
        "full" => Ok(PrivacyFilterOnnxVariant::Full),
        _ => Err(format!("unsupported variant: {value}")),
    }
}

fn variant_name(variant: PrivacyFilterOnnxVariant) -> &'static str {
    match variant {
        PrivacyFilterOnnxVariant::Full => "full",
        PrivacyFilterOnnxVariant::Fp16 => "fp16",
        PrivacyFilterOnnxVariant::Quantized => "quantized",
        PrivacyFilterOnnxVariant::Q4 => "q4",
        PrivacyFilterOnnxVariant::Q4F16 => "q4f16",
    }
}

fn parse_decode_mode(value: &str) -> Result<DecodeMode, String> {
    match value {
        "viterbi" => Ok(DecodeMode::Viterbi),
        "argmax" => Ok(DecodeMode::Argmax),
        _ => Err(format!("unsupported decode mode: {value}")),
    }
}

fn provider_name(provider: OnnxExecutionProviderPreference) -> &'static str {
    match provider {
        OnnxExecutionProviderPreference::Auto => "auto",
        OnnxExecutionProviderPreference::Cpu => "cpu",
        OnnxExecutionProviderPreference::CudaThenCpu => "cuda",
        OnnxExecutionProviderPreference::CoreMlThenCpu => "coreml",
        OnnxExecutionProviderPreference::CudaCoreMlThenCpu => "cuda-coreml",
    }
}

fn selected_provider_name(provider: OnnxExecutionProvider) -> &'static str {
    match provider {
        OnnxExecutionProvider::Cpu => "cpu",
        OnnxExecutionProvider::Cuda => "cuda",
        OnnxExecutionProvider::CoreMl => "coreml",
    }
}

fn parse_usize(value: &str, name: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|err| format!("invalid {name}: {err}"))
}

fn next_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {name}"))
}

fn safe_rate(count: usize, seconds: f64) -> f64 {
    if seconds <= 0.0 {
        0.0
    } else {
        count as f64 / seconds
    }
}

fn safe_average(total: usize, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total as f64 / count as f64
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn default_texts() -> Vec<String> {
    vec![
        r#"Customer support transcript

User: I cannot access the billing portal. The account is under Maria Chen, email maria.chen@northwind.example, phone +1-206-555-0198. The last four digits on the card are 4242 and the invoice I need is INV-2026-0417.

Assistant: I found the tenant northwind-prod and the most recent failed checkout. The error says payment_method_requires_action. Please ask the customer to re-authenticate the payment method. Do not paste the full card number into Slack. If you need to escalate, use ticket SUP-88421 and reference customer id cus_9a7f2b1c.

Internal note: The user also pasted a temporary recovery code, RC-8JDK-22PL-91QZ, in the previous message. Redact it before forwarding the transcript to the vendor."#,
        r#"Incident summary draft

At 03:17 UTC the job scheduler started returning 502 for POST /v1/runs. The impacted workspace was acme-analytics, primary contact devops@acme.example. Logs show repeated attempts from 192.168.44.12 and 10.42.8.19. The on-call engineer used SSH bastion prod-bastion-03 with username elena.ops.

Representative stack trace:
RuntimeError: failed to refresh credentials for service account svc-runner-prod
caused by: token exchange rejected for client_id 0oa7exampleClientId
request_id=req_01HVQ82AM9R4G4FX9K9RCZ8K55

The mitigation was to rotate the scheduler secret and restart workers in us-east-1. The new secret name in Vault is secret/data/prod/scheduler/api-token. Do not include the raw token in the customer-facing postmortem."#,
        r#"Procurement and contract message

Please review the attached order for Contoso Robotics Ltd. The purchasing contact is Priya Raman at priya.raman@contoso.example, office +44 20 7946 0182. Billing address: 12 High Street, London SW1A 1AA, United Kingdom. Shipping address: Dock 4, Unit 17, Milton Keynes MK9 1BB.

The model generated the following summary for legal review: the supplier will provide 120 edge gateways, 36 months of monitoring, and an SLA credit schedule capped at 15 percent of monthly fees. The draft contains bank routing details and a tax identifier that should be removed before we send it to the implementation partner. Reference agreement CTR-2026-1199 and purchase order PO-77820."#,
        r#"Developer message with configuration

We need to deploy the staging callback service. The assistant proposed this env block:

DATABASE_URL=postgres://staging_app:replace-me@db-staging.internal:5432/app
OPENAI_API_KEY=sk-proj-example-redacted-but-shaped-like-a-real-key
SLACK_WEBHOOK_URL=REDACTED_SLACK_WEBHOOK_URL
JWT_ISSUER=https://auth.example.com/realms/staging
SENTRY_DSN=https://public@example.ingest.sentry.io/123456

The code review comment says the assistant should not echo these values into a public ticket. Please produce a sanitized deployment checklist that keeps variable names, target environments, and rollout order, but removes credentials, webhooks, and internal hostnames."#,
        r#"Healthcare scheduling conversation

Patient message: My name is Daniel Brooks, date of birth 1981-07-14. I need to reschedule my cardiology appointment with Dr. Patel. I can be reached at daniel.brooks@example.net or 503-555-0133. My member id is HN-4429107. The assistant summarized my symptoms as chest tightness after exercise, elevated resting pulse, and dizziness when standing.

Clinic note: The generated response should mention available appointment windows and general preparation instructions, but it must not expose the member id, full date of birth, email, phone number, or detailed symptom history in logs sent to analytics."#,
        r#"Long agent trace

The user asked the agent to reconcile a failed refund. The agent looked up order ORD-33019, customer Liam O'Connor, email liam.oconnor@retail.example, shipping phone +353 1 555 0100. The order contained a laptop, dock, and extended warranty. The refund processor returned error payout_account_closed for merchant account acct_1PQExample.

Tool output included:
{"refund_id":"rf_2026_04_18_9081","amount":129900,"currency":"eur","customer_ip":"203.0.113.45","support_pin":"739204","notes":"customer provided IBAN IE29AIBK93115212345678"}

The model output should explain that the refund needs manual review, list safe next steps, and redact personal financial details, support pins, email addresses, phone numbers, and IP addresses. It may keep high-level order status and non-sensitive product categories."#,
        r#"Enterprise admin request

An administrator from Globex wants a usage report for the last quarter. Their message includes admin user id usr_01J0KQZ7AGP4H8Z, email admin.globex@example.com, and organization slug globex-production. They also pasted a bearer token by mistake: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.example.signature.

The assistant's draft response describes how to export audit logs, filter events by workspace, and schedule a recurring report. It should preserve the procedural instructions but mask the token, user id, email, and any exact internal endpoint such as https://api.internal.example.com/v1/admin/audit."#,
        r#"Mixed chat history

System: You are a careful assistant that helps summarize customer messages.
Developer: Never reveal secrets, payment data, authentication tokens, addresses, phone numbers, or emails.
User: Build a concise summary of this conversation for handoff. Sarah Nguyen at sarah.nguyen@example.org reported that her address changed from 88 Market Street, San Francisco, CA 94105 to 210 Pine Avenue, Seattle, WA 98101. Her phone is +1-415-555-0177. She also mentioned passport number X12345678 while verifying identity.

Assistant draft: Sarah changed her mailing address and needs the subscription invoice reissued. The safe handoff should say that identity verification was completed and address update is pending, but it should not include the raw address, phone, email, or passport number."#,
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn usage() -> String {
    "usage: privacy_filter_perf [--cache-dir DIR] [--input FILE.jsonl] [--variant q4|q4f16|quantized|fp16|full] [--requests N] [--warmup N] [--concurrency N] [--provider auto|cpu|coreml|cuda] [--intra-threads N] [--inter-threads N] [--context-window-length N] [--decode-mode viterbi|argmax] [--jsonl]".to_string()
}
