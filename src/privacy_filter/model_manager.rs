use std::path::PathBuf;

use hf_hub::Cache;
use hf_hub::api::tokio::{ApiBuilder, ApiError, Progress};
use thiserror::Error;

use super::endpoint_probe::{EndpointProbe, ResolvedEndpoint};
use super::model_files::{
    DownloadGroup, MODEL_REPOSITORY, ModelFile, PrivacyFilterOnnxVariant, base_files, variant_files,
};

#[derive(Debug, Clone)]
pub struct PrivacyFilterModelManager {
    cache_dir: PathBuf,
    endpoint_probe: EndpointProbe,
}

#[derive(Debug, Clone)]
pub struct ModelStatus {
    pub cache_dir: PathBuf,
    pub base: FileGroupStatus,
    pub selected_variant: VariantStatus,
    pub complete: bool,
}

#[derive(Debug, Clone)]
pub struct FileGroupStatus {
    pub group: DownloadGroup,
    pub complete: bool,
    pub files: Vec<ModelFileStatus>,
}

#[derive(Debug, Clone)]
pub struct VariantStatus {
    pub variant: PrivacyFilterOnnxVariant,
    pub complete: bool,
    pub files: Vec<ModelFileStatus>,
}

#[derive(Debug, Clone)]
pub struct ModelFileStatus {
    pub relative_path: String,
    pub local_path: Option<PathBuf>,
    pub exists: bool,
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct ResolvedModelPaths {
    pub endpoint: ResolvedEndpoint,
    pub cache_dir: PathBuf,
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub tokenizer_config_path: PathBuf,
    pub viterbi_calibration_path: PathBuf,
    pub model_path: PathBuf,
    pub model_data_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub group: DownloadGroup,
    pub file: String,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub file_percent: Option<f64>,
}

#[derive(Error, Debug)]
pub enum ModelManagerError {
    #[error(
        "no model download endpoint is available: official={official_error}; mirror={mirror_error}"
    )]
    NoEndpointAvailable {
        official_error: String,
        mirror_error: String,
    },

    #[error("hf-hub operation failed: {source}")]
    HfHub { source: ApiError },

    #[error("file metadata failed for {path}: {source}")]
    Metadata {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("downloaded model file was not resolved: {relative_path}")]
    MissingResolvedPath { relative_path: String },
}

impl Default for PrivacyFilterModelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PrivacyFilterModelManager {
    pub fn new() -> Self {
        Self::with_cache_dir(PathBuf::from(".model"))
    }

    pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
            endpoint_probe: EndpointProbe::new(),
        }
    }

    #[cfg(test)]
    pub fn with_cache_dir_and_probe(
        cache_dir: impl Into<PathBuf>,
        endpoint_probe: EndpointProbe,
    ) -> Self {
        Self {
            cache_dir: cache_dir.into(),
            endpoint_probe,
        }
    }

    pub async fn resolve_endpoint(&self) -> Result<ResolvedEndpoint, ModelManagerError> {
        self.endpoint_probe.resolve().await
    }

    pub fn base_files(&self) -> Vec<ModelFile> {
        base_files()
    }

    pub fn variant_files(&self, variant: PrivacyFilterOnnxVariant) -> Vec<ModelFile> {
        variant_files(variant)
    }

    pub async fn base_status(&self) -> Result<FileGroupStatus, ModelManagerError> {
        let files = self.file_statuses(self.base_files()).await?;
        Ok(FileGroupStatus {
            group: DownloadGroup::Base,
            complete: files.iter().all(|file| file.exists),
            files,
        })
    }

    pub async fn variant_status(
        &self,
        variant: PrivacyFilterOnnxVariant,
    ) -> Result<VariantStatus, ModelManagerError> {
        let files = self.file_statuses(self.variant_files(variant)).await?;
        Ok(VariantStatus {
            variant,
            complete: files.iter().all(|file| file.exists),
            files,
        })
    }

    pub async fn status(
        &self,
        variant: PrivacyFilterOnnxVariant,
    ) -> Result<ModelStatus, ModelManagerError> {
        let base = self.base_status().await?;
        let selected_variant = self.variant_status(variant).await?;
        let complete = base.complete && selected_variant.complete;
        Ok(ModelStatus {
            cache_dir: self.cache_dir.clone(),
            base,
            selected_variant,
            complete,
        })
    }

    pub async fn ensure_downloaded<F>(
        &self,
        variant: PrivacyFilterOnnxVariant,
        progress: F,
    ) -> Result<ResolvedModelPaths, ModelManagerError>
    where
        F: Fn(DownloadProgress) + Clone + Send + Sync + 'static,
    {
        let endpoint = self.resolve_endpoint().await?;
        let repo = ApiBuilder::new()
            .with_cache_dir(self.cache_dir.clone())
            .with_endpoint(endpoint.base_url.clone())
            .with_progress(false)
            .build()
            .map_err(|source| ModelManagerError::HfHub { source })?
            .model(MODEL_REPOSITORY.to_string());

        for file in self.base_files() {
            let progress_adapter = ProgressAdapter::new(
                DownloadGroup::Base,
                file.relative_path.clone(),
                progress.clone(),
            );
            repo.download_with_progress(&file.relative_path, progress_adapter)
                .await
                .map_err(|source| ModelManagerError::HfHub { source })?;
        }

        for file in self.variant_files(variant) {
            let progress_adapter = ProgressAdapter::new(
                DownloadGroup::Variant(variant),
                file.relative_path.clone(),
                progress.clone(),
            );
            repo.download_with_progress(&file.relative_path, progress_adapter)
                .await
                .map_err(|source| ModelManagerError::HfHub { source })?;
        }

        self.resolved_paths(endpoint, variant).await
    }

    async fn file_statuses(
        &self,
        files: Vec<ModelFile>,
    ) -> Result<Vec<ModelFileStatus>, ModelManagerError> {
        let cache_repo = Cache::new(self.cache_dir.clone()).model(MODEL_REPOSITORY.to_string());
        let mut statuses = Vec::with_capacity(files.len());

        for file in files {
            let local_path = cache_repo.get(&file.relative_path);
            let size_bytes = match &local_path {
                Some(path) => {
                    let metadata = tokio::fs::metadata(path).await.map_err(|source| {
                        ModelManagerError::Metadata {
                            path: path.clone(),
                            source,
                        }
                    })?;
                    Some(metadata.len())
                }
                None => None,
            };

            statuses.push(ModelFileStatus {
                relative_path: file.relative_path,
                exists: local_path.is_some(),
                local_path,
                size_bytes,
            });
        }

        Ok(statuses)
    }

    async fn resolved_paths(
        &self,
        endpoint: ResolvedEndpoint,
        variant: PrivacyFilterOnnxVariant,
    ) -> Result<ResolvedModelPaths, ModelManagerError> {
        let base_status = self.base_status().await?;
        let variant_status = self.variant_status(variant).await?;

        Ok(ResolvedModelPaths {
            endpoint,
            cache_dir: self.cache_dir.clone(),
            config_path: required_path(&base_status.files, "config.json")?,
            tokenizer_path: required_path(&base_status.files, "tokenizer.json")?,
            tokenizer_config_path: required_path(&base_status.files, "tokenizer_config.json")?,
            viterbi_calibration_path: required_path(
                &base_status.files,
                "viterbi_calibration.json",
            )?,
            model_path: required_model_path(&variant_status.files)?,
            model_data_paths: variant_status
                .files
                .iter()
                .filter(|file| !file.relative_path.ends_with(".onnx"))
                .map(|file| {
                    file.local_path
                        .clone()
                        .ok_or_else(|| ModelManagerError::MissingResolvedPath {
                            relative_path: file.relative_path.clone(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

fn required_path(
    files: &[ModelFileStatus],
    relative_path: &str,
) -> Result<PathBuf, ModelManagerError> {
    files
        .iter()
        .find(|file| file.relative_path == relative_path)
        .and_then(|file| file.local_path.clone())
        .ok_or_else(|| ModelManagerError::MissingResolvedPath {
            relative_path: relative_path.to_string(),
        })
}

fn required_model_path(files: &[ModelFileStatus]) -> Result<PathBuf, ModelManagerError> {
    let model_file = files
        .iter()
        .find(|file| file.relative_path.ends_with(".onnx"))
        .ok_or_else(|| ModelManagerError::MissingResolvedPath {
            relative_path: "*.onnx".to_string(),
        })?;

    model_file
        .local_path
        .clone()
        .ok_or_else(|| ModelManagerError::MissingResolvedPath {
            relative_path: model_file.relative_path.clone(),
        })
}

#[derive(Clone)]
struct ProgressAdapter<F>
where
    F: Fn(DownloadProgress) + Clone + Send + Sync + 'static,
{
    group: DownloadGroup,
    file: String,
    downloaded_bytes: u64,
    total_bytes: Option<u64>,
    callback: F,
}

impl<F> ProgressAdapter<F>
where
    F: Fn(DownloadProgress) + Clone + Send + Sync + 'static,
{
    fn new(group: DownloadGroup, file: String, callback: F) -> Self {
        Self {
            group,
            file,
            downloaded_bytes: 0,
            total_bytes: None,
            callback,
        }
    }

    fn emit(&self) {
        let file_percent = match self.total_bytes {
            Some(total_bytes) if total_bytes > 0 => {
                Some((self.downloaded_bytes as f64 / total_bytes as f64) * 100.0)
            }
            _ => None,
        };

        (self.callback)(DownloadProgress {
            group: self.group.clone(),
            file: self.file.clone(),
            downloaded_bytes: self.downloaded_bytes,
            total_bytes: self.total_bytes,
            file_percent,
        });
    }
}

impl<F> Progress for ProgressAdapter<F>
where
    F: Fn(DownloadProgress) + Clone + Send + Sync + 'static,
{
    async fn init(&mut self, size: usize, filename: &str) {
        self.file = filename.to_string();
        self.downloaded_bytes = 0;
        self.total_bytes = Some(size as u64);
        self.emit();
    }

    async fn update(&mut self, size: usize) {
        self.downloaded_bytes = self.downloaded_bytes.saturating_add(size as u64);
        self.emit();
    }

    async fn finish(&mut self) {
        if let Some(total_bytes) = self.total_bytes {
            self.downloaded_bytes = total_bytes;
        }
        self.emit();
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temp_cache_dir(test_name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        std::env::temp_dir().join(format!("rapi-{test_name}-{suffix}"))
    }

    fn write_cached_file(cache_dir: &std::path::Path, relative_path: &str, content: &[u8]) {
        let ref_path = cache_dir
            .join("models--openai--privacy-filter")
            .join("refs")
            .join("main");
        fs::create_dir_all(ref_path.parent().expect("cache ref parent")).expect("create refs dir");
        fs::write(ref_path, b"test-commit").expect("write cache ref");

        let path = cache_dir
            .join("models--openai--privacy-filter")
            .join("snapshots")
            .join("test-commit")
            .join(relative_path);
        fs::create_dir_all(path.parent().expect("cached file parent")).expect("create cache dir");
        fs::write(path, content).expect("write cache file");
    }

    fn write_complete_cache(cache_dir: &std::path::Path, variant: PrivacyFilterOnnxVariant) {
        for file in base_files() {
            write_cached_file(cache_dir, &file.relative_path, b"base");
        }

        for file in variant_files(variant) {
            write_cached_file(cache_dir, &file.relative_path, b"variant");
        }
    }

    #[tokio::test]
    async fn status_is_incomplete_without_cached_files() {
        let cache_dir = temp_cache_dir("empty-status");
        let manager = PrivacyFilterModelManager::with_cache_dir(&cache_dir);

        let status = manager
            .status(PrivacyFilterOnnxVariant::Quantized)
            .await
            .expect("status");

        assert!(!status.base.complete);
        assert!(!status.selected_variant.complete);
        assert!(!status.complete);

        let _ = fs::remove_dir_all(cache_dir);
    }

    #[tokio::test]
    async fn status_is_complete_when_base_and_variant_are_cached() {
        let cache_dir = temp_cache_dir("complete-status");
        write_complete_cache(&cache_dir, PrivacyFilterOnnxVariant::Quantized);
        let manager = PrivacyFilterModelManager::with_cache_dir(&cache_dir);

        let status = manager
            .status(PrivacyFilterOnnxVariant::Quantized)
            .await
            .expect("status");

        assert!(status.base.complete);
        assert!(status.selected_variant.complete);
        assert!(status.complete);
        assert!(
            status
                .base
                .files
                .iter()
                .all(|file| file.size_bytes == Some(4))
        );

        let _ = fs::remove_dir_all(cache_dir);
    }

    #[tokio::test]
    async fn status_requires_selected_variant() {
        let cache_dir = temp_cache_dir("selected-variant-status");
        write_complete_cache(&cache_dir, PrivacyFilterOnnxVariant::Q4);
        let manager = PrivacyFilterModelManager::with_cache_dir(&cache_dir);

        let status = manager
            .status(PrivacyFilterOnnxVariant::Quantized)
            .await
            .expect("status");

        assert!(status.base.complete);
        assert!(!status.selected_variant.complete);
        assert!(!status.complete);

        let _ = fs::remove_dir_all(cache_dir);
    }

    #[tokio::test]
    async fn progress_adapter_reports_file_percent() {
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let callback_events = events.clone();
        let callback = move |event: DownloadProgress| {
            callback_events.lock().expect("events lock").push(event);
        };
        let mut progress =
            ProgressAdapter::new(DownloadGroup::Base, "config.json".to_string(), callback);

        progress.init(100, "config.json").await;
        progress.update(25).await;
        progress.finish().await;

        drop(progress);

        let events = events.lock().expect("events lock");
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].file_percent, Some(0.0));
        assert_eq!(events[1].file_percent, Some(25.0));
        assert_eq!(events[2].file_percent, Some(100.0));
    }
}
