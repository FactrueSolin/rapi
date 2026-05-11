#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrivacyFilterOnnxVariant {
    Full,
    Fp16,
    Quantized,
    Q4,
    Q4F16,
}

impl Default for PrivacyFilterOnnxVariant {
    fn default() -> Self {
        Self::Quantized
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownloadGroup {
    Base,
    Variant(PrivacyFilterOnnxVariant),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFile {
    pub relative_path: String,
}

impl ModelFile {
    pub fn new(relative_path: impl Into<String>) -> Self {
        Self {
            relative_path: relative_path.into(),
        }
    }

    pub fn is_model_file(&self) -> bool {
        self.relative_path.ends_with(".onnx")
    }
}

pub const MODEL_REPOSITORY: &str = "openai/privacy-filter";
pub const DEFAULT_REVISION: &str = "main";

pub fn base_files() -> Vec<ModelFile> {
    [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "viterbi_calibration.json",
    ]
    .into_iter()
    .map(ModelFile::new)
    .collect()
}

pub fn variant_files(variant: PrivacyFilterOnnxVariant) -> Vec<ModelFile> {
    match variant {
        PrivacyFilterOnnxVariant::Full => [
            "onnx/model.onnx",
            "onnx/model.onnx_data",
            "onnx/model.onnx_data_1",
            "onnx/model.onnx_data_2",
        ]
        .into_iter()
        .map(ModelFile::new)
        .collect(),
        PrivacyFilterOnnxVariant::Fp16 => [
            "onnx/model_fp16.onnx",
            "onnx/model_fp16.onnx_data",
            "onnx/model_fp16.onnx_data_1",
        ]
        .into_iter()
        .map(ModelFile::new)
        .collect(),
        PrivacyFilterOnnxVariant::Quantized => [
            "onnx/model_quantized.onnx",
            "onnx/model_quantized.onnx_data",
        ]
        .into_iter()
        .map(ModelFile::new)
        .collect(),
        PrivacyFilterOnnxVariant::Q4 => ["onnx/model_q4.onnx", "onnx/model_q4.onnx_data"]
            .into_iter()
            .map(ModelFile::new)
            .collect(),
        PrivacyFilterOnnxVariant::Q4F16 => ["onnx/model_q4f16.onnx", "onnx/model_q4f16.onnx_data"]
            .into_iter()
            .map(ModelFile::new)
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(files: Vec<ModelFile>) -> Vec<String> {
        files.into_iter().map(|file| file.relative_path).collect()
    }

    #[test]
    fn base_file_list_is_stable() {
        assert_eq!(
            paths(base_files()),
            vec![
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "viterbi_calibration.json",
            ]
        );
    }

    #[test]
    fn quantized_variant_file_list_is_stable() {
        assert_eq!(
            paths(variant_files(PrivacyFilterOnnxVariant::Quantized)),
            vec![
                "onnx/model_quantized.onnx",
                "onnx/model_quantized.onnx_data",
            ]
        );
    }

    #[test]
    fn every_variant_has_one_onnx_model_file() {
        for variant in [
            PrivacyFilterOnnxVariant::Full,
            PrivacyFilterOnnxVariant::Fp16,
            PrivacyFilterOnnxVariant::Quantized,
            PrivacyFilterOnnxVariant::Q4,
            PrivacyFilterOnnxVariant::Q4F16,
        ] {
            let model_count = variant_files(variant)
                .into_iter()
                .filter(ModelFile::is_model_file)
                .count();
            assert_eq!(model_count, 1, "variant {variant:?}");
        }
    }

    #[test]
    fn default_variant_is_quantized() {
        assert_eq!(
            PrivacyFilterOnnxVariant::default(),
            PrivacyFilterOnnxVariant::Quantized
        );
    }
}
