use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use serde::Deserialize;
use thiserror::Error;

use super::model_manager::ResolvedModelPaths;

const BACKGROUND_CLASS_LABEL: &str = "O";
const DEFAULT_CATEGORY_VERSION: &str = "v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryTag {
    Begin,
    Inside,
    End,
    Single,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelInfo {
    pub category_version: String,
    pub token_class_names: Vec<String>,
    pub span_class_names: Vec<String>,
    pub token_to_span_label: Vec<usize>,
    pub token_boundary_tags: Vec<Option<BoundaryTag>>,
    pub background_token_label: usize,
    pub background_span_label: usize,
}

#[derive(Error, Debug)]
pub enum LabelSpaceError {
    #[error("label config read failed for {path}: {source}")]
    ConfigRead {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("label config parse failed for {path}: {source}")]
    ConfigParse {
        path: PathBuf,
        source: serde_json::Error,
    },

    #[error("unsupported privacy-filter category_version: {0}")]
    UnsupportedCategoryVersion(String),

    #[error("label config is invalid: {0}")]
    InvalidConfig(String),
}

#[derive(Debug, Deserialize)]
struct LabelConfigFile {
    category_version: Option<String>,
    num_labels: Option<usize>,
    span_class_names: Option<Vec<String>>,
    ner_class_names: Option<Vec<String>>,
    id2label: Option<HashMap<String, String>>,
    label2id: Option<HashMap<String, usize>>,
}

impl LabelInfo {
    pub fn from_paths(paths: &ResolvedModelPaths) -> Result<Self, LabelSpaceError> {
        Self::from_config_path(&paths.config_path)
    }

    pub fn from_config_path(path: &PathBuf) -> Result<Self, LabelSpaceError> {
        let content =
            std::fs::read_to_string(path).map_err(|source| LabelSpaceError::ConfigRead {
                path: path.clone(),
                source,
            })?;
        let config: LabelConfigFile =
            serde_json::from_str(&content).map_err(|source| LabelSpaceError::ConfigParse {
                path: path.clone(),
                source,
            })?;
        build_label_info(config)
    }

    pub fn label_count(&self) -> usize {
        self.token_class_names.len()
    }
}

fn build_label_info(config: LabelConfigFile) -> Result<LabelInfo, LabelSpaceError> {
    let category_version = normalized_category_version(config.category_version.as_deref());
    let token_class_names = if let Some(ner_class_names) = config.ner_class_names.clone() {
        normalize_string_list(ner_class_names, "ner_class_names")?
    } else if let Some(span_class_names) = config.span_class_names.clone() {
        expand_span_class_names(&normalize_span_class_names(span_class_names)?)
    } else if let Some(id2label) = config.id2label.clone() {
        token_class_names_from_id2label(id2label)?
    } else {
        let inferred_version = if let Some(num_labels) = config.num_labels {
            category_version_for_label_count(num_labels).map(str::to_string)
        } else {
            None
        };
        let version = category_version
            .clone()
            .or(inferred_version)
            .unwrap_or_else(|| DEFAULT_CATEGORY_VERSION.to_string());
        expand_span_class_names(builtin_span_class_names(&version)?)
    };

    if let Some(expected) = config.num_labels {
        if expected != token_class_names.len() {
            return Err(LabelSpaceError::InvalidConfig(format!(
                "num_labels={expected} does not match token label count={}",
                token_class_names.len()
            )));
        }
    }

    if let Some(label2id) = config.label2id {
        validate_label2id(&token_class_names, &label2id)?;
    }

    let category_version = category_version
        .or_else(|| category_version_for_label_count(token_class_names.len()).map(str::to_string))
        .unwrap_or_else(|| "custom".to_string());
    build_label_info_from_token_class_names(category_version, token_class_names)
}

fn normalized_category_version(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_ascii_lowercase())
}

fn normalize_string_list(
    values: Vec<String>,
    field_name: &str,
) -> Result<Vec<String>, LabelSpaceError> {
    if values.is_empty() {
        return Err(LabelSpaceError::InvalidConfig(format!(
            "{field_name} must not be empty"
        )));
    }

    let mut normalized = Vec::with_capacity(values.len());
    for (index, value) in values.into_iter().enumerate() {
        let value = value.trim().to_string();
        if value.is_empty() {
            return Err(LabelSpaceError::InvalidConfig(format!(
                "{field_name}[{index}] must not be empty"
            )));
        }
        normalized.push(value);
    }
    ensure_unique(&normalized, field_name)?;
    Ok(normalized)
}

fn normalize_span_class_names(values: Vec<String>) -> Result<Vec<String>, LabelSpaceError> {
    let values = normalize_string_list(values, "span_class_names")?;
    if !values.iter().any(|value| value == BACKGROUND_CLASS_LABEL) {
        return Err(LabelSpaceError::InvalidConfig(format!(
            "span_class_names must include background label {BACKGROUND_CLASS_LABEL:?}"
        )));
    }

    let mut normalized = vec![BACKGROUND_CLASS_LABEL.to_string()];
    normalized.extend(
        values
            .into_iter()
            .filter(|value| value != BACKGROUND_CLASS_LABEL),
    );
    Ok(normalized)
}

fn ensure_unique(values: &[String], field_name: &str) -> Result<(), LabelSpaceError> {
    let mut seen = HashMap::new();
    for value in values {
        if seen.insert(value, ()).is_some() {
            return Err(LabelSpaceError::InvalidConfig(format!(
                "{field_name} contains duplicate label {value:?}"
            )));
        }
    }
    Ok(())
}

fn token_class_names_from_id2label(
    id2label: HashMap<String, String>,
) -> Result<Vec<String>, LabelSpaceError> {
    if id2label.is_empty() {
        return Err(LabelSpaceError::InvalidConfig(
            "id2label must not be empty".to_string(),
        ));
    }

    let mut ordered = BTreeMap::new();
    for (raw_id, label) in id2label {
        let id = raw_id.parse::<usize>().map_err(|_| {
            LabelSpaceError::InvalidConfig(format!("id2label key {raw_id:?} is not a label id"))
        })?;
        ordered.insert(id, label);
    }

    let mut labels = Vec::with_capacity(ordered.len());
    for expected_id in 0..ordered.len() {
        let label = ordered.get(&expected_id).ok_or_else(|| {
            LabelSpaceError::InvalidConfig(format!(
                "id2label is missing contiguous label id {expected_id}"
            ))
        })?;
        labels.push(label.clone());
    }
    normalize_string_list(labels, "id2label")
}

fn validate_label2id(
    token_class_names: &[String],
    label2id: &HashMap<String, usize>,
) -> Result<(), LabelSpaceError> {
    for (expected_id, label) in token_class_names.iter().enumerate() {
        match label2id.get(label) {
            Some(actual_id) if *actual_id == expected_id => {}
            Some(actual_id) => {
                return Err(LabelSpaceError::InvalidConfig(format!(
                    "label2id[{label:?}]={actual_id} does not match expected id {expected_id}"
                )));
            }
            None => {
                return Err(LabelSpaceError::InvalidConfig(format!(
                    "label2id is missing label {label:?}"
                )));
            }
        }
    }
    Ok(())
}

fn build_label_info_from_token_class_names(
    category_version: String,
    token_class_names: Vec<String>,
) -> Result<LabelInfo, LabelSpaceError> {
    ensure_unique(&token_class_names, "token_class_names")?;

    let mut span_class_names = vec![BACKGROUND_CLASS_LABEL.to_string()];
    let mut span_label_lookup = HashMap::from([(BACKGROUND_CLASS_LABEL.to_string(), 0usize)]);
    let mut token_to_span_label = Vec::with_capacity(token_class_names.len());
    let mut token_boundary_tags = Vec::with_capacity(token_class_names.len());
    let mut background_token_label = None;
    let mut boundaries_by_label: HashMap<String, Vec<BoundaryTag>> = HashMap::new();

    for (token_id, name) in token_class_names.iter().enumerate() {
        if name == BACKGROUND_CLASS_LABEL {
            background_token_label = Some(token_id);
            token_to_span_label.push(0);
            token_boundary_tags.push(None);
            continue;
        }

        let (boundary, base_label) = parse_token_label(name)?;
        let span_id = match span_label_lookup.get(base_label) {
            Some(span_id) => *span_id,
            None => {
                let span_id = span_class_names.len();
                span_class_names.push(base_label.to_string());
                span_label_lookup.insert(base_label.to_string(), span_id);
                span_id
            }
        };
        token_to_span_label.push(span_id);
        token_boundary_tags.push(Some(boundary));
        boundaries_by_label
            .entry(base_label.to_string())
            .or_default()
            .push(boundary);
    }

    let background_token_label = background_token_label.ok_or_else(|| {
        LabelSpaceError::InvalidConfig(format!(
            "token class names must include background label {BACKGROUND_CLASS_LABEL:?}"
        ))
    })?;

    for (label, boundaries) in boundaries_by_label {
        for required in [
            BoundaryTag::Begin,
            BoundaryTag::Inside,
            BoundaryTag::End,
            BoundaryTag::Single,
        ] {
            if !boundaries.contains(&required) {
                return Err(LabelSpaceError::InvalidConfig(format!(
                    "token class names missing {required:?} boundary for span label {label:?}"
                )));
            }
        }
    }

    Ok(LabelInfo {
        category_version,
        token_class_names,
        span_class_names,
        token_to_span_label,
        token_boundary_tags,
        background_token_label,
        background_span_label: 0,
    })
}

fn parse_token_label(name: &str) -> Result<(BoundaryTag, &str), LabelSpaceError> {
    let (boundary, base_label) = name.split_once('-').ok_or_else(|| {
        LabelSpaceError::InvalidConfig(format!(
            "token label {name:?} must use '<B|I|E|S>-<label>' format"
        ))
    })?;
    if base_label.is_empty() {
        return Err(LabelSpaceError::InvalidConfig(format!(
            "token label {name:?} has an empty span label"
        )));
    }

    let boundary = match boundary {
        "B" => BoundaryTag::Begin,
        "I" => BoundaryTag::Inside,
        "E" => BoundaryTag::End,
        "S" => BoundaryTag::Single,
        _ => {
            return Err(LabelSpaceError::InvalidConfig(format!(
                "token label {name:?} must use B, I, E, or S boundary"
            )));
        }
    };
    Ok((boundary, base_label))
}

fn expand_span_class_names(span_class_names: &[String]) -> Vec<String> {
    let mut token_class_names = vec![BACKGROUND_CLASS_LABEL.to_string()];
    for span_class_name in span_class_names {
        if span_class_name == BACKGROUND_CLASS_LABEL {
            continue;
        }
        for boundary in ["B", "I", "E", "S"] {
            token_class_names.push(format!("{boundary}-{span_class_name}"));
        }
    }
    token_class_names
}

fn builtin_span_class_names(version: &str) -> Result<&'static [String], LabelSpaceError> {
    match version {
        "v2" => Ok(v2_span_class_names()),
        "v4" => Ok(v4_span_class_names()),
        "v7" => Ok(v7_span_class_names()),
        other => Err(LabelSpaceError::UnsupportedCategoryVersion(
            other.to_string(),
        )),
    }
}

fn category_version_for_label_count(label_count: usize) -> Option<&'static str> {
    ["v2", "v4", "v7"].into_iter().find(|version| {
        builtin_span_class_names(version)
            .ok()
            .map(expand_span_class_names)
            .is_some_and(|labels| labels.len() == label_count)
    })
}

fn v2_span_class_names() -> &'static [String] {
    static VALUE: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    VALUE.get_or_init(|| {
        strings(&[
            "O",
            "account_number",
            "private_address",
            "private_date",
            "private_email",
            "private_person",
            "private_phone",
            "private_url",
            "secret",
        ])
    })
}

fn v4_span_class_names() -> &'static [String] {
    static VALUE: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    VALUE.get_or_init(|| {
        strings(&[
            "O",
            "private_person",
            "other_person",
            "personal_url",
            "other_url",
            "personal_location",
            "other_location",
            "personal_email",
            "other_email",
            "personal_phone",
            "other_phone",
            "personal_date",
            "other_date",
            "personal_id",
            "secret",
        ])
    })
}

fn v7_span_class_names() -> &'static [String] {
    static VALUE: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    VALUE.get_or_init(|| {
        strings(&[
            "O",
            "personal_name",
            "personal_handle",
            "other_person",
            "personal_email",
            "other_email",
            "personal_phone",
            "other_phone",
            "personal_location",
            "other_location",
            "personal_url",
            "other_url",
            "personal_org",
            "personal_gov_id",
            "personal_fin_id",
            "personal_health_id",
            "personal_device_id",
            "personal_vehicle_id",
            "personal_property_id",
            "personal_edu_id",
            "personal_emp_id",
            "personal_membership_id",
            "personal_registry_id",
            "personal_date",
            "secret",
            "secret_url",
        ])
    })
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| value.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_spans(spans: &[&str]) -> LabelConfigFile {
        LabelConfigFile {
            category_version: None,
            num_labels: None,
            span_class_names: Some(strings(spans)),
            ner_class_names: None,
            id2label: None,
            label2id: None,
        }
    }

    #[test]
    fn expands_span_class_names_to_bioes_labels() {
        let info = build_label_info(config_with_spans(&["O", "personal_email"])).unwrap();

        assert_eq!(
            info.token_class_names,
            strings(&[
                "O",
                "B-personal_email",
                "I-personal_email",
                "E-personal_email",
                "S-personal_email"
            ])
        );
        assert_eq!(info.span_class_names, strings(&["O", "personal_email"]));
        assert_eq!(info.background_token_label, 0);
        assert_eq!(info.token_boundary_tags[4], Some(BoundaryTag::Single));
    }

    #[test]
    fn infers_builtin_category_version_from_label_count() {
        let config = LabelConfigFile {
            category_version: None,
            num_labels: Some(33),
            span_class_names: None,
            ner_class_names: None,
            id2label: None,
            label2id: None,
        };

        let info = build_label_info(config).unwrap();

        assert_eq!(info.category_version, "v2");
        assert_eq!(info.token_class_names.len(), 33);
    }

    #[test]
    fn rejects_missing_boundary_label() {
        let config = LabelConfigFile {
            category_version: None,
            num_labels: None,
            span_class_names: None,
            ner_class_names: Some(strings(&["O", "B-email", "I-email", "E-email"])),
            id2label: None,
            label2id: None,
        };

        assert!(build_label_info(config).is_err());
    }

    #[test]
    fn orders_id2label_by_numeric_id() {
        let config = LabelConfigFile {
            category_version: None,
            num_labels: Some(5),
            span_class_names: None,
            ner_class_names: None,
            id2label: Some(HashMap::from([
                ("4".to_string(), "S-email".to_string()),
                ("0".to_string(), "O".to_string()),
                ("1".to_string(), "B-email".to_string()),
                ("2".to_string(), "I-email".to_string()),
                ("3".to_string(), "E-email".to_string()),
            ])),
            label2id: None,
        };

        let info = build_label_info(config).unwrap();

        assert_eq!(info.token_class_names[1], "B-email");
        assert_eq!(info.token_class_names[4], "S-email");
    }
}
