//! Data types for tensor elements

use std::fmt;

/// Supported data types for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point (IEEE 754)
    F16,
    /// Brain floating point (16-bit)
    BF16,
}

impl DType {
    /// Size of the dtype in bytes
    #[must_use]
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }

    /// Convert from safetensors dtype string
    #[must_use]
    pub fn from_safetensors(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::BF16),
            _ => None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::BF16 => write!(f, "bf16"),
        }
    }
}

/// Trait for types that can be used as tensor elements
pub trait TensorDType: Copy + Clone + Default + Send + Sync + 'static {
    /// The corresponding `DType` enum value
    const DTYPE: DType;
}

impl TensorDType for f32 {
    const DTYPE: DType = DType::F32;
}

impl TensorDType for half::f16 {
    const DTYPE: DType = DType::F16;
}

impl TensorDType for half::bf16 {
    const DTYPE: DType = DType::BF16;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size_in_bytes() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F16.size_in_bytes(), 2);
        assert_eq!(DType::BF16.size_in_bytes(), 2);
    }

    #[test]
    fn test_dtype_from_safetensors() {
        assert_eq!(DType::from_safetensors("F32"), Some(DType::F32));
        assert_eq!(DType::from_safetensors("F16"), Some(DType::F16));
        assert_eq!(DType::from_safetensors("BF16"), Some(DType::BF16));
        assert_eq!(DType::from_safetensors("I32"), None);
        assert_eq!(DType::from_safetensors("invalid"), None);
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(format!("{}", DType::F32), "f32");
        assert_eq!(format!("{}", DType::F16), "f16");
        assert_eq!(format!("{}", DType::BF16), "bf16");
    }

    #[test]
    fn test_tensor_dtype_trait() {
        assert_eq!(f32::DTYPE, DType::F32);
        assert_eq!(half::f16::DTYPE, DType::F16);
        assert_eq!(half::bf16::DTYPE, DType::BF16);
    }
}
