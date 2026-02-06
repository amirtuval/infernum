//! Operations (kernels) for tensor computations

mod attention;
mod matmul;
mod rmsnorm;
mod rope;
mod silu;
mod softmax;

pub use attention::attention;
pub use matmul::matmul;
pub use rmsnorm::rms_norm;
pub use rope::{apply_rope, precompute_rope_cache};
pub use silu::{silu, silu_mul};
pub use softmax::{softmax, softmax_causal};
