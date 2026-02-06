//! Llama model implementation

mod config;
#[cfg(feature = "cuda")]
mod model;

pub use config::LlamaConfig;
#[cfg(feature = "cuda")]
pub use model::LlamaModel;
