//! CUDA context management

use cudarc::cublas::CudaBlas;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

use crate::Result;

/// Manages CUDA device and associated resources (cuBLAS handle, etc.)
#[derive(Clone)]
pub struct CudaContext {
    device: Arc<CudaDevice>,
    blas: Arc<CudaBlas>,
}

impl CudaContext {
    /// Create a new CUDA context for the specified device ordinal
    ///
    /// # Errors
    /// Returns an error if CUDA device initialization fails
    pub fn new(ordinal: usize) -> Result<Self> {
        let device = CudaDevice::new(ordinal)?;
        let blas = CudaBlas::new(device.clone())?;
        Ok(Self {
            device,
            blas: Arc::new(blas),
        })
    }

    /// Get a reference to the underlying CUDA device
    #[must_use]
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get a reference to the cuBLAS handle
    #[must_use]
    pub fn blas(&self) -> &Arc<CudaBlas> {
        &self.blas
    }

    /// Synchronize the CUDA device (wait for all operations to complete)
    ///
    /// # Errors
    /// Returns an error if synchronization fails
    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize()?;
        Ok(())
    }
}
