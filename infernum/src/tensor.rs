//! Tensor trait definition

use crate::dtype::DType;

/// Core tensor trait that defines the interface for all tensor implementations
///
/// Different backends (CUDA, Metal, CPU) implement this trait to provide
/// hardware-specific tensor operations while maintaining a unified interface.
pub trait Tensor: Sized {
    /// Returns the shape of the tensor as a slice of dimensions
    fn shape(&self) -> &[usize];

    /// Returns the data type of tensor elements
    fn dtype(&self) -> DType;

    /// Returns the total number of elements in the tensor
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Returns the number of dimensions (rank) of the tensor
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Returns the stride for each dimension
    fn strides(&self) -> Vec<usize> {
        let shape = self.shape();
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Returns true if the tensor is contiguous in memory
    fn is_contiguous(&self) -> bool {
        true // Default: assume contiguous
    }

    /// Returns the size of the tensor data in bytes
    fn size_in_bytes(&self) -> usize {
        self.numel() * self.dtype().size_in_bytes()
    }
}
