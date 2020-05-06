//! Implementations of the Count-Min sketch data structure for summarizing
//! data streams.
//!
//! Count-Min is a *sublinear space* data structure for approximating item
//! frequencies. Originally, it was proposed by G. Cormode et al. in
//! *An Improved Data Stream Summary: The Count-Min Sketch and its
//! Applications.*.
//!
//! Current implementations:
//!
//! * [`CountMin`]

use std::fmt;

mod countmin;

pub use countmin::CountMin;

#[derive(Debug, PartialEq)]
pub enum CountMinError {
    CounterOverflow,
    InvalidDimensions,
    IncompatibleHashers,
    IncompatibleBuilders,
    IncompatibleDimensions,
}

impl fmt::Display for CountMinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CountMinError::CounterOverflow => "counter overflow.".fmt(f),
            CountMinError::InvalidDimensions => "invalid dimensions.".fmt(f),
            CountMinError::IncompatibleHashers => {
                "incompatible hashers.".fmt(f)
            },
            CountMinError::IncompatibleBuilders => {
                "incompatible builders.".fmt(f)
            },
            CountMinError::IncompatibleDimensions => {
                "incompatible dimensions.".fmt(f)
            },
        }
    }
}
