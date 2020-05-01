use std::fmt;

mod countmin;
mod numeric;

pub use countmin::CountMin;

#[derive(Debug, PartialEq)]
pub enum CountMinError {
    CounterOverflow,
    InvalidDimensions,
}

impl fmt::Display for CountMinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CountMinError::CounterOverflow => "counter overflowed.".fmt(f),
            CountMinError::InvalidDimensions => "invalid dimensions.".fmt(f),
        }
    }
}
