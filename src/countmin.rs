use std::cmp::Ordering;
use std::f64;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;

use rand;

use crate::numeric::Zero;
use crate::CountMinError;

#[derive(Debug)]
pub struct CountMin<H, B, C>
where
    H: Hash + ?Sized,
    B: BuildHasher,
    C: Copy + Zero + PartialOrd,
{
    width:   usize,
    counts:  Vec<C>,
    hashers: Vec<(u64, u64)>,
    builder: B,
    phantom: PhantomData<H>,
}

impl<H, B, C> CountMin<H, B, C>
where
    H: Hash + ?Sized,
    B: BuildHasher,
    C: Copy + Zero + PartialOrd,
{
    // A large 32-bit prime stored in a u64.
    const MOD: u64 = 2147483647;

    pub fn new(
        epsilon: f64,
        delta: f64,
        builder: B,
    ) -> Result<Self, CountMinError> {
        Self::with_dimensions(
            (f64::consts::E / epsilon).ceil() as usize,
            (1.0 / delta).ln().ceil() as usize,
            builder,
        )
    }

    pub fn with_dimensions(
        width: usize,
        depth: usize,
        builder: B,
    ) -> Result<Self, CountMinError> {
        if width as u64 > Self::MOD || width < 1 || depth < 1 {
            return Err(CountMinError::InvalidDimensions);
        }

        Ok(CountMin {
            width:   width,
            counts:  vec![C::zero(); width * depth],
            hashers: Self::build_hashers(depth),
            builder: builder,
            phantom: PhantomData,
        })
    }

    pub fn update(&mut self, item: &H, diff: C) {
        let mut hasher = self.builder.build_hasher();

        item.hash(&mut hasher);

        let x: u64 = hasher.finish() % Self::MOD;

        for (i, (a, b)) in self.hashers.iter().enumerate() {
            // Here a, b and x fit in u32 integers but are stored as u64.
            // This calculation should not overflow.
            let hash = ((((a * x) % Self::MOD) + b) % Self::MOD) as usize;

            let index = i * self.width + hash % self.width;

            self.counts[index] += diff;
        }
    }

    pub fn query(&self, item: &H) -> C {
        let mut hasher = self.builder.build_hasher();

        item.hash(&mut hasher);

        let x: u64 = hasher.finish() % Self::MOD;

        // Here we unwrap since we know that `self.hashers` is not empty.
        // The dimensions were checked in the constructor.

        self.hashers
            .iter()
            .enumerate()
            .map(|(i, (a, b))| {
                // Here a, b and x fit in u32 integers but are stored as u64.
                // This calculation should not overflow.
                let hash = ((((a * x) % Self::MOD) + b) % Self::MOD) as usize;

                let index = i * self.width + hash % self.width;

                self.counts[index]
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap()
    }

    fn build_hashers(count: usize) -> Vec<(u64, u64)> {
        (0..count)
            .map(|_| {
                (
                    rand::random::<u64>() % Self::MOD,
                    rand::random::<u64>() % Self::MOD,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::hash::{BuildHasher, Hasher};

    #[derive(Debug)]
    struct PassThroughHasher(u64);

    impl Hasher for PassThroughHasher {
        #[inline]
        fn finish(&self) -> u64 {
            self.0
        }

        #[inline]
        fn write(&mut self, _: &[u8]) {}

        #[inline]
        fn write_u64(&mut self, i: u64) {
            self.0 = i;
        }
    }

    #[derive(Debug)]
    struct PassThroughHasherBuilder;

    impl BuildHasher for PassThroughHasherBuilder {
        type Hasher = PassThroughHasher;

        fn build_hasher(&self) -> Self::Hasher {
            PassThroughHasher(0)
        }
    }

    #[test]
    fn test_new() {
        let cms: CountMin<u64, PassThroughHasherBuilder, u32> =
            CountMin::new(0.001, 0.01, PassThroughHasherBuilder {}).unwrap();

        assert_eq!(cms.width, 2719);

        assert_eq!(cms.counts.len(), 2719 * 5);
    }

    #[test]
    fn test_with_dimensions() {
        let cms: Result<
            CountMin<u64, PassThroughHasherBuilder, u32>,
            CountMinError,
        > = CountMin::with_dimensions(0, 13, PassThroughHasherBuilder {});

        assert_eq!(cms.err(), Some(CountMinError::InvalidDimensions));

        let cms: Result<
            CountMin<u64, PassThroughHasherBuilder, u32>,
            CountMinError,
        > = CountMin::with_dimensions(22, 0, PassThroughHasherBuilder {});

        assert_eq!(cms.err(), Some(CountMinError::InvalidDimensions));

        let cms: Result<
            CountMin<u64, PassThroughHasherBuilder, u32>,
            CountMinError,
        > = CountMin::with_dimensions(1 << 33, 12, PassThroughHasherBuilder {});

        assert_eq!(cms.err(), Some(CountMinError::InvalidDimensions));
    }

    #[test]
    fn test_update() {
        let builder = PassThroughHasherBuilder {};

        let mut cms: CountMin<u64, PassThroughHasherBuilder, u32> =
            CountMin::with_dimensions(10, 3, builder).unwrap();

        cms.hashers = vec![(1, 0), (2, 0), (3, 0)];

        cms.update(&1, 1);

        cms.update(&12, 2);

        let expected = vec![
            0, 1, 2, 0, 0, 0, 0, 0, 0, 0, // depth: 0.
            0, 0, 1, 0, 2, 0, 0, 0, 0, 0, // depth: 1.
            0, 0, 0, 1, 0, 0, 2, 0, 0, 0, // depth: 2.
        ];

        assert_eq!(cms.counts, expected);

        cms.update(&3, 3);

        cms.update(&22, 3);

        let expected = vec![
            0, 1, 5, 3, 0, 0, 0, 0, 0, 0, // depth: 0.
            0, 0, 1, 0, 5, 0, 3, 0, 0, 0, // depth: 1.
            0, 0, 0, 1, 0, 0, 5, 0, 0, 3, // depth: 2.
        ];

        assert_eq!(cms.counts, expected);

        let builder = PassThroughHasherBuilder {};

        let mut cms: CountMin<u64, PassThroughHasherBuilder, f32> =
            CountMin::with_dimensions(10, 3, builder).unwrap();

        cms.hashers = vec![(1, 0), (2, 0), (3, 0)];

        cms.update(&1, 1.2);

        cms.update(&12, 2.3);

        let expected = vec![
            0.0, 1.2, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // depth: 0.
            0.0, 0.0, 1.2, 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, // depth: 1.
            0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 2.3, 0.0, 0.0, 0.0, // depth: 2.
        ];

        assert_eq!(cms.counts, expected);
    }

    #[test]
    fn test_query() {
        let builder = PassThroughHasherBuilder {};

        let mut cms: CountMin<u64, PassThroughHasherBuilder, u32> =
            CountMin::with_dimensions(10, 3, builder).unwrap();

        cms.hashers = vec![(1, 0), (2, 0), (3, 0)];

        cms.update(&1, 1);

        cms.update(&12, 2);

        cms.update(&3, 3);

        cms.update(&22, 3);

        let expected = vec![
            0, 1, 5, 3, 0, 0, 0, 0, 0, 0, // depth: 0.
            0, 0, 1, 0, 5, 0, 3, 0, 0, 0, // depth: 1.
            0, 0, 0, 1, 0, 0, 5, 0, 0, 3, // depth: 2.
        ];

        assert_eq!(cms.counts, expected);

        cms.counts[2] = 4;
        cms.counts[9] = 6;

        assert_eq!(cms.query(&1), 1);

        assert_eq!(cms.query(&2), 4);

        assert_eq!(cms.query(&3), 3);

        let builder = PassThroughHasherBuilder {};

        let mut cms: CountMin<u64, PassThroughHasherBuilder, f32> =
            CountMin::with_dimensions(10, 3, builder).unwrap();

        cms.hashers = vec![(1, 0), (2, 0), (3, 0)];

        cms.update(&1, 1.2);

        cms.update(&12, 2.3);

        let expected = vec![
            0.0, 1.2, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // depth: 0.
            0.0, 0.0, 1.2, 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, // depth: 1.
            0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 2.3, 0.0, 0.0, 0.0, // depth: 2.
        ];

        assert_eq!(cms.counts, expected);

        cms.counts[2] = 0.07;
        cms.counts[12] = 2.3;

        assert_eq!(cms.query(&1), 1.2);

        assert_eq!(cms.query(&12), 0.07);
    }
}
