use std::cmp::Ordering;
use std::f64;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;

use num_traits::{Bounded, CheckedAdd, CheckedMul, One, Zero};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use serde::{Deserialize, Serialize};

use crate::CountMinError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountMin<H, B, C>
where
    H: Hash + ?Sized,
    B: BuildHasher,
    C: Copy + Zero + One + PartialOrd + Bounded,
{
    width:   usize,
    counts:  Vec<C>,
    hashers: Vec<(u64, u64)>,
    builder: B,
    phantom: PhantomData<H>,
}

pub trait CountMinChecked<H, B, C>
where
    H: Hash + ?Sized,
    B: BuildHasher,
    C: Copy + Zero + One + PartialOrd + CheckedAdd + CheckedMul + Bounded,
{
    fn update_checked(
        &mut self,
        item: &H,
        diff: C,
    ) -> Result<(), CountMinError>;

    fn merge_checked(
        &mut self,
        other: &CountMin<H, B, C>,
    ) -> Result<(), CountMinError>;

    fn inner_checked(
        &self,
        other: &CountMin<H, B, C>,
    ) -> Result<C, CountMinError>;
}

impl<H, B, C> CountMin<H, B, C>
where
    H: Hash + ?Sized,
    B: BuildHasher,
    C: Copy + Zero + One + PartialOrd + Bounded,
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

    pub fn new_from_seed(
        epsilon: f64,
        delta: f64,
        seed: u64,
        builder: B,
    ) -> Result<Self, CountMinError> {
        Self::with_dimensions_from_seed(
            (f64::consts::E / epsilon).ceil() as usize,
            (1.0 / delta).ln().ceil() as usize,
            seed,
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
            hashers: Self::build_hashers(depth, None),
            builder: builder,
            phantom: PhantomData,
        })
    }

    pub fn with_dimensions_from_seed(
        width: usize,
        depth: usize,
        seed: u64,
        builder: B,
    ) -> Result<Self, CountMinError> {
        if width as u64 > Self::MOD || width < 1 || depth < 1 {
            return Err(CountMinError::InvalidDimensions);
        }

        Ok(CountMin {
            width:   width,
            counts:  vec![C::zero(); width * depth],
            hashers: Self::build_hashers(depth, Some(seed)),
            builder: builder,
            phantom: PhantomData,
        })
    }

    pub fn update(&mut self, item: &H, diff: C) {
        let mut hasher = self.builder.build_hasher();

        item.hash(&mut hasher);

        let x: u64 = hasher.finish() % Self::MOD;

        for (i, (a, b)) in self.hashers.iter().enumerate() {
            let index = i * self.width + self.index(*a, *b, x) % self.width;

            self.counts[index] = self.counts[index] + diff;
        }
    }

    pub fn query(&self, item: &H) -> C {
        let mut hasher = self.builder.build_hasher();

        item.hash(&mut hasher);

        let x: u64 = hasher.finish() % Self::MOD;

        self.hashers
            .iter()
            .enumerate()
            .map(|(i, (a, b))| {
                let index = i * self.width + self.index(*a, *b, x) % self.width;

                self.counts[index]
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.iter().all(|c| c.is_zero())
    }

    pub fn merge(
        &mut self,
        other: &CountMin<H, B, C>,
    ) -> Result<(), CountMinError> {
        // XXX: What about the case when a `self.builder` has state ...

        if self.width != other.width || self.counts.len() != other.counts.len()
        {
            return Err(CountMinError::IncompatibleDimensions);
        }
        if self.hashers != other.hashers {
            return Err(CountMinError::IncompatibleHashers);
        }

        self.counts
            .iter_mut()
            .zip(other.counts.iter())
            .for_each(|(x, y)| *x = *x + *y);

        Ok(())
    }

    pub fn clear(&mut self) {
        self.counts.iter_mut().for_each(|x| *x = C::zero());
    }

    pub fn inner(&self, other: &CountMin<H, B, C>) -> Result<C, CountMinError> {
        // XXX: What about the case when a `self.builder` has state ...

        if self.width != other.width || self.counts.len() != other.counts.len()
        {
            return Err(CountMinError::IncompatibleDimensions);
        }
        if self.hashers != other.hashers {
            return Err(CountMinError::IncompatibleHashers);
        }

        let (mut inner, mut cur) = (C::max_value(), C::zero());

        let counts = self.counts.iter().zip(other.counts.iter()).enumerate();

        for (i, (&a, &b)) in counts {
            cur = cur + a * b;

            if (i + 1) % self.width == 0 {
                if cur < inner {
                    inner = cur;
                }
                cur = C::zero();
            }
        }

        Ok(inner)
    }

    #[inline]
    fn index(&self, a: u64, b: u64, x: u64) -> usize {
        // Here a, b and x fit in u32 integers but are stored as u64.
        // This calculation should not overflow.
        let index = (a * x) + b;

        (((index >> 31) + index) & Self::MOD) as usize
    }

    fn build_hashers(count: usize, seed: Option<u64>) -> Vec<(u64, u64)> {
        let mut rng = match seed {
            Some(seed) => ChaChaRng::seed_from_u64(seed),
            None => ChaChaRng::from_entropy(),
        };

        (0..count)
            .map(|_| {
                (rng.gen::<u64>() & Self::MOD, rng.gen::<u64>() & Self::MOD)
            })
            .collect()
    }
}

impl<H, B, C> CountMinChecked<H, B, C> for CountMin<H, B, C>
where
    H: Hash + ?Sized,
    B: BuildHasher,
    C: Copy + Zero + One + PartialOrd + CheckedAdd + CheckedMul + Bounded,
{
    fn update_checked(
        &mut self,
        item: &H,
        diff: C,
    ) -> Result<(), CountMinError> {
        let mut hasher = self.builder.build_hasher();

        item.hash(&mut hasher);

        let x: u64 = hasher.finish() % Self::MOD;

        for (i, (a, b)) in self.hashers.iter().enumerate() {
            let index = i * self.width + self.index(*a, *b, x) % self.width;

            self.counts[index] = self.counts[index]
                .checked_add(&diff)
                .ok_or(CountMinError::CounterOverflow)?;
        }

        Ok(())
    }

    fn merge_checked(
        &mut self,
        other: &CountMin<H, B, C>,
    ) -> Result<(), CountMinError> {
        if self.width != other.width || self.counts.len() != other.counts.len()
        {
            return Err(CountMinError::IncompatibleDimensions);
        }

        let counts = self.counts.iter_mut().zip(other.counts.iter());

        for (x, y) in counts {
            *x = x.checked_add(y).ok_or(CountMinError::CounterOverflow)?;
        }

        Ok(())
    }

    fn inner_checked(
        &self,
        other: &CountMin<H, B, C>,
    ) -> Result<C, CountMinError> {
        // XXX: What about the case when a `self.builder` has state ...

        if self.width != other.width || self.counts.len() != other.counts.len()
        {
            return Err(CountMinError::IncompatibleDimensions);
        }
        if self.hashers != other.hashers {
            return Err(CountMinError::IncompatibleHashers);
        }

        let (mut inner, mut cur) = (C::max_value(), C::zero());

        let counts = self.counts.iter().zip(other.counts.iter()).enumerate();

        for (i, (a, b)) in counts {
            cur = cur
                .checked_add(
                    &a.checked_mul(b).ok_or(CountMinError::CounterOverflow)?,
                )
                .ok_or(CountMinError::CounterOverflow)?;

            if (i + 1) % self.width == 0 {
                if cur < inner {
                    inner = cur;
                }
                cur = C::zero();
            }
        }

        Ok(inner)
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

    #[test]
    fn test_is_empty() {
        let builder = PassThroughHasherBuilder {};

        let mut cms: CountMin<u64, PassThroughHasherBuilder, u32> =
            CountMin::with_dimensions(10, 3, builder).unwrap();

        assert!(cms.is_empty());

        cms.update(&1, 1);

        assert!(!cms.is_empty());
    }

    #[test]
    fn test_update_checked() {
        let builder = PassThroughHasherBuilder {};

        let mut cms: CountMin<u64, PassThroughHasherBuilder, u8> =
            CountMin::with_dimensions(10, 3, builder).unwrap();

        cms.hashers = vec![(1, 0), (2, 0), (3, 0)];

        assert!(cms.update_checked(&1, 1).is_ok());

        assert!(cms.update_checked(&12, 2).is_ok());

        let expected = vec![
            0, 1, 2, 0, 0, 0, 0, 0, 0, 0, // depth: 0.
            0, 0, 1, 0, 2, 0, 0, 0, 0, 0, // depth: 1.
            0, 0, 0, 1, 0, 0, 2, 0, 0, 0, // depth: 2.
        ];

        assert_eq!(cms.counts, expected);

        assert!(cms.update_checked(&3, 3).is_ok());

        assert!(cms.update_checked(&22, 3).is_ok());

        let expected = vec![
            0, 1, 5, 3, 0, 0, 0, 0, 0, 0, // depth: 0.
            0, 0, 1, 0, 5, 0, 3, 0, 0, 0, // depth: 1.
            0, 0, 0, 1, 0, 0, 5, 0, 0, 3, // depth: 2.
        ];

        assert_eq!(cms.counts, expected);

        assert!(cms.update_checked(&2, 250).is_ok());

        assert_eq!(
            cms.update_checked(&2, 1),
            Err(CountMinError::CounterOverflow)
        );

        /*
            // Does not build for f32.
            //
            let builder = PassThroughHasherBuilder {};

            let mut cms: CountMin<u64, PassThroughHasherBuilder, f32> =
                CountMin::with_dimensions(10, 3, builder).unwrap();

            cms.hashers = vec![(1, 0), (2, 0), (3, 0)];

            assert!(cms.update_checked(&1, 1.2).is_ok());

            assert!(cms.update_checked(&12, 2.3).is_ok());

            let expected = vec![
                0.0, 1.2, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // depth: 0.
                0.0, 0.0, 1.2, 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, // depth: 1.
                0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 2.3, 0.0, 0.0, 0.0, // depth: 2.
            ];

            assert_eq!(cms.counts, expected);
        */
    }

    #[test]
    fn test_merge() {
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

        let builder = PassThroughHasherBuilder {};

        let mut other: CountMin<u64, PassThroughHasherBuilder, u32> =
            CountMin::with_dimensions(10, 3, builder).unwrap();

        other.hashers = vec![(1, 0), (2, 0), (3, 0)];

        other.update(&1, 1);

        other.update(&12, 7);

        let expected = vec![
            0, 1, 7, 0, 0, 0, 0, 0, 0, 0, // depth: 0.
            0, 0, 1, 0, 7, 0, 0, 0, 0, 0, // depth: 1.
            0, 0, 0, 1, 0, 0, 7, 0, 0, 0, // depth: 2.
        ];

        assert_eq!(other.counts, expected);

        assert!(cms.merge(&other).is_ok());

        let expected = vec![
            0, 2, 12, 3, 0, 0, 0, 0, 0, 0, // depth: 0.
            0, 0, 2, 0, 12, 0, 3, 0, 0, 0, // depth: 1.
            0, 0, 0, 2, 0, 0, 12, 0, 0, 3, // depth: 2.
        ];

        assert_eq!(cms.counts, expected);

        other.hashers = vec![(1, 1), (2, 1), (3, 1)];

        assert_eq!(cms.merge(&other), Err(CountMinError::IncompatibleHashers));

        let builder = PassThroughHasherBuilder {};

        let other: CountMin<u64, PassThroughHasherBuilder, u32> =
            CountMin::with_dimensions(10, 4, builder).unwrap();

        assert_eq!(
            cms.merge(&other),
            Err(CountMinError::IncompatibleDimensions)
        );
    }

    #[test]
    fn test_merge_checked() {
        let builder = PassThroughHasherBuilder {};

        let mut cms: CountMin<u64, PassThroughHasherBuilder, u8> =
            CountMin::with_dimensions(10, 3, builder).unwrap();

        cms.hashers = vec![(1, 0), (2, 0), (3, 0)];

        assert!(cms.update_checked(&1, 254).is_ok());

        assert!(cms.update_checked(&12, 2).is_ok());

        assert!(cms.update_checked(&3, 3).is_ok());

        assert!(cms.update_checked(&22, 3).is_ok());

        let expected = vec![
            0, 254, 5, 3, 0, 0, 0, 0, 0, 0, // depth: 0.
            0, 0, 254, 0, 5, 0, 3, 0, 0, 0, // depth: 1.
            0, 0, 0, 254, 0, 0, 5, 0, 0, 3, // depth: 2.
        ];

        assert_eq!(cms.counts, expected);

        let builder = PassThroughHasherBuilder {};

        let mut other: CountMin<u64, PassThroughHasherBuilder, u8> =
            CountMin::with_dimensions(10, 3, builder).unwrap();

        other.hashers = vec![(1, 0), (2, 0), (3, 0)];

        assert!(other.update_checked(&1, 1).is_ok());

        let expected = vec![
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, // depth: 0.
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, // depth: 1.
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // depth: 2.
        ];

        assert_eq!(other.counts, expected);

        assert!(cms.merge(&other).is_ok());

        let expected = vec![
            0, 255, 5, 3, 0, 0, 0, 0, 0, 0, // depth: 0.
            0, 0, 255, 0, 5, 0, 3, 0, 0, 0, // depth: 1.
            0, 0, 0, 255, 0, 0, 5, 0, 0, 3, // depth: 2.
        ];

        assert_eq!(cms.counts, expected);

        assert_eq!(
            cms.merge_checked(&other),
            Err(CountMinError::CounterOverflow)
        );
    }

    #[test]
    fn test_clear() {
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

        cms.clear();

        assert!(cms.is_empty());
    }

    #[test]
    fn test_inner() {
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

        let builder = PassThroughHasherBuilder {};

        let mut other: CountMin<u64, PassThroughHasherBuilder, u32> =
            CountMin::with_dimensions(10, 3, builder).unwrap();

        other.hashers = vec![(1, 0), (2, 0), (3, 0)];

        other.update(&1, 1);

        other.update(&12, 7);

        let expected = vec![
            0, 1, 7, 0, 0, 0, 0, 0, 0, 0, // depth: 0.
            0, 0, 1, 0, 7, 0, 0, 0, 0, 0, // depth: 1.
            0, 0, 0, 1, 0, 0, 7, 0, 0, 0, // depth: 2.
        ];

        assert_eq!(other.counts, expected);

        assert_eq!(cms.inner(&other), Ok(36));

        other.counts[2] = 4;

        assert_eq!(cms.inner(&other), Ok(21));

        other.hashers = vec![(1, 1), (2, 1), (3, 1)];

        assert_eq!(cms.inner(&other), Err(CountMinError::IncompatibleHashers));

        let builder = PassThroughHasherBuilder {};

        let other: CountMin<u64, PassThroughHasherBuilder, u32> =
            CountMin::with_dimensions(10, 4, builder).unwrap();

        assert_eq!(
            cms.inner(&other),
            Err(CountMinError::IncompatibleDimensions)
        );
    }

    #[test]
    fn test_inner_checked() {
        let builder = PassThroughHasherBuilder {};

        let mut cms: CountMin<u64, PassThroughHasherBuilder, u8> =
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

        let builder = PassThroughHasherBuilder {};

        let mut other: CountMin<u64, PassThroughHasherBuilder, u8> =
            CountMin::with_dimensions(10, 3, builder).unwrap();

        other.hashers = vec![(1, 0), (2, 0), (3, 0)];

        other.update(&1, 1);

        other.update(&12, 7);

        let expected = vec![
            0, 1, 7, 0, 0, 0, 0, 0, 0, 0, // depth: 0.
            0, 0, 1, 0, 7, 0, 0, 0, 0, 0, // depth: 1.
            0, 0, 0, 1, 0, 0, 7, 0, 0, 0, // depth: 2.
        ];

        assert_eq!(other.counts, expected);

        other.counts[2] = 50;

        assert_eq!(cms.inner_checked(&other), Ok(36));

        other.counts[14] = 50;
        other.counts[26] = 50;

        assert_eq!(cms.inner_checked(&other), Ok(251));

        other.counts[2] = 51;

        assert_eq!(
            cms.inner_checked(&other),
            Err(CountMinError::CounterOverflow)
        );
    }
}
