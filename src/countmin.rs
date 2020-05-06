use std::cmp::Ordering;
use std::f64;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;

use num_traits::{Bounded, CheckedAdd, CheckedMul, One, Zero};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use serde::{Deserialize, Serialize};

use crate::CountMinError;

/// Implements the Count-Min Sketch data structure for summarizing data streams.
///
/// This implementation is based on the original paper of G. Cormode et al:
///
/// *An Improved Data Stream Summary: The Count-Min Sketch and its
/// Applications.*
///
/// - Supports overflow checks.
/// - Can be used with integer and floating point counters.
/// - Supports serialization/deserialization through `serde`.
///
/// # Examples
///
/// ```
/// use std::collections::hash_map::RandomState;
/// use countminsketch::CountMin;
///
/// let mut cms = CountMin::with_dimensions(2048, 4, RandomState::new()).unwrap();
///
/// cms.update(&1234, 1);
/// cms.update(&1234, 3);
/// cms.update(&2345, 2);
///
/// assert_eq!(cms.query(&1234), 4);
/// assert_eq!(cms.query(&2345), 2);
/// ```
///
/// # References
///
/// - ["An Improved Data Stream Summary: The Count-Min Sketch and its
///   Applications."](http://dimacs.rutgers.edu/~graham/pubs/papers/cm-full.pdf)
///
/// - ["Finding Frequent Items in Data Streams."](https://www.vldb.org/pvldb/vol1/1454225.pdf)
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountMin<K, C, S>
where
    K: Hash + ?Sized,
    C: Copy + Zero + One + PartialOrd + Bounded,
    S: BuildHasher,
{
    width:   usize,
    counts:  Vec<C>,
    hashers: Vec<(u64, u64)>,
    builder: S,
    phantom: PhantomData<K>,
}

impl<K, C, S> CountMin<K, C, S>
where
    K: Hash + ?Sized,
    C: Copy + Zero + One + PartialOrd + Bounded,
    S: BuildHasher,
{
    // A large 32-bit prime stored in a u64. Used to create
    // pairwise independent hash functions.
    const MOD: u64 = 2147483647;

    /// Creates a new CountMin sketch instance.
    ///
    /// The dimensions of the sketch are calculated based on provided
    /// error (`epsilon`) and probability of error (`delta`).
    pub fn new(
        epsilon: f64,
        delta: f64,
        builder: S,
    ) -> Result<Self, CountMinError> {
        Self::with_dimensions(
            (f64::consts::E / epsilon).ceil() as usize,
            (1.0 / delta).ln().ceil() as usize,
            builder,
        )
    }

    /// Creates a new CountMin sketch instance.
    ///
    /// The dimensions of the sketch are calculated based on provided
    /// error (`epsilon`) and probability of error (`delta`).
    ///
    /// The random number generator used to create the pairwise independent
    /// hash functions is seeded with `seed`.
    pub fn new_from_seed(
        epsilon: f64,
        delta: f64,
        seed: u64,
        builder: S,
    ) -> Result<Self, CountMinError> {
        Self::with_dimensions_from_seed(
            (f64::consts::E / epsilon).ceil() as usize,
            (1.0 / delta).ln().ceil() as usize,
            seed,
            builder,
        )
    }

    /// Creates a new CountMin sketch instance with specified dimensions.
    pub fn with_dimensions(
        width: usize,
        depth: usize,
        builder: S,
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

    /// Creates a new CountMin sketch instance with specified dimensions.
    ///
    /// The random number generator used to create the pairwise independent
    /// hash functions is seeded with `seed`.
    pub fn with_dimensions_from_seed(
        width: usize,
        depth: usize,
        seed: u64,
        builder: S,
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

    /// Updates `item`'s counter by `diff`.
    pub fn update(&mut self, item: &K, diff: C) {
        // Create a new hasher.
        let mut hasher = self.builder.build_hasher();
        // Calculate the hash.
        item.hash(&mut hasher);
        // Ensure `x` is less than `MOD`.
        let x: u64 = hasher.finish() % Self::MOD;

        // For each row of the sketch increment the corresponding
        // counter by `diff`.
        for (i, (a, b)) in self.hashers.iter().enumerate() {
            let index = i * self.width + self.index(*a, *b, x) % self.width;

            self.counts[index] = self.counts[index] + diff;
        }
    }

    /// Queries for the `item`'s count.
    pub fn query(&self, item: &K) -> C {
        // Create a new hasher.
        let mut hasher = self.builder.build_hasher();
        // Calculate the hash.
        item.hash(&mut hasher);
        // Ensure `x` is less than `MOD`.
        let x: u64 = hasher.finish() % Self::MOD;

        // Return the minimum of the counter values that correspond to `item`.
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

    /// Merges the `other` CountMin sketch into `self`.
    ///
    /// The two sketches must be compatible, that is
    /// [check_compatible_with](CountMin::check_compatible_with) must not
    /// return an error, compatibility is not checked here!
    pub fn merge(&mut self, other: &CountMin<K, C, S>) {
        self.counts
            .iter_mut()
            .zip(other.counts.iter())
            .for_each(|(x, y)| *x = *x + *y);
    }

    /// Computes the inner product of `self` with the `other` CountMin sketch.
    ///
    /// The two sketches must be compatible, that is
    /// [check_compatible_with](CountMin::check_compatible_with) must not
    /// return an error, compatibility is not checked here!
    pub fn inner(&self, other: &CountMin<K, C, S>) -> C {
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

        inner
    }

    /// Returns `true` if all the counters are set to zero.
    pub fn is_empty(&self) -> bool {
        self.counts.iter().all(|c| c.is_zero())
    }

    /// Clears the sketch, sets all the counters to zero.
    pub fn clear(&mut self) {
        self.counts.iter_mut().for_each(|x| *x = C::zero());
    }

    #[inline] // Computes the function: `f(x) = a * x + b`.
    fn index(&self, a: u64, b: u64, x: u64) -> usize {
        // Here a, b and x fit in u32 integers but are stored as u64.
        // This calculation should not overflow.
        let index = (a * x) + b;

        (((index >> 31) + index) & Self::MOD) as usize
    }

    // Builds the pairwise independent hash functions.
    //
    // All functions are in the form of `f(x) = a * x + b` where x is
    // the hash of the input item.
    //
    // If a `seed` is provided, it is used to seed the random number generator.
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

impl<K, C, S> CountMin<K, C, S>
where
    K: Hash + ?Sized,
    C: Copy + Zero + One + PartialOrd + CheckedAdd + CheckedMul + Bounded,
    S: BuildHasher,
{
    /// Updates `item`'s counter by `diff`.
    ///
    /// Returns an error in case of trying to add with overflow.
    pub fn update_checked(
        &mut self,
        item: &K,
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

    /// Merges the `other` CountMin sketch into `self`.
    ///
    /// The two sketches must be compatible, that is
    /// [check_compatible_with](CountMin::check_compatible_with) must not
    /// return an error, compatibility is not checked here!
    ///
    /// Returns an error in case of trying to add with overflow.
    pub fn merge_checked(
        &mut self,
        other: &CountMin<K, C, S>,
    ) -> Result<(), CountMinError> {
        let counts = self.counts.iter_mut().zip(other.counts.iter());

        for (x, y) in counts {
            *x = x.checked_add(y).ok_or(CountMinError::CounterOverflow)?;
        }

        Ok(())
    }

    /// Computes the inner product of `self` with the `other` CountMin sketch.
    ///
    /// The two sketches must be compatible, that is
    /// [check_compatible_with](CountMin::check_compatible_with) must not
    /// return an error, compatibility is not checked here!
    ///
    /// Returns an error in case of trying to add or multiply with overflow.
    pub fn inner_checked(
        &self,
        other: &CountMin<K, C, S>,
    ) -> Result<C, CountMinError> {
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

impl<K, C, S> CountMin<K, C, S>
where
    K: Hash + ?Sized,
    C: Copy + Zero + One + PartialOrd + Bounded,
    S: BuildHasher + Eq,
{
    /// Checks if the `other` CountMin sketch is compatible with `self`.
    ///
    /// For two sketches to be compatible they have to have equal dimensions,
    /// equal pairwise independent hash functions and equal hasher builders.
    pub fn check_compatible_with(
        &self,
        other: &CountMin<K, C, S>,
    ) -> Result<(), CountMinError> {
        if self.width != other.width || self.counts.len() != other.counts.len()
        {
            return Err(CountMinError::IncompatibleDimensions);
        }
        if self.hashers != other.hashers {
            return Err(CountMinError::IncompatibleHashers);
        }
        if self.builder != other.builder {
            return Err(CountMinError::IncompatibleBuilders);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::hash::{BuildHasher, Hasher};

    #[derive(Debug, PartialEq)]
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

    #[derive(Debug, PartialEq, Eq)]
    struct PassThroughHasherBuilder;

    impl BuildHasher for PassThroughHasherBuilder {
        type Hasher = PassThroughHasher;

        fn build_hasher(&self) -> Self::Hasher {
            PassThroughHasher(0)
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct OtherPassThroughHasherBuilder(u64);

    impl BuildHasher for OtherPassThroughHasherBuilder {
        type Hasher = PassThroughHasher;

        fn build_hasher(&self) -> Self::Hasher {
            PassThroughHasher(self.0)
        }
    }

    #[test]
    fn test_new() {
        let cms: CountMin<u64, u32, PassThroughHasherBuilder> =
            CountMin::new(0.001, 0.01, PassThroughHasherBuilder {}).unwrap();

        assert_eq!(cms.width, 2719);

        assert_eq!(cms.counts.len(), 2719 * 5);
    }

    #[test]
    fn test_with_dimensions() {
        let cms: Result<
            CountMin<u64, u32, PassThroughHasherBuilder>,
            CountMinError,
        > = CountMin::with_dimensions(0, 13, PassThroughHasherBuilder {});

        assert_eq!(cms.err(), Some(CountMinError::InvalidDimensions));

        let cms: Result<
            CountMin<u64, u32, PassThroughHasherBuilder>,
            CountMinError,
        > = CountMin::with_dimensions(22, 0, PassThroughHasherBuilder {});

        assert_eq!(cms.err(), Some(CountMinError::InvalidDimensions));

        let cms: Result<
            CountMin<u64, u32, PassThroughHasherBuilder>,
            CountMinError,
        > = CountMin::with_dimensions(1 << 33, 12, PassThroughHasherBuilder {});

        assert_eq!(cms.err(), Some(CountMinError::InvalidDimensions));
    }

    #[test]
    fn test_update() {
        let builder = PassThroughHasherBuilder {};

        let mut cms: CountMin<u64, u32, PassThroughHasherBuilder> =
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

        let mut cms: CountMin<u64, f32, PassThroughHasherBuilder> =
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

        let mut cms: CountMin<u64, u32, PassThroughHasherBuilder> =
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

        let mut cms: CountMin<u64, f32, PassThroughHasherBuilder> =
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

        let mut cms: CountMin<u64, u32, PassThroughHasherBuilder> =
            CountMin::with_dimensions(10, 3, builder).unwrap();

        assert!(cms.is_empty());

        cms.update(&1, 1);

        assert!(!cms.is_empty());
    }

    #[test]
    fn test_update_checked() {
        let builder = PassThroughHasherBuilder {};

        let mut cms: CountMin<u64, u8, PassThroughHasherBuilder> =
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
        */
    }

    #[test]
    fn test_merge() {
        let builder = PassThroughHasherBuilder {};

        let mut cms: CountMin<u64, u32, PassThroughHasherBuilder> =
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

        let mut other: CountMin<u64, u32, PassThroughHasherBuilder> =
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

        cms.merge(&other);

        let expected = vec![
            0, 2, 12, 3, 0, 0, 0, 0, 0, 0, // depth: 0.
            0, 0, 2, 0, 12, 0, 3, 0, 0, 0, // depth: 1.
            0, 0, 0, 2, 0, 0, 12, 0, 0, 3, // depth: 2.
        ];

        assert_eq!(cms.counts, expected);
    }

    #[test]
    fn test_merge_checked() {
        let builder = PassThroughHasherBuilder {};

        let mut cms: CountMin<u64, u8, PassThroughHasherBuilder> =
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

        let mut other: CountMin<u64, u8, PassThroughHasherBuilder> =
            CountMin::with_dimensions(10, 3, builder).unwrap();

        other.hashers = vec![(1, 0), (2, 0), (3, 0)];

        assert!(other.update_checked(&1, 1).is_ok());

        let expected = vec![
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, // depth: 0.
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, // depth: 1.
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // depth: 2.
        ];

        assert_eq!(other.counts, expected);

        cms.merge(&other);

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

        let mut cms: CountMin<u64, u32, PassThroughHasherBuilder> =
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

        let mut cms: CountMin<u64, u32, PassThroughHasherBuilder> =
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

        let mut other: CountMin<u64, u32, PassThroughHasherBuilder> =
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

        assert_eq!(cms.inner(&other), 36);

        other.counts[2] = 4;

        assert_eq!(cms.inner(&other), 21);
    }

    #[test]
    fn test_inner_checked() {
        let builder = PassThroughHasherBuilder {};

        let mut cms: CountMin<u64, u8, PassThroughHasherBuilder> =
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

        let mut other: CountMin<u64, u8, PassThroughHasherBuilder> =
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

    #[test]
    fn test_check_compatible_with() {
        let builder = PassThroughHasherBuilder {};

        let mut cms: CountMin<u64, u32, PassThroughHasherBuilder> =
            CountMin::with_dimensions(10, 3, builder).unwrap();

        cms.hashers = vec![(1, 0), (2, 0), (3, 0)];

        let builder = PassThroughHasherBuilder {};

        let mut other: CountMin<u64, u32, PassThroughHasherBuilder> =
            CountMin::with_dimensions(10, 3, builder).unwrap();

        other.hashers = vec![(1, 0), (2, 0), (3, 0)];

        assert!(cms.check_compatible_with(&other).is_ok());

        other.hashers = vec![(1, 1), (2, 2), (3, 3)];

        assert_eq!(
            cms.check_compatible_with(&other),
            Err(CountMinError::IncompatibleHashers)
        );

        let builder = OtherPassThroughHasherBuilder(0);

        let mut cms: CountMin<u64, u32, OtherPassThroughHasherBuilder> =
            CountMin::with_dimensions(10, 3, builder).unwrap();

        cms.hashers = vec![(1, 0), (2, 0), (3, 0)];

        let builder = OtherPassThroughHasherBuilder(0);

        let mut other: CountMin<u64, u32, OtherPassThroughHasherBuilder> =
            CountMin::with_dimensions(10, 3, builder).unwrap();

        other.hashers = vec![(1, 0), (2, 0), (3, 0)];

        assert!(cms.check_compatible_with(&other).is_ok());

        other.builder = OtherPassThroughHasherBuilder(3);

        assert_eq!(
            cms.check_compatible_with(&other),
            Err(CountMinError::IncompatibleBuilders)
        );

        let builder = OtherPassThroughHasherBuilder(0);

        let mut other: CountMin<u64, u32, OtherPassThroughHasherBuilder> =
            CountMin::with_dimensions(11, 3, builder).unwrap();

        other.hashers = vec![(1, 0), (2, 0), (3, 0)];

        assert_eq!(
            cms.check_compatible_with(&other),
            Err(CountMinError::IncompatibleDimensions)
        );
    }
}
