# Count-Min Sketch

Count-Min sketch is a probabilistic data structure for estimating item 
frequencies using *sublinear space* proposed by G. Cormode et al. Accuracy
guarantees are made in terms of a pair of user specified parameters `epsilon`
and `delta`, meaning that the error is within a factor of `epsilon` with 
probability `delta`.

The implementation provided is based on the
[original paper](http://dimacs.rutgers.edu/~graham/pubs/papers/cm-full.pdf).


## Usage

A simple example:

```rust
use std::collections::hash_map::RandomState;
use countminsketch::CountMin;

let mut cms = CountMin::with_dimensions(2048, 4, RandomState::new()).unwrap();

cms.update(&1234, 1);
cms.update(&1234, 3);
cms.update(&2345, 2);

assert_eq!(cms.query(&1234), 4);
assert_eq!(cms.query(&2345), 2);
```

## Experimental Evaluation

[Here](evaluation/) you can find figures and discussion on experimental evaluation.
