use std::cmp;
use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::f64;
use std::fmt;
use std::fs::File;
use std::hash::Hash;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::str;
use std::sync::{Arc, Mutex};
use std::thread;

use countminsketch::CountMin;

enum FileType {
    Int,
    UInt,
    Text,
}

struct Configuration {
    eps:    f64,
    jobs:   usize,
    width:  usize,
    depth:  usize,
    ftype:  FileType,
    output: String,
}

fn load<T>(filepath: &String) -> Vec<T>
where
    T: str::FromStr + fmt::Debug,
{
    let file = File::open(filepath).unwrap();

    let reader = BufReader::new(file);

    let mut nums = Vec::with_capacity(10000);

    for line in reader.lines() {
        nums.push(
            line.unwrap()
                .parse::<T>()
                .map_err(|_| "Parsing line failed")
                .expect("Failed to parse line."),
        );
    }

    nums
}

fn run<T>(conf: Configuration, args: Vec<String>, pos: usize)
where
    T: str::FromStr + fmt::Debug + Hash + Eq + Ord,
{
    let conf = Arc::new(conf);
    let args = Arc::new(args);
    let counter = Arc::new(Mutex::new(pos));

    let threads: Vec<thread::JoinHandle<Vec<(f64, f64, f64)>>> = (0..conf.jobs)
        .map(|_| {
            let conf = Arc::clone(&conf);
            let args = Arc::clone(&args);
            let counter = Arc::clone(&counter);

            thread::spawn(move || -> Vec<(f64, f64, f64)> {
                let mut results = Vec::new();

                loop {
                    let i;

                    {
                        let mut counter = counter.lock().unwrap();

                        if *counter == args.len() {
                            break;
                        }

                        i = *counter;

                        *counter += 1;

                        println!(
                            "processing: {}/{}",
                            i - pos,
                            args.len() - pos
                        );
                    }

                    let hashes: Vec<T> = load(&args[i]);

                    let mut cms: CountMin<T, RandomState, u32> =
                        CountMin::with_dimensions(
                            conf.width,
                            conf.depth,
                            RandomState::new(),
                        )
                        .unwrap();

                    let mut counts: HashMap<&T, u32> = HashMap::new();

                    let (mut minres, mut maxres, mut avgres) =
                        (vec![], vec![], vec![]);

                    let samples = cmp::max(1, hashes.len() / 1000);

                    for (j, num) in hashes.iter().enumerate() {
                        cms.update(num, 1);

                        let entry = counts.entry(num).or_insert(0);

                        *entry += 1;

                        if j > 0 && j % samples == 0 {
                            let thres = (conf.eps * j as f64).ceil() as u32;

                            let (_, minre, maxre, avgre) =
                                heavy_hitters_relative_errors(
                                    &cms, &counts, thres,
                                );

                            minres.push(minre);
                            maxres.push(maxre);
                            avgres.push(avgre);
                        }
                    }

                    let thres = (conf.eps * hashes.len() as f64).ceil() as u32;

                    let (_, minre, maxre, avgre) =
                        heavy_hitters_relative_errors(&cms, &counts, thres);

                    minres.push(minre);
                    maxres.push(maxre);
                    avgres.push(avgre);

                    results.push((
                        minres.iter().sum::<f64>() / minres.len() as f64,
                        maxres.iter().sum::<f64>() / maxres.len() as f64,
                        avgres.iter().sum::<f64>() / avgres.len() as f64,
                    ));
                }

                results
            })
        })
        .collect();

    let mut results: Vec<(f64, f64, f64)> = Vec::new();

    for thread in threads {
        match thread.join() {
            Ok(res) => results.extend(res.iter()),
            Err(err) => panic!(err),
        }
    }

    let location = Path::new(&conf.output).to_path_buf();

    let basename = Path::new(&args[pos]).file_name().unwrap().to_str().unwrap();

    let filename = format!("rerr-e{}-{}", conf.eps, basename);
    let filepath = Path::new(&location).join(filename);

    let file = File::create(filepath).unwrap();

    let mut writer = BufWriter::new(file);

    write!(writer, "min-rel-err max-rel-err avg-rel-err\n").unwrap();

    for res in results {
        write!(writer, "{} {} {}\n", res.0, res.1, res.2).unwrap();
    }

    writer.flush().unwrap();
}

fn heavy_hitters_relative_errors<T>(
    cms: &CountMin<T, RandomState, u32>,
    counts: &HashMap<&T, u32>,
    thres: u32,
) -> (u32, f64, f64, f64)
where
    T: str::FromStr + fmt::Debug + Hash + Eq + Ord,
{
    let (mut minre, mut maxre, mut sumre, mut norm) =
        (f64::MAX, f64::MIN, 0.0f64, 0);

    let mut count = 0;

    for (num, act) in counts.iter() {
        if *act >= thres {
            let est = cms.query(num);

            assert!(est >= *act);

            let curre = f64::from(est - *act) / f64::from(*act);

            if curre < minre {
                minre = curre;
            }
            if maxre < curre {
                maxre = curre;
            }

            sumre += curre;

            count += 1;
        }

        norm += act;
    }

    (norm, minre, maxre, sumre / count as f64)
}

fn usage() {
    println!("evaluate [OPTIONS] [INPUT FILE]...");
    println!("         [--eps EPSILON] [--width WIDTH] [--depth DEPTH]");
    println!("         [--type TYPE] [--jobs JOBS] [--output LOCATION]");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut conf = Configuration {
        eps:    0.001,
        jobs:   1,
        width:  (2.0f64 / 0.001f64).ceil() as usize,
        depth:  4,
        ftype:  FileType::UInt,
        output: String::from("./"),
    };

    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "-e" | "--eps" => {
                conf.eps = args[i + 1]
                    .parse::<f64>()
                    .expect("Failed to parse epsilon.");

                conf.width = (2.0f64 / conf.eps).ceil() as usize;

                i += 2;
            },
            "-w" | "--width" => {
                conf.width = args[i + 1]
                    .parse::<usize>()
                    .expect("Failed to parse width.");

                i += 2;
            },
            "-d" | "--depth" => {
                conf.depth = args[i + 1]
                    .parse::<usize>()
                    .expect("Failed to parse depth.");

                i += 2;
            },
            "-t" | "--type" => {
                conf.ftype = match args[i + 1].as_str() {
                    "t" => FileType::Text,
                    "i" => FileType::Int,
                    "u" => FileType::UInt,
                    _ => panic!("Failed to parse type"),
                };

                i += 2;
            },
            "-j" | "--jobs" => {
                conf.jobs = args[i + 1]
                    .parse::<usize>()
                    .expect("Failed to parse jobs.");

                i += 2;
            },
            "-o" | "--output" => {
                conf.output = args[i + 1].clone();

                i += 2;
            },
            "-h" | "--help" => {
                usage();
                std::process::exit(0);
            },
            _ => {
                break;
            },
        }
    }

    if i == args.len() {
        usage();
        std::process::exit(0);
    }

    match conf.ftype {
        FileType::Text => run::<String>(conf, args, i),
        FileType::UInt => run::<u64>(conf, args, i),
        FileType::Int => run::<i64>(conf, args, i),
    }
}
