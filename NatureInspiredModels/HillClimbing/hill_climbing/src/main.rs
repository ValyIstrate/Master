use std::env;
use rand::Rng;
use std::f64::consts::PI;

const RASRTRIGIN_MIN: f64 = -5.12;
const RASRTRIGIN_MAX: f64 = 5.12;
const ITERATIONS: i32 = 1000;

fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n + x.iter().map(|&xi| xi.powi(2) - 10.0 * (2.0 * PI * xi).cos()).sum::<f64>()
}

fn bitstring_to_real(bitstring: &[u8]) -> f64 {
    let bit_len = bitstring.len();
    let mut int_value = 0;

    // Convert bitstring to an integer
    for (i, &bit) in bitstring.iter().enumerate() {
        int_value |= (bit as usize) << (bit_len - i - 1);
    }

    // Map the integer to the real range [-5.12, 5.12]
    RASRTRIGIN_MIN + (int_value as f64 / (2_usize.pow(bit_len as u32) - 1) as f64) * (RASRTRIGIN_MAX - RASRTRIGIN_MIN)
}

fn validate_args() -> i32 {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <int_value>. Otherwise, the value 2 will be used!", args[0]);
        return 2;
    }

    let value: i32 = match args[1].parse() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("No valid integer value provided. Will use 2, as the default value");
            return 2;
        }
    };

    println!("You provided the value: {}", value);

    return value;
}

fn generate_random_bitstring(bitstring_length: i32) -> String {
    let mut random_number_generator = rand::thread_rng();

    let bitstring: String = (0..bitstring_length)
    .map(|_| if random_number_generator.gen::<bool>() { '1' } else { '0' })
    .collect();

    bitstring
}

fn hill_climbing() {
    let mut iter: i32 = 0;
    
}

fn main() {
    let dimension = validate_args();
    println!("{}", generate_random_bitstring(dimension).to_string());
}
