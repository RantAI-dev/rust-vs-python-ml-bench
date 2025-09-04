use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::error::Error;
use std::fs;
use std::path::Path;

use clap::Parser;
use tch::{nn, nn::OptimizerConfig, Device, Tensor, Kind};
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use sysinfo::{System, SystemExt, CpuExt};
use anyhow::Result;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    mode: String,
    
    #[arg(short, long)]
    dataset: String,
    
    #[arg(short, long)]
    architecture: String,
    
    #[arg(long, default_value = "{}")]
    hyperparams: String,
    
    #[arg(short, long)]
    run_id: Option<String>,
    
    #[arg(short, long, default_value = ".")]
    output_dir: String,
    
    #[arg(long, default_value = "cpu")]
    device: String,
    
    #[arg(short = 'b', long, default_value = "32")]
    batch_size: usize,
    
    #[arg(long, default_value = "10")]
    epochs: usize,
    
    #[arg(long, default_value = "0.001")]
    learning_rate: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct HardwareConfig {
    cpu_model: String,
    cpu_cores: usize,
    cpu_threads: usize,
    memory_gb: f64,
    gpu_model: Option<String>,
    gpu_memory_gb: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PerformanceMetrics {
    training_time_seconds: Option<f64>,
    inference_latency_ms: Option<f64>,
    throughput_samples_per_second: Option<f64>,
    latency_p50_ms: Option<f64>,
    latency_p95_ms: Option<f64>,
    latency_p99_ms: Option<f64>,
    tokens_per_second: Option<f64>,
    convergence_epochs: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ResourceMetrics {
    peak_memory_mb: f64,
    average_memory_mb: f64,
    cpu_utilization_percent: f64,
    peak_gpu_memory_mb: Option<f64>,
    average_gpu_memory_mb: Option<f64>,
    gpu_utilization_percent: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct QualityMetrics {
    accuracy: Option<f64>,
    f1_score: Option<f64>,
    precision: Option<f64>,
    recall: Option<f64>,
    loss: Option<f64>,
    rmse: Option<f64>,
    mae: Option<f64>,
    r2_score: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Language {
    Python,
    Rust,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TaskType {
    ClassicalMl,
    DeepLearning,
    ReinforcementLearning,
    Llm,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkResult {
    framework: String,
    language: Language,
    task_type: TaskType,
    model_name: String,
    dataset: String,
    run_id: String,
    timestamp: DateTime<Utc>,
    hardware_config: HardwareConfig,
    performance_metrics: PerformanceMetrics,
    resource_metrics: ResourceMetrics,
    quality_metrics: QualityMetrics,
    metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug)]
struct SimpleCNN {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl SimpleCNN {
    fn new(vs: &nn::Path) -> Self {
        Self {
            conv1: nn::conv2d(vs / "conv1", 1, 32, 3, Default::default()),
            conv2: nn::conv2d(vs / "conv2", 32, 64, 3, Default::default()),
            fc1: nn::linear(vs / "fc1", 9216, 128, Default::default()),
            fc2: nn::linear(vs / "fc2", 128, 10, Default::default()),
        }
    }
    
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = xs.view([-1, 1, 28, 28]);
        let xs = xs.apply(&self.conv1).relu();
        let xs = xs.apply(&self.conv2).relu();
        let xs = xs.max_pool2d_default(2);
        let xs = xs.dropout(0.25, true);
        let xs = xs.view([-1, 9216]);
        let xs = xs.apply(&self.fc1).relu();
        let xs = xs.dropout(0.5, true);
        xs.apply(&self.fc2)
    }
}

impl nn::Module for SimpleCNN {
    fn forward(&self, xs: &Tensor) -> Tensor {
        SimpleCNN::forward(self, xs)
    }
}

#[derive(Debug)]
struct LeNet {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
}

impl LeNet {
    fn new(vs: &nn::Path) -> Self {
        Self {
            conv1: nn::conv2d(vs / "conv1", 1, 6, 5, Default::default()),
            conv2: nn::conv2d(vs / "conv2", 6, 16, 5, Default::default()),
            fc1: nn::linear(vs / "fc1", 16 * 4 * 4, 120, Default::default()),
            fc2: nn::linear(vs / "fc2", 120, 84, Default::default()),
            fc3: nn::linear(vs / "fc3", 84, 10, Default::default()),
        }
    }
    
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = xs.view([-1, 1, 28, 28]);
        let xs = xs.apply(&self.conv1).relu().max_pool2d_default(2);
        let xs = xs.apply(&self.conv2).relu().max_pool2d_default(2);
        let xs = xs.view([-1, 16 * 4 * 4]);
        let xs = xs.apply(&self.fc1).relu();
        let xs = xs.apply(&self.fc2).relu();
        xs.apply(&self.fc3)
    }
}

impl nn::Module for LeNet {
    fn forward(&self, xs: &Tensor) -> Tensor {
        LeNet::forward(self, xs)
    }
}

struct CNNBenchmark {
    framework: String,
    device: Device,
    model: Option<Box<dyn nn::Module>>,
    vs: nn::VarStore,
    resource_monitor: ResourceMonitor,
}

impl CNNBenchmark {
    fn new(framework: String, device: Device) -> Self {
        let vs = nn::VarStore::new(device);
        
        Self {
            framework,
            device,
            model: None,
            vs,
            resource_monitor: ResourceMonitor::new(),
        }
    }
    
    fn load_dataset(&self, dataset_name: &str) -> Result<(Tensor, Tensor)> {
        match dataset_name {
            "mnist" => self.load_mnist_dataset(),
            "cifar10" => self.load_cifar10_dataset(),
            _ => Err(anyhow::anyhow!("Unknown dataset: {}", dataset_name)),
        }
    }
    
    fn load_mnist_dataset(&self) -> Result<(Tensor, Tensor)> {
        // Use real MNIST via tch::vision with same normalization as Python
        let dataset = tch::vision::mnist::load_dir("./data")?;
        // dataset.train_images: [60000, 28*28]; normalize to match Python (0.1307, 0.3081)
        let mean = 0.1307;
        let std = 0.3081;
        let images = (&dataset.train_images - mean) / std;
        let labels = dataset.train_labels;
        Ok((images.to_device(self.device), labels.to_device(self.device)))
    }
    
    fn load_cifar10_dataset(&self) -> Result<(Tensor, Tensor)> {
        // Use real CIFAR-10 via tch::vision with same normalization as Python
        let dataset = tch::vision::cifar::load_dir("./data")?;
        // Normalize to Python's (0.4914, 0.4822, 0.4465) and (0.2023, 0.1994, 0.2010)
        let mean = Tensor::f_from_slice(&[0.4914, 0.4822, 0.4465])?.view([3, 1, 1]);
        let std = Tensor::f_from_slice(&[0.2023, 0.1994, 0.2010])?.view([3, 1, 1]);
        let images = (dataset.train_images - &mean) / &std;
        let labels = dataset.train_labels;
        Ok((images.to_device(self.device), labels.to_device(self.device)))
    }
    
    fn create_model(&mut self, architecture: &str, num_classes: i64) -> Result<()> {
        let model: Box<dyn nn::Module> = match architecture {
            "lenet" => Box::new(LeNet::new(&self.vs.root())),
            "simple_cnn" => Box::new(SimpleCNN::new(&self.vs.root())),
            _ => return Err(anyhow::anyhow!("Unknown architecture: {}", architecture)),
        };
        
        self.model = Some(model);
        Ok(())
    }
    
    fn train_model(&mut self, 
                   X_train: &Tensor, 
                   y_train: &Tensor, 
                   epochs: usize, 
                   learning_rate: f64) -> Result<(f64, ResourceMetrics)> {
        self.resource_monitor.start_monitoring();
        
        let start_time = Instant::now();
        
        // Create optimizer using the SAME VarStore as the model
        let mut opt = nn::Adam::default().build(&self.vs, learning_rate)?;
        
        let batch_size = 32; // Match Python batch size
        let n_samples = X_train.size()[0] as usize;
        let n_batches = (n_samples + batch_size - 1) / batch_size;
        
        println!("Training with {} samples, {} batches per epoch", n_samples, n_batches);
        
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut correct = 0i64;
            let mut total = 0i64;
            
            // Process data in batches like Python
            for batch_idx in 0..n_batches {
                let start_idx = batch_idx * batch_size;
                let end_idx = std::cmp::min(start_idx + batch_size, n_samples);
                let actual_batch_size = end_idx - start_idx;
                
                if actual_batch_size == 0 { break; }
                
                // Get batch data
                let batch_X = X_train.narrow(0, start_idx as i64, actual_batch_size as i64);
                let batch_y = y_train.narrow(0, start_idx as i64, actual_batch_size as i64);
                
                // Forward pass
                let batch_loss = if let Some(ref model) = self.model {
                    let ys = model.forward(&batch_X);
                    let loss = ys.cross_entropy_for_logits(&batch_y);
                    
                    // Calculate batch accuracy
                    let predicted = ys.argmax(-1, false);
                    let batch_correct = predicted.eq_tensor(&batch_y).sum(Kind::Int64);
                    let batch_correct_i64 = batch_correct.int64_value(&[]);
                    correct += batch_correct_i64;
                    total += actual_batch_size as i64;
                    
                    loss
                } else {
                    return Err(anyhow::anyhow!("Model not created"));
                };
                
                // Backward pass
                opt.zero_grad();
                batch_loss.backward();
                opt.step();
                
                let batch_loss_value = batch_loss.double_value(&[]);
                epoch_loss += batch_loss_value;
                
                // Print progress every 100 batches like Python
                if batch_idx % 100 == 0 {
                    println!("Epoch: {}, Batch: {}, Loss: {:.4}", epoch, batch_idx, batch_loss_value);
                }
            }
            
            let avg_loss = epoch_loss / n_batches as f64;
            let accuracy = 100.0 * correct as f64 / total as f64;
            println!("Epoch {}: Average Loss: {:.4}, Accuracy: {:.2}%", epoch, avg_loss, accuracy);
        }
        
        let training_time = start_time.elapsed().as_secs_f64();
        let resource_metrics = self.resource_monitor.stop_monitoring();
        
        Ok((training_time, resource_metrics))
    }
    
    fn evaluate_model(&self, X_test: &Tensor, y_test: &Tensor) -> Result<HashMap<String, f64>> {
        if let Some(ref model) = self.model {
            let predictions = model.forward(X_test);
            let predicted_labels = predictions.argmax(-1, false);
            
            // Calculate accuracy
            let correct = predicted_labels.eq_tensor(y_test).sum(Kind::Int64).int64_value(&[]);
            let total = y_test.size()[0] as f64;
            let accuracy = (correct as f64) / total;
            
            // Calculate loss
            let loss = predictions.cross_entropy_for_logits(y_test);
            let loss_value = loss.double_value(&[]);
            
            let mut metrics = HashMap::new();
            metrics.insert("accuracy".to_string(), accuracy);
            metrics.insert("loss".to_string(), loss_value);
            
            Ok(metrics)
        } else {
            Err(anyhow::anyhow!("Model not trained"))
        }
    }
    
    fn run_inference_benchmark(&self, X_test: &Tensor, batch_sizes: &[usize]) -> Result<HashMap<String, f64>> {
        if let Some(ref model) = self.model {
            let mut latencies = Vec::new();
            let mut throughputs = Vec::new();
            
            for &batch_size in batch_sizes {
                let mut batch_latencies = Vec::new();
                
                let n_samples = X_test.size()[0] as usize;
                for i in (0..n_samples).step_by(batch_size) {
                    let end = std::cmp::min(i + batch_size, n_samples);
                    let batch = X_test.narrow(0, i as i64, (end - i) as i64);
                    
                    let start_time = Instant::now();
                    let _predictions = model.forward(&batch);
                    let latency = start_time.elapsed().as_millis() as f64;
                    
                    batch_latencies.push(latency);
                }
                
                let avg_latency = batch_latencies.iter().sum::<f64>() / batch_latencies.len() as f64;
                latencies.push(avg_latency);
                // samples per second
                throughputs.push((batch_size as f64) / (avg_latency / 1000.0));
            }
            
            let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
            let p50 = Self::percentile(&latencies, 50.0);
            let p95 = Self::percentile(&latencies, 95.0);
            let p99 = Self::percentile(&latencies, 99.0);
            
            let mut metrics = HashMap::new();
            metrics.insert("inference_latency_ms".to_string(), avg_latency);
            metrics.insert("latency_p50_ms".to_string(), p50);
            metrics.insert("latency_p95_ms".to_string(), p95);
            metrics.insert("latency_p99_ms".to_string(), p99);
            metrics.insert("throughput_samples_per_second".to_string(), throughputs.iter().sum::<f64>() / throughputs.len() as f64);
            
            Ok(metrics)
        } else {
            Err(anyhow::anyhow!("Model not trained"))
        }
    }

    fn percentile(values: &Vec<f64>, percentile: f64) -> f64 {
        if values.is_empty() { return 0.0; }
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let rank = (percentile / 100.0) * ((sorted.len() - 1) as f64);
        let lower = rank.floor() as usize;
        let upper = rank.ceil() as usize;
        if lower == upper { sorted[lower] } else {
            let weight = rank - (lower as f64);
            sorted[lower] * (1.0 - weight) + sorted[upper] * weight
        }
    }

    fn get_hardware_config(&self) -> HardwareConfig {
        let mut sys = System::new_all();
        sys.refresh_all();
        
        HardwareConfig {
            cpu_model: "Unknown".to_string(),
            cpu_cores: sys.cpus().len(),
            cpu_threads: sys.cpus().len(),
            memory_gb: sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0),
            gpu_model: if self.device == Device::Cuda(0) {
                Some("CUDA GPU".to_string())
            } else {
                None
            },
            gpu_memory_gb: if self.device == Device::Cuda(0) {
                Some(8.0) // Simplified
            } else {
                None
            },
        }
    }
    
    fn run_benchmark(&mut self, 
                     dataset: &str, 
                     architecture: &str, 
                     hyperparams: &HashMap<String, f64>,
                     run_id: &str,
                     mode: &str) -> Result<BenchmarkResult> {
        
        // Load dataset
        let (X, y) = self.load_dataset(dataset)?;
        
        // Split into train/test
        let n_samples = X.size()[0];
        let split_idx = (n_samples * 8) / 10;
        let X_train = X.narrow(0, 0, split_idx);
        let X_test = X.narrow(0, split_idx, n_samples - split_idx);
        let y_train = y.narrow(0, 0, split_idx);
        let y_test = y.narrow(0, split_idx, n_samples - split_idx);
        
        // Get hyperparameters
        let epochs = *hyperparams.get("epochs").unwrap_or(&10.0) as usize;
        let learning_rate = hyperparams.get("learning_rate").unwrap_or(&0.001);
        let num_classes = 10; // Default for MNIST/CIFAR-10
        
        // Create model
        self.create_model(architecture, num_classes)?;
        
        // Get hardware configuration
        let hardware_config = self.get_hardware_config();
        
        if mode == "training" {
            // Training benchmark
            let (training_time, resource_metrics) = self.train_model(&X_train, &y_train, epochs, *learning_rate)?;
            let quality_metrics = self.evaluate_model(&X_test, &y_test)?;
            
            return Ok(BenchmarkResult {
                framework: self.framework.clone(),
                language: Language::Rust,
                task_type: TaskType::DeepLearning,
                model_name: format!("{}_cnn", architecture),
                dataset: dataset.to_string(),
                run_id: run_id.to_string(),
                timestamp: Utc::now(),
                hardware_config,
                performance_metrics: PerformanceMetrics {
                    training_time_seconds: Some(training_time),
                    inference_latency_ms: None,
                    throughput_samples_per_second: None,
                    latency_p50_ms: None,
                    latency_p95_ms: None,
                    latency_p99_ms: None,
                    tokens_per_second: None,
                    convergence_epochs: Some(epochs),
                },
                resource_metrics,
                quality_metrics: QualityMetrics {
                    accuracy: quality_metrics.get("accuracy").copied(),
                    f1_score: None,
                    precision: None,
                    recall: None,
                    loss: quality_metrics.get("loss").copied(),
                    rmse: None,
                    mae: None,
                    r2_score: None,
                },
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("architecture".to_string(), serde_json::Value::String(architecture.to_string()));
                    meta.insert("hyperparameters".to_string(), serde_json::to_value(hyperparams)?);
                    meta.insert("device".to_string(), serde_json::Value::String(format!("{:?}", self.device)));
                    meta.insert("epochs".to_string(), serde_json::Value::Number(serde_json::Number::from(epochs)));
                    meta.insert("learning_rate".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(*learning_rate).unwrap()));
                    meta.insert("dataset_size".to_string(), serde_json::Value::Number(serde_json::Number::from(n_samples)));
                    meta
                },
            });
        } else if mode == "inference" {
            // Train model first
            self.train_model(&X_train, &y_train, epochs, *learning_rate)?;
            
            // Inference benchmark (aligned with Python: [1, 10, 100])
            let inference_metrics = self.run_inference_benchmark(&X_test, &[1, 10, 100])?;
            
            return Ok(BenchmarkResult {
                framework: self.framework.clone(),
                language: Language::Rust,
                task_type: TaskType::DeepLearning,
                model_name: format!("{}_cnn", architecture),
                dataset: dataset.to_string(),
                run_id: run_id.to_string(),
                timestamp: Utc::now(),
                hardware_config,
                performance_metrics: PerformanceMetrics {
                    training_time_seconds: None,
                    inference_latency_ms: inference_metrics.get("inference_latency_ms").copied(),
                    throughput_samples_per_second: inference_metrics.get("throughput_samples_per_second").copied(),
                    latency_p50_ms: inference_metrics.get("latency_p50_ms").copied(),
                    latency_p95_ms: inference_metrics.get("latency_p95_ms").copied(),
                    latency_p99_ms: inference_metrics.get("latency_p99_ms").copied(),
                    tokens_per_second: None,
                    convergence_epochs: None,
                },
                resource_metrics: ResourceMetrics {
                    peak_memory_mb: 0.0,
                    average_memory_mb: 0.0,
                    cpu_utilization_percent: 0.0,
                    peak_gpu_memory_mb: None,
                    average_gpu_memory_mb: None,
                    gpu_utilization_percent: None,
                },
                quality_metrics: QualityMetrics {
                    accuracy: None,
                    f1_score: None,
                    precision: None,
                    recall: None,
                    loss: None,
                    rmse: None,
                    mae: None,
                    r2_score: None,
                },
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("architecture".to_string(), serde_json::Value::String(architecture.to_string()));
                    meta.insert("hyperparameters".to_string(), serde_json::to_value(hyperparams)?);
                    meta.insert("device".to_string(), serde_json::Value::String(format!("{:?}", self.device)));
                    meta.insert("epochs".to_string(), serde_json::Value::Number(serde_json::Number::from(epochs)));
                    meta.insert("learning_rate".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(*learning_rate).unwrap()));
                    meta.insert("dataset_size".to_string(), serde_json::Value::Number(serde_json::Number::from(n_samples)));
                    meta
                },
            });
        }
        
        Err(anyhow::anyhow!("Unknown mode: {}", mode))
    }
}

struct ResourceMonitor {
    start_memory: Option<u64>,
    peak_memory: u64,
    memory_samples: Vec<u64>,
    cpu_samples: Vec<f32>,
    start_time: Option<Instant>,
}

impl ResourceMonitor {
    fn new() -> Self {
        Self {
            start_memory: None,
            peak_memory: 0,
            memory_samples: Vec::new(),
            cpu_samples: Vec::new(),
            start_time: None,
        }
    }
    
    fn start_monitoring(&mut self) {
        let mut sys = System::new_all();
        sys.refresh_all();
        
        self.start_time = Some(Instant::now());
        self.start_memory = Some(sys.used_memory());
        self.peak_memory = sys.used_memory();
        self.memory_samples = vec![sys.used_memory()];
        self.cpu_samples = vec![sys.global_cpu_info().cpu_usage()];
    }
    
    fn stop_monitoring(&mut self) -> ResourceMetrics {
        let mut sys = System::new_all();
        sys.refresh_all();
        
        let final_memory = sys.used_memory();
        let final_cpu = sys.global_cpu_info().cpu_usage();
        
        self.memory_samples.push(final_memory);
        self.cpu_samples.push(final_cpu);
        
        let peak_memory = self.memory_samples.iter().max().unwrap_or(&0);
        let avg_memory = self.memory_samples.iter().sum::<u64>() / self.memory_samples.len() as u64;
        let avg_cpu = self.cpu_samples.iter().sum::<f32>() / self.cpu_samples.len() as f32;
        
        ResourceMetrics {
            peak_memory_mb: *peak_memory as f64 / (1024.0 * 1024.0),
            average_memory_mb: avg_memory as f64 / (1024.0 * 1024.0),
            cpu_utilization_percent: avg_cpu as f64,
            peak_gpu_memory_mb: None,
            average_gpu_memory_mb: None,
            gpu_utilization_percent: None,
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    
    let args = Args::parse();
    
    // Set device
    let device = if args.device == "gpu" || args.device == "cuda" {
        if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            println!("CUDA not available, falling back to CPU");
            Device::Cpu
        }
    } else {
        Device::Cpu
    };
    
    println!("Using device: {:?}", device);
    
    // Generate run ID if not provided
    let run_id = args.run_id.unwrap_or_else(|| Uuid::new_v4().to_string());
    
    // Parse hyperparameters and add CLI params
    let mut hyperparams: HashMap<String, f64> = serde_json::from_str(&args.hyperparams)?;
    hyperparams.insert("epochs".to_string(), args.epochs as f64);
    hyperparams.insert("learning_rate".to_string(), args.learning_rate);
    hyperparams.insert("batch_size".to_string(), args.batch_size as f64);
    
    // Create benchmark instance
    let mut benchmark = CNNBenchmark::new("tch".to_string(), device);
    
    // Run benchmark
    let result = benchmark.run_benchmark(
        &args.dataset,
        &args.architecture,
        &hyperparams,
        &run_id,
        &args.mode,
    )?;
    
    // Save results
    let output_file = format!("{}_{}_{}_{}_results.json", 
                             args.dataset, args.architecture, run_id, args.mode);
    let output_path = Path::new(&args.output_dir).join(output_file);
    
    let json_result = serde_json::to_string_pretty(&result)?;
    fs::write(&output_path, json_result)?;
    
    println!("Benchmark completed. Results saved to: {}", output_path.display());
    
    Ok(())
} 