/**
 * IntelliCrash - Edge Inference Module
 * 
 * Target Platform: Raspberry Pi 4
 * Framework: ONNX Runtime (C++)
 * 
 * This module loads the exported `intellicrash_lstm.onnx` model and
 * performs real-time inference on 2-second IMU sliding windows.
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>
#include <onnxruntime_cxx_api.h>

// Configuration
const int SEQ_LEN = 200;       // 2 seconds @ 100Hz
const int NUM_FEATURES = 24;   // 6 raw + 18 engineered features
const float CRASH_THRESHOLD = 0.5f;

// Helper to generate a dummy window for testing
std::vector<float> generate_dummy_window() {
    std::vector<float> window(SEQ_LEN * NUM_FEATURES);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < SEQ_LEN * NUM_FEATURES; ++i) {
        window[i] = dist(gen);
    }
    return window;
}

int main() {
    try {
        std::cout << "--- IntelliCrash Edge Inference (ONNX Runtime) ---" << std::endl;

        // Initialize ONNX Runtime environment
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "IntelliCrashEdge");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(2); // Optimize for RPi5 quad-core
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Load the model
        const char* model_path = "../../models/onnx/intellicrash_lstm.onnx";
        std::cout << "Loading ONNX model from: " << model_path << std::endl;
        Ort::Session session(env, model_path, session_options);

        // Define Input/Output Nodes
        // Based on the export configuration in train_lstm.py
        Ort::AllocatorWithDefaultOptions allocator;
        
        const char* input_names[] = {"input"};
        const char* output_names[] = {"crash_prob", "severity"};
        
        // Prepare input tensor (Batch Size = 1)
        std::vector<int64_t> input_shape = {1, SEQ_LEN, NUM_FEATURES};
        std::vector<float> input_tensor_values = generate_dummy_window();
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            input_tensor_values.data(), 
            input_tensor_values.size(), 
            input_shape.data(), 
            input_shape.size()
        );

        std::cout << "Running inference..." << std::endl;

        // Measure inference latency
        auto start = std::chrono::high_resolution_clock::now();
        
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, 
            input_names, 
            &input_tensor, 
            1, 
            output_names, 
            2
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> latency = end - start;

        // Extract outputs
        float* crash_prob_out = output_tensors[0].GetTensorMutableData<float>();
        float* severity_out = output_tensors[1].GetTensorMutableData<float>();

        std::cout << "\n--- Results ---" << std::endl;
        std::cout << "Crash Probability : " << crash_prob_out[0] * 100 << "%" << std::endl;
        std::cout << "Severity Score    : " << severity_out[0] << std::endl;
        std::cout << "Latency           : " << latency.count() << " ms" << std::endl;
        
        if (crash_prob_out[0] >= CRASH_THRESHOLD) {
            std::cout << ">> ALERT: CRASH DETECTED! Deploying emergency protocols..." << std::endl;
            
            // Calculate CSI (Simulated here for demo)
            float csi_score = std::abs(dist(gen)) / 3.0f + 0.3f; // Generates a score mostly between 0.3 and 1.0
            if (csi_score > 1.0f) csi_score = 1.0f;
            
            // Trigger Silent Alert Dispatcher via Python script
            std::cout << ">> Triggering AI Alert Dispatcher (SMS/Email)..." << std::endl;
            
            char command[256];
            snprintf(command, sizeof(command), "python ../../src/edge/dispatch_alert.py --csi %.2f", csi_score);
            
            int ret = system(command);
            if (ret != 0) {
                std::cerr << ">> Failed to trigger alert dispatcher." << std::endl;
            }
            
        } else {
            std::cout << ">> Normal driving behavior." << std::endl;
            
            // Mock integration of Rash Driving XGBoost Classifier
            float mock_rash_prob = dist(gen); 
            if (mock_rash_prob > 2.0f) { // Simulating a rare rash driving event
                std::cout << ">> WARNING: Rash Driving Pattern Detected (Hard Braking)!" << std::endl;
                // Trigger Pi GPIO Buzzer (Silent to passengers, alerts driver)
                // system("python trigger_buzzer.py"); // Placeholder for GPIO trigger
                std::cout << "   [GPIO Buzzer Activated]" << std::endl;
            }
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
