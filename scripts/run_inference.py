import os
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import psutil
import humanize
import logging

def load_config(config_path="../config/model_config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(config):
    """Setup logging based on configuration"""
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        filename=config['logging']['file']
    )
    return logging.getLogger(__name__)

def create_model_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def get_model_stats(model, model_path, config):
    """Gather various statistics about the model"""
    stats = {}
    
    # Get number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get model size on disk
    model_size = sum(f.stat().st_size for f in Path(model_path).rglob('*') if f.is_file())
    
    # Get parameter data type
    param_dtype = next(model.parameters()).dtype
    
    # Get memory usage if monitoring is enabled
    if config['monitoring']['track_gpu']:
        memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
        stats["gpu_memory_usage"] = humanize.naturalsize(memory_usage) if memory_usage else "N/A"
    
    stats["total_parameters"] = total_params
    stats["trainable_parameters"] = trainable_params
    stats["parameter_dtype"] = str(param_dtype)
    stats["model_size_bytes"] = model_size
    stats["model_size_human"] = humanize.naturalsize(model_size)
    
    return stats

def main():
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    # Set up model path from config
    model_path = os.path.expanduser(config['model']['cache_dir'])
    create_model_directory(model_path)
    
    # Get model ID from config
    model_id = config['model']['name']
    
    logger.info(f"Loading model {model_id}")
    logger.info(f"Model will be cached in {model_path}")
    
    try:
        # Load tokenizer with config settings
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=model_path,
            **config['tokenizer']
        )
        
        # Load model with config settings
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=model_path,
            torch_dtype=getattr(torch, config['memory']['torch_dtype']),
            device_map=config['hardware']['device_map'],
            trust_remote_code=config['model']['trust_remote_code']
        )
        
        # Get model statistics if enabled
        if config['monitoring']['track_memory']:
            stats = get_model_stats(model, model_path, config)
            logger.info("Model Statistics:")
            for key, value in stats.items():
                logger.info(f"{key}: {value}")
        
        # Prepare input
        question = "Discuss the evolution of the C family of programming languages."
        inputs = tokenizer(question, **config['tokenizer'])
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response using config settings
        logger.info("Generating response...")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                **config['inference']
            )
        
        # Decode and print response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Question: {question}")
        logger.info(f"Response: {response}")
        
        # Save performance metrics if enabled
        if config['monitoring']['save_metrics']:
            # Implementation for saving metrics
            pass
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    
    finally:
        # Cleanup based on config
        if config['cache']['cleanup_on_exit']:
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()