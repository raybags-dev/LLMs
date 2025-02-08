import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datetime import datetime
import os
import sys
import gc


def load_config():
    print("Loading configuration...")
    with open("model_config.json", "r") as f:
        return json.load(f)


def log_interaction(prompt, response, log_file):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Response: {response}\n")
        f.write(f"{'='*50}\n")


def print_memory_usage():
    import psutil

    print(
        f"Available Memory: {psutil.virtual_memory().available / (1024 * 1024 * 1024):.1f}GB"
    )


def generate_response(pipe, prompt):
    print(f"\nGenerating response for: '{prompt}'")
    sys.stdout.flush()

    try:
        response = pipe(
            prompt, max_new_tokens=128, num_return_sequences=1, do_sample=True
        )[0]["generated_text"]

        print("\nGeneration completed!")
        return response
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return f"Error: {str(e)}"


def main():
    try:
        config = load_config()
        log_file = "/app/responses/model-responses.log"

        print(f"\nSystem Info:")
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print_memory_usage()

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        print("\nStep 1/3: Loading tokenizer...")
        sys.stdout.flush()
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

        print("Step 2/3: Loading model...")
        sys.stdout.flush()

        # Load model with minimal memory usage
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"], torch_dtype=torch.float32, low_cpu_mem_usage=True
        )

        print("Step 3/3: Initializing pipeline...")
        sys.stdout.flush()
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=config["max_length"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            repetition_penalty=config["repetition_penalty"],
        )

        print("\n✓ Model loaded successfully!")
        print("✓ Responses will be logged to: /app/responses/model-responses.log")
        print("\nReady for input! Type your prompt below (type 'quit' to exit):")

        while True:
            try:
                prompt = input("\n> ").strip()
                if not prompt:
                    continue

                if prompt.lower() == "quit":
                    break

                print("Processing...")
                sys.stdout.flush()

                result = generate_response(pipe, prompt)
                print(f"\nResponse: {result}")

                # Log the interaction
                log_interaction(prompt, result, log_file)
                print("(Response logged to file)")

                # Clean up memory
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except KeyboardInterrupt:
                print("\nExiting gracefully...")
                break
            except Exception as e:
                print(f"Error processing prompt: {str(e)}")
                continue

    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        raise


if __name__ == "__main__":
    main()
