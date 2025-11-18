"""
GPT-2 124M Shakespeare Text Generator - Hugging Face Spaces App
This app uses a trained GPT-2 model to generate Shakespearean text.
"""

import gradio as gr
import torch
import torch.nn.functional as F
import tiktoken
from model import GPT, GPTConfig
import os

# Device setup
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f"Using device: {device}")

# Load model
def load_model(checkpoint_path="model.pt"):
    """Load the trained model from checkpoint"""
    config = GPTConfig()
    model = GPT(config)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} not found")

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Case A: full model saved
    if isinstance(ckpt, GPT):
        print("Loaded full model object")
        model = ckpt
        model.to(device)
        model.eval()
        return model

    # Case B: wrapped checkpoint {"model_state_dict": ...}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        print("Loaded checkpoint with model_state_dict")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Training loss: {ckpt.get('loss', 'N/A')}")
    else:
        # Case C: raw state dict
        print("Loaded raw state_dict")
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model

# Initialize model and tokenizer
model = load_model()
enc = tiktoken.get_encoding('gpt2')

def generate_text(
    prompt,
    max_new_tokens=150,
    temperature=1.0,
    top_k=50,
    num_samples=1
):
    """
    Generate text based on the given prompt
    
    Args:
        prompt: Input text to start generation
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Number of top tokens to consider for sampling
        num_samples: Number of samples to generate
    
    Returns:
        Generated text samples
    """
    if not prompt:
        return "Please provide a prompt!"
    
    model.eval()
    outputs = []
    
    with torch.no_grad():
        for sample_idx in range(num_samples):
            # Encode prompt
            tokens = enc.encode(prompt)
            idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

            for _ in range(max_new_tokens):
                # Forward pass
                logits, _ = model(idx)

                # Take last token logits
                logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k is not None:
                    values, indices = torch.topk(logits, k=top_k)
                    logits = logits.masked_fill(logits < values[:, -1:], float('-inf'))

                probs = F.softmax(logits, dim=-1)

                # Sample token
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                idx = torch.cat([idx, next_token], dim=1)
            
            # Decode the generated sequence
            decoded = enc.decode(idx[0].tolist())
            outputs.append(decoded)
    
    return "\n\n" + "="*80 + "\n\n".join(outputs) if num_samples > 1 else outputs[0]

def generate_simple(prompt, max_tokens, temperature):
    """Wrapper for minimal UI"""
    return generate_text(prompt, max_tokens, temperature, top_k=50, num_samples=1)

# Create minimal Gradio interface
with gr.Blocks(title="Shakespeare GPT", theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🎭 Shakespeare Text Generator")
    
    with gr.Row():
        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="Enter your prompt (e.g., 'ROMEO:', 'Once upon a time')",
            lines=2
        )
    
    with gr.Row():
        max_tokens = gr.Slider(50, 300, value=150, step=10, label="Max Tokens")
        temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
    
    generate_btn = gr.Button("Generate", variant="primary")
    
    output_text = gr.Textbox(
        label="Generated Text",
        lines=15
    )
    
    gr.Examples(
        examples=[
            ["Once upon a time"],
            ["ROMEO:"],
            ["The meaning of life is"],
        ],
        inputs=prompt_input
    )
    
    generate_btn.click(
        fn=generate_simple,
        inputs=[prompt_input, max_tokens, temperature],
        outputs=output_text
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()

