#import necessary libraries

import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Check if GPU is available, otherwise use CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# Load pre-trained model and tokenizer from Hugging Face
model_name = "upstage/TinySolar-248m-4k"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad_token to eos_token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move model to the available device (GPU or CPU)
model.to(device)

# Ensure the model is in evaluation mode
model.eval()


def text_streamer(prompt: str, temperature: float = 0.5):
    """
    Generate a continuation of the input prompt using the pre-trained language model.

    Args:
        prompt (str): The input prompt to generate text from.
        temperature (float, optional): The temperature for sampling, controlling randomness in the output. 
                                       Default is 0.5.

    Returns:
        str: The generated text that continues the prompt.
    """
    # Check if the prompt is empty and return an error message
    if not prompt.strip():  # .strip() removes any leading/trailing spaces
        return "No prompt given. Please enter a valid prompt to continue."

    # Define a maximum length for truncation (for example, 512 tokens)
    max_input_length = 512
    
    # Encode the input text with truncation
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, 
                       max_length=max_input_length).to(device)
    
    # Manually create the attention mask (1 for real tokens, 0 for padding tokens)
    attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else None
    
    # Generate the output tokens with dynamic temperature parameter
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'], 
            attention_mask=attention_mask, 
            max_length=200, 
            num_return_sequences=1, 
            no_repeat_ngram_size=2, 
            pad_token_id=tokenizer.eos_token_id, 
            temperature=temperature,  # Only the temperature parameter is used now
            do_sample=True  # Enable sampling for temperature to work
        )
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def start_gradio_interface():
    """
    Set up and launch the Gradio interface for text generation.

    This function creates a Gradio interface that allows users to input a prompt and 
    generate text based on the input using a pre-trained model.
    """
    interface = gr.Interface(
        fn=text_streamer, 
        inputs=[
            gr.Textbox(placeholder="Try 'I am in love with... or I went to...'", 
                       label="Enter your prompt"),
            gr.Slider(minimum=0.1, maximum=0.9, step=0.1, value=0.5, label="Temperature")  # Temperature slider
        ], 
        outputs="text", 
        title="AI Text Streamer: Complete Your Thoughts with AI", 
        description=("This AI-powered text streamer helps you finish your sentences. "
                     "Simply provide half of a sentence, and the model will generate the rest using the autoregressive "
                     "text generation model upstage/TinySolar-248m-4k model. "
                     "Perfect for creative writing, brainstorming, and expanding ideas!"),
        flagging_mode="never"
    )
    interface.launch()


# Run Gradio interface
if __name__ == "__main__":
    start_gradio_interface()
