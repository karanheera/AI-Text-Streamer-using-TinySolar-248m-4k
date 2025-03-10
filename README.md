# AI Text Streamer using TinySolar-248m-4k from upstage

## Description

This app allows users to complete unfinished text prompts using a pre-trained language model, **TinySolar-248m-4k**. Built on the **upstage/TinySolar-248m-4k** model, an autoregressive language model fine-tuned for text generation tasks. It provides an interactive interface using **Gradio**, where users can input a sentence fragment, and the app will automatically generate a continuation based on the input.

### Key Features:
- **Text Completion**: The app generates text to complete partially provided prompts using a pre-trained autoregressive model.
- **Temperature Control**: A slider allows users to adjust the randomness of the generated text by modifying the temperature parameter.
- **Interactive Interface**: Built using **Gradio** for easy and interactive text generation.
- **Supports Multi-platforms**: Works on both CPU and GPU (if available), providing flexibility for different hardware configurations.

## Demo
You can try the app by entering an incomplete sentence in the text box. The AI will complete it with the next part of the sentence. Adjust the temperature to change how creative or deterministic the generated continuation is.

## Model Used
**TinySolar-248m-4k** is a compact yet powerful Large Language Model (LLM) developed by Upstage AI. With 248 million parameters, it delivers performance comparable to larger models like GPT-3.5 while being significantly faster and more efficient.

## Technologies Used

- **Gradio**: A Python library for building interactive machine learning demos.
- **Hugging Face Transformers**: Used to load and interact with the **TinySolar-248m-4k** model for text generation.
- **PyTorch**: A deep learning framework used to run the model on CPU or GPU.
- **Python**: The primary programming language used to build the app.

## Installation

### Prerequisites

To run this project locally, you need to have Python installed. You also need to install the required Python packages listed in `requirements.txt`.

1. Clone the repository:
   ```bash
   git clone https://github.com/karanheera/AI-Text-Streamer-using-TinySolar-248m-4k.git
   ```

2. Navigate to the project directory:
   ```bash
   cd AI-Text-Streamer-using-TinySolar-248m-4k
   ```

3. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

To run the app locally, use the following command:
```bash
python app.py
```

Once the app starts, you can interact with the text generation interface through your web browser. The Gradio interface will typically be available at [http://127.0.0.1:7860](http://127.0.0.1:7860/).

## File Structure

```plaintext
/AI-Text-Streamer-using-TinySolar-248m-4k
│
├── app.py              # The main app file
├── CODE_OF_CONDUCT.md  # Code of conduct file
├── CONTRIBUTING.md     # Contribution guidelines
├── LICENSE             # MIT License file
├── README.md           # This file
├── requirements.txt    # List of required Python libraries
```

## Usage

### Enter Prompt
Provide an incomplete sentence or prompt in the text box. For example, you can type "I am in love with..." or "I went to...".

### Adjust Temperature
Use the temperature slider to adjust the creativity level of the generated text. A lower temperature results in more deterministic and logical output, while a higher temperature gives more random and creative results.

### Generate Text
Once you enter your prompt and adjust the temperature, the model will generate a continuation for your input text. The generated text will be displayed below the output area.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

Special thanks to:
- **[Upstage](https://huggingface.co/upstage/TinySolar-248m-4k)**: For providing the **TinySolar-248m-4k** language model that powers this text generation app.
- **[Deeplearning.ai](https://www.deeplearning.ai/)**: For their fantastic educational resources that helped shape the development of this project.
- **[Hugging Face](https://huggingface.co/)**: For their **Transformers** library, which enables seamless integration of pre-trained models for various AI applications.

## Contributing

Contributions are welcome! If you find a bug or want to add a feature, feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push them to your forked repository.
4. Create a pull request with a clear explanation of your changes.

## Contact

For any questions or inquiries, please contact the project maintainer:

**Karan Heera**  
- LinkedIn: [https://www.linkedin.com/in/karanheera/](https://www.linkedin.com/in/karanheera/)  
- GitHub: [https://github.com/karanheera](https://github.com/karanheera)
