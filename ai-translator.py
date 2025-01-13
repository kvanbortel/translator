# Import necessary libraries
from transformers import MarianMTModel, MarianTokenizer
import gradio as gr

def translate(text, lang_pair):
    model_name = language_pairs[lang_pair]
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize the input text
    tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')

    # Perform the translation
    translation = model.generate(**tokenized_text)

    # Decode the translated text
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)

    return translated_text

# English <-> French, German
language_pairs = {
    "English to German": "Helsinki-NLP/opus-mt-en-de",
    "German to English": "Helsinki-NLP/opus-mt-de-en",
    "English to French": "Helsinki-NLP/opus-mt-en-fr",
    "French to English": "Helsinki-NLP/opus-mt-fr-en",
}

# Create a Gradio interface
iface = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(label="Enter text to translate"),
        gr.Dropdown(list(language_pairs.keys()), label="Language Pair"),
    ],
    outputs=gr.Textbox(label="Translated text"),
    title="Multi-Language Translator",
    description="Enter text and get the selected language translation."
)

# Launch the Gradio interface
iface.launch()
