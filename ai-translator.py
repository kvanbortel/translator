# Import necessary libraries
from transformers import MarianMTModel, MarianTokenizer, pipeline
import gradio as gr

# Load English sentiment and emotion models
sentiment_analyzer = pipeline("sentiment-analysis")
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

def analyze_english(text):
    sentiment = sentiment_analyzer(text)[0]  # E.g., {'label': 'POSITIVE', 'score': 0.98}
    emotion = emotion_analyzer(text)[0]      # E.g., {'label': 'joy', 'score': 0.85}
    return sentiment, emotion

# Load German sentiment and emotion models
german_sentiment_analyzer = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")
german_emotion_analyzer = pipeline("text-classification", model="FacebookAI/xlm-roberta-base")
# german_emotion_analyzer = pipeline("text-classification", model="padmalcom/wav2vec2-large-emotion-detection-german")

def analyze_german(text):
    sentiment = german_sentiment_analyzer(text)[0]  # E.g., {'label': 'positive', 'score': 0.92}
    emotion = german_emotion_analyzer(text)[0]      # E.g., {'label': 'Freude', 'score': 0.88}
    return sentiment, emotion


def translate_and_analyze(text, lang_pair):
    model_name = language_pairs[lang_pair]
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize the input text
    tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')

    # Perform the translation
    translation = model.generate(**tokenized_text)

    # Decode the translated text
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)

    # Analyze English
    english_sentiment, english_emotion = analyze_english(text)

    # Analyze German
    german_sentiment, german_emotion = analyze_german(translated_text)

    # Display results
    return {
        "Original Text": text,
        "English Sentiment": english_sentiment,
        "English Emotion": english_emotion,
        "Translated Text": translated_text,
        "German Sentiment": german_sentiment,
        "German Emotion": german_emotion,
    }

# English <-> French, German
language_pairs = {
    "English to German": "Helsinki-NLP/opus-mt-en-de",
    "German to English": "Helsinki-NLP/opus-mt-de-en",
    "English to French": "Helsinki-NLP/opus-mt-en-fr",
    "French to English": "Helsinki-NLP/opus-mt-fr-en",
}

def translate_and_analyze_gradio(text, lang_pair):
    results = translate_and_analyze(text, lang_pair)
    return (
        results["Translated Text"],
        results["English Sentiment"],
        results["English Emotion"],
        results["German Sentiment"],
        results["German Emotion"],
    )

# Create a Gradio interface
iface = gr.Interface(
    fn=translate_and_analyze_gradio,
    inputs=[
        gr.Textbox(label="Enter text to translate"),
        gr.Dropdown(list(language_pairs.keys()), label="Language Pair"),
    ],
    outputs=[
        gr.Textbox(label="Translated Text"),
        gr.Textbox(label="English Sentiment"),
        gr.Textbox(label="English Emotion"),
        gr.Textbox(label="German Sentiment"),
        gr.Textbox(label="German Emotion"),
    ],
    title="Multi-Language Translator",
    description="Enter text and get the selected language translation."
)

# Launch the Gradio interface
iface.launch()
