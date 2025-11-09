import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_NAME = "incidelen/electra-base-turkish-sentiment-analysis-cased"
LABELS = ["NEGATİF", "NÖTR", "POZİTİF"]  


try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    print(f"✅ Model '{MODEL_NAME}' başarıyla yüklendi.")
except Exception as e:
    print(f"❌ Model yüklenirken hata oluştu: {e}")

def analyze_sentiment(text):
    if not text or text.strip() == "":
        return "nötr"
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        predicted_index = torch.argmax(probabilities, dim=-1).item()
        sentiment = LABELS[predicted_index]

        confidence = probabilities[0][predicted_index].item()
        print(f"Analiz edilen metin: '{text[:50]}...' → Duygu: {sentiment} (güven: {confidence:.2f})")

        return sentiment

    except Exception as e:
        print(f"❌ Duygu analizi sırasında hata: {e}")
        return "nötr"

iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=5, label="Mesajı Buraya Girin"),
    outputs=gr.Textbox(label="Duygu"),
)

if __name__ == "__main__":
    iface.launch()
