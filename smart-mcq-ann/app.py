import os
import pickle
import pandas as pd
import torch
import torch.nn.functional as F
import gradio as gr

try:
    import spaces
except ModuleNotFoundError:
    class _SpacesFallback:
        @staticmethod
        def GPU(func):
            return func

    spaces = _SpacesFallback()

from model import ANNClassifier

# ==========================================================
# Device
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_ann.pt")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "id2label.pkl")

# ==========================================================
# Load Vectorizer
# ==========================================================

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# ==========================================================
# Load Labels
# ==========================================================

with open(LABEL_PATH, "rb") as f:
    id2label = pickle.load(f)

# ==========================================================
# Load Model
# ==========================================================

input_dim = len(vectorizer.get_feature_names_out())

model = ANNClassifier(
    input_dim=input_dim,
    num_classes=len(id2label)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.to(device)
model.eval()

# ==========================================================
# UI CSS
# ==========================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

body{
    font-family: Inter, sans-serif;
    background: #0f172a;
}

.gradio-container{
    max-width: 1300px !important;
    margin: auto;
    padding: 20px;
}

/* Cards */

.block{
    border-radius: 18px !important;
}

.gr-box,
.gr-group{
    border-radius: 18px !important;
}

/* Inputs */

textarea,
input{
    border-radius: 12px !important;
    font-size: 16px !important;
}

/* Prediction Card */

.top-result{
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(99,102,241,.35);
}

.top-result h2{
    margin-bottom: 8px;
    font-size: 24px;
}

.top-result h1{
    margin: 10px 0;
    font-size: 28px;
}

.top-result p{
    margin-top: 10px;
    font-size: 16px;
    opacity: .9;
}

.meta{
    font-size: 18px;
    opacity: .9;
}

/* Top 3 */

.predictions{
    background: #1e293b;
    color: white;
    border-radius: 18px;
    padding: 20px;
}

.prediction{
    background: #334155;
    padding: 16px;
    border-radius: 14px;
    margin-top: 14px;
    transition: .3s;
}

.prediction:hover{
    transform: translateY(-3px);
    background: #475569;
}

.prediction h3{
    margin: 0;
    color: #f8fafc;
}

.prediction p{
    margin-top: 8px;
    color: #cbd5e1;
}

/* Button */

button{
    border-radius: 14px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

.primary{
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
}

/* Footer */

.footer{
    text-align: center;
    color: #94a3b8;
    margin-top: 20px;
}
"""

# ==========================================================
# Prediction
# ==========================================================

@spaces.GPU
def predict(question, option_a, option_b, option_c, option_d, option_e):

    if not all([
        question.strip(),
        option_a.strip(),
        option_b.strip(),
        option_c.strip(),
        option_d.strip(),
        option_e.strip()
    ]):
        empty_df = pd.DataFrame(columns=["Class", "Probability"])
        return (
            "<div class='top-result'>⚠️ Please fill in all fields.</div>",
            empty_df,
            "<div class='predictions'>⚠️ Please fill in all the fields before predicting.</div>"
        )

    text = f"""
Question: {question}

A: {option_a}

B: {option_b}

C: {option_c}

D: {option_d}

E: {option_e}
"""

    features = vectorizer.transform([text]).toarray()

    inputs = torch.tensor(
        features,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)

    probs = probs.squeeze().cpu()

    confidence_dict = {
        id2label[i]: float(probs[i])
        for i in range(len(probs))
    }

    # Prepare DataFrame for BarPlot
    df = pd.DataFrame({
        "Class": list(confidence_dict.keys()),
        "Probability": [v * 100 for v in confidence_dict.values()]
    })

    values, indices = torch.topk(probs, 3)

    medals = ["🥇", "🥈", "🥉"]

    options = {
        "A": option_a,
        "B": option_b,
        "C": option_c,
        "D": option_d,
        "E": option_e
    }

    top_label = id2label[indices[0].item()]
    top_conf = values[0].item() * 100

    top_html = f"""
<div class="top-result">
<h2>🏆 Option {top_label}</h2>
<h1>{options[top_label]}</h1>
<p>
Confidence: <b>{top_conf:.2f}%</b>
</p>
</div>
"""

    result_html = "<div class='predictions'><h2>🎯 Top-3 Predictions</h2>"

    for i in range(3):
        label = id2label[indices[i].item()]
        pct = values[i].item() * 100
        result_html += f"""
<div class="prediction">
<h3>{medals[i]} Option {label}</h3>
<p><b>{options[label]}</b></p>
<p>
Confidence: <b>{pct:.2f}%</b>
</p>
</div>
"""

    result_html += "</div>"

    return top_html, df, result_html

# ==========================================================
# UI Design
# ==========================================================

with gr.Blocks(css=CSS, title="Smart MCQ Solver", theme=gr.themes.Base(primary_hue="indigo", secondary_hue="violet")) as demo:

    gr.Markdown(
        """
        # 🧠 Smart MCQ Solver (ANN)
        Welcome! Enter a multiple-choice question and its options below. Our Artificial Neural Network (ANN) model will analyze the text and predict the top answers with confidence scores.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 📝 Enter Question Details")
            question = gr.Textbox(
                label="Question Statement", 
                lines=4, 
                placeholder="Type or paste your question here..."
            )

            with gr.Row():
                option_a = gr.Textbox(label="Option A")
                option_b = gr.Textbox(label="Option B")
            
            with gr.Row():
                option_c = gr.Textbox(label="Option C")
                option_d = gr.Textbox(label="Option D")

            option_e = gr.Textbox(label="Option E")

            btn = gr.Button("🚀 Predict Answer", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Model Evaluation")
            top_prediction = gr.HTML(
                value="<div class='top-result'>Top prediction will appear here.</div>",
                label="Top Prediction"
            )
            confidence = gr.BarPlot(
                x="Class",
                y="Probability",
                title="Prediction Confidence (%)",
                y_lim=[0, 100]
            )
            output = gr.HTML(
                label="Detailed Top-3 Results", 
                value="<div class='predictions'>Results will appear here after clicking predict.</div>"
            )

    gr.Markdown(
        """
        ---
        <div class="footer">
            Powered by PyTorch • TF-IDF • Gradio
        </div>
        """
    )

    btn.click(
        fn=predict,
        inputs=[
            question,
            option_a,
            option_b,
            option_c,
            option_d,
            option_e
        ],
        outputs=[
            top_prediction,
            confidence,
            output
        ]
    )

if __name__ == "__main__":
    demo.launch()