import os
import json
import re
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

app    = Flask(__name__, static_folder='static')
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

PAGES_FILE      = 'pages_data.json'
EMBEDDINGS_FILE = 'embeddings.npy'
MODEL_NAME      = 'paraphrase-multilingual-MiniLM-L12-v2'
TOP_K           = 6   # pages retrieved per question


# ── Load pre-processed data ────────────────────────────────────────────────
print("🚀 Starting Psychology Chatbot...")

if not Path(PAGES_FILE).exists() or not Path(EMBEDDINGS_FILE).exists():
    print("❌  Processed data not found!")
    print("    Please run first:  python preprocess.py")
    raise SystemExit(1)

with open(PAGES_FILE, encoding='utf-8') as f:
    pages_data = json.load(f)

page_embeddings = np.load(EMBEDDINGS_FILE)   # shape: (n_pages, dim), L2-normalised
print(f"✓ Loaded {len(pages_data)} pages from the book")

print("Loading embedding model...")
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer(MODEL_NAME)
print("✓ Ready!  Visit http://localhost:5000\n")


# ── Helpers ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """אתה עוזר לימוד אקדמי מועיל ומדויק לקורס "מבוא לפסיכולוגיה".
ענה תמיד בעברית בלבד.
התשובות שלך חייבות להתבסס אך ורק על הקטעים מהספר שנשלחו אליך.

חשוב מאוד: בכל פעם שאתה מביא מידע, ציין את מספר העמוד בפורמט [עמוד X].
לדוגמה: "תיאוריית פרויד מסבירה... [עמוד 45]"
אם המידע נמצא בכמה עמודים, ציין את כולם.
אם לא מצאת מידע רלוונטי בקטעים שסופקו, אמור זאת בבירור.
תן תשובות ברורות, מובנות ומפורטות."""


def find_relevant_pages(question: str) -> list[dict]:
    """Return the TOP_K most semantically similar pages, sorted by page number."""
    q_emb = embed_model.encode([question], normalize_embeddings=True)
    scores = (page_embeddings @ q_emb.T).flatten()
    top_idx = np.argsort(scores)[::-1][:TOP_K]
    top_idx_sorted = sorted(top_idx, key=lambda i: pages_data[i]['page'])
    return [pages_data[i] for i in top_idx_sorted]


def extract_page_numbers(text: str) -> list[int]:
    matches = re.findall(r'\[עמוד\s*(\d+)\]', text)
    return sorted(set(int(p) for p in matches))


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data     = request.get_json()
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'יש להזין שאלה'}), 400

    # Retrieve relevant pages
    relevant = find_relevant_pages(question)

    context = "\n\n".join(
        f"=== עמוד {p['page']} ===\n{p['text']}"
        for p in relevant
    )

    user_message = (
        f'להלן קטעים רלוונטיים מהספר "מבוא לפסיכולוגיה":\n\n'
        f'{context}\n\n---\n\nשאלה: {question}'
    )

    def generate():
        full_text = ''
        try:
            with client.messages.stream(
                model='claude-opus-4-6',
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': user_message}]
            ) as stream:
                for chunk in stream.text_stream:
                    full_text += chunk
                    yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"

            pages = extract_page_numbers(full_text)
            yield f"data: {json.dumps({'done': True, 'pages': pages}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


if __name__ == '__main__':
    app.run(debug=False, port=5000)
