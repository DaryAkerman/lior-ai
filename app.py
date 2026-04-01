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
TOP_K           = 10  # pages retrieved for general questions
TOP_K_CHAPTER   = 15  # pages retrieved when search is scoped to a chapter
TOP_K_SUMMARY   = 30  # pages retrieved for chapter-summary requests

# Hebrew keywords that signal a broad/summary question
_SUMMARY_KEYWORDS = {'סכם', 'תסכם', 'סיכום', 'תאר', 'הסבר', 'עיקרי', 'עיקר', 'מה עוסק', 'מה נלמד', 'בקצרה', 'כלל'}


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
print("✓ Embedding model ready")


# ── Chapter map ─────────────────────────────────────────────────────────────
# Maps "פרק X" references in queries to the correct PDF page range.
#
# Key insight: "פרק X" appears as a RUNNING HEADER on every page inside that
# chapter (and in the answer-key section). We cannot reliably detect chapter
# START pages by scanning for that text. Instead we:
#   1. Parse the TOC ('תוכן עניינים כללי') for book page numbers.
#   2. Estimate the constant PDF↔book offset via distinctive chapter title phrases.
#   3. Interpolate any chapters whose page numbers weren't captured from the TOC.

HEBREW_ORDINALS = {
    'ראשון': 1, 'שני': 2, 'שלישי': 3, 'רביעי': 4, 'חמישי': 5,
    'שישי': 6, 'שביעי': 7, 'שמיני': 8, 'תשיעי': 9, 'עשירי': 10,
    'אחד עשר': 11, 'שנים עשר': 12, 'שלושה עשר': 13, 'ארבעה עשר': 14,
    'חמישה עשר': 15,
}

HEBREW_LETTERS = {
    'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5,
    'ו': 6, 'ז': 7, 'ח': 8, 'ט': 9, 'י': 10,
}

# ── Chapter map ─────────────────────────────────────────────────────────────
# Hard-coded from 'תוכן עניינים כללי' (TOC, PDF pages 1-12).
# PDF offset = 10  →  pdf_page = book_page + 10.
# Chapters 7 and 9 are linearly interpolated (page numbers unclear in TOC).
_CHAPTER_BOOK_STARTS = {
    1:   3,   # הפסיכולוגיה בחיינו
    2:  37,   # שיטות מחקר בפסיכולוגיה
    3:  93,   # היסודות הביולוגיים של ההתנהגות
    4: 147,   # חישה ותפיסה
    5: 217,   # מודעות ומצבי תודעה
    6: 261,   # למידה וניתוח התנהגות
    7: 317,   # זיכרון (interpolated: midpoint of 261–373)
    8: 373,   # תהליכים קוגניטיביים
    9: 430,   # משכל והערכת משכל (interpolated: midpoint of 373–487)
   10: 487,   # התפתחות האדם
}
_PDF_OFFSET = 10   # pdf_page = book_page + _PDF_OFFSET


def _build_chapter_map(pages_data: list) -> dict:
    last_page = pages_data[-1]['page']
    starts = sorted(
        (chap, max(13, bp + _PDF_OFFSET))
        for chap, bp in _CHAPTER_BOOK_STARTS.items()
    )
    return {
        chap: (start, starts[i + 1][1] - 1 if i + 1 < len(starts) else last_page)
        for i, (chap, start) in enumerate(starts)
    }


chapter_map = _build_chapter_map(pages_data)
print(f"✓ Chapter map: { {c: r for c, r in sorted(chapter_map.items())} }")
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


def detect_chapter(question: str) -> int | None:
    """Return the chapter number if the question explicitly references one."""
    # Arabic numeral: "פרק 7"
    m = re.search(r'פרק\s+(\d+)', question)
    if m:
        return int(m.group(1))
    # Hebrew ordinal word: "פרק שביעי"
    for word, num in HEBREW_ORDINALS.items():
        if re.search(rf'פרק\s+{word}', question):
            return num
    # Hebrew letter: "פרק ז"
    m = re.search(r'פרק\s+([א-י]+)\b', question)
    if m:
        return HEBREW_LETTERS.get(m.group(1))
    return None


def find_relevant_pages(question: str) -> tuple[list[dict], int | None]:
    """
    Return (pages, detected_chapter_num).

    If the question mentions a chapter (e.g. 'פרק 7'), the semantic search
    is restricted to that chapter's PDF pages so the answer stays focused.
    Falls back to a full-book search if no chapter is detected or the chapter
    is not in the map.
    """
    q_emb       = embed_model.encode([question], normalize_embeddings=True)
    chapter_num = detect_chapter(question)

    if chapter_num and chapter_num in chapter_map:
        start, end = chapter_map[chapter_num]
        indices    = [i for i, p in enumerate(pages_data) if start <= p['page'] <= end]
        if indices:
            is_summary = any(kw in question for kw in _SUMMARY_KEYWORDS)
            top_k      = min(TOP_K_SUMMARY if is_summary else TOP_K_CHAPTER, len(indices))
            ch_emb     = page_embeddings[indices]
            scores     = (ch_emb @ q_emb.T).flatten()
            top_local  = np.argsort(scores)[::-1][:top_k]
            top_idx    = sorted([indices[j] for j in top_local],
                                key=lambda i: pages_data[i]['page'])
            return [pages_data[i] for i in top_idx], chapter_num

    # Full-book search (no chapter detected, or chapter not mapped)
    scores         = (page_embeddings @ q_emb.T).flatten()
    top_idx        = np.argsort(scores)[::-1][:TOP_K]
    top_idx_sorted = sorted(top_idx, key=lambda i: pages_data[i]['page'])
    return [pages_data[i] for i in top_idx_sorted], None


def extract_page_numbers(text: str) -> list[int]:
    matches = re.findall(r'\[עמוד\s*(\d+)\]', text)
    return sorted(set(int(p) for p in matches))


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/brain/')
def brain():
    return send_from_directory('.', 'index.html')


@app.route('/brain/brain.glb')
def brain_glb():
    return send_from_directory('.', 'brain.glb')


@app.route('/brain/models/<path:filename>')
def brain_models(filename):
    return send_from_directory('models', filename)


@app.route('/chapters')
def chapters():
    """Debug endpoint — shows the detected chapter → page range map."""
    return jsonify({
        str(chap): {'start': start, 'end': end}
        for chap, (start, end) in sorted(chapter_map.items())
    })


@app.route('/chat', methods=['POST'])
def chat():
    data     = request.get_json()
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'יש להזין שאלה'}), 400

    relevant, chapter_num = find_relevant_pages(question)

    context = "\n\n".join(
        f"=== עמוד {p['page']} ===\n{p['text']}"
        for p in relevant
    )

    # When the query targets a specific chapter, tell Claude explicitly so it
    # stays focused and doesn't guess at scope.
    chapter_note = (
        f'שים לב: המשתמש שואל ספציפית על **פרק {chapter_num}** מהספר. '
        f'התמקד אך ורק בתוכן פרק זה.\n\n'
        if chapter_num else ''
    )

    user_message = (
        f'{chapter_note}'
        f'להלן קטעים רלוונטיים מהספר "מבוא לפסיכולוגיה":\n\n'
        f'{context}\n\n---\n\nשאלה: {question}'
    )

    def generate():
        full_text = ''
        try:
            with client.messages.stream(
                model='claude-opus-4-6',
                max_tokens=4096,
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


BRAIN_SYSTEM_PROMPT = """אתה עוזר לימוד מומחה בנוירואנטומיה ופסיכולוגיה ביולוגית, המתמחה בספר "מבוא לפסיכולוגיה" מאת גריג וזימברדו.
תפקידך לענות על שאלות הקשורות לאזורי המוח, תפקודיהם, והקשר בינהם לבין ההתנהגות האנושית.
ענה בעברית בתשובות קצרות, ממוקדות וברורות.
אם יש מידע רלוונטי בספר, ציין את מספר העמוד בפורמט [עמוד X].
אל תמציא מידע שאינו מופיע בספר."""


@app.route('/brain-chat', methods=['POST'])
def brain_chat():
    data     = request.get_json()
    question = data.get('question', '').strip()
    region   = data.get('region', '').strip()  # currently viewed brain region

    if not question:
        return jsonify({'error': 'יש להזין שאלה'}), 400

    relevant, _ = find_relevant_pages(question)

    context = "\n\n".join(
        f"=== עמוד {p['page']} ===\n{p['text']}"
        for p in relevant
    )

    region_note = f'המשתמש כרגע צופה באזור "{region}" במודל המוח התלת-ממדי. ' if region else ''

    user_message = (
        f'{region_note}'
        f'להלן קטעים רלוונטיים מהספר:\n\n'
        f'{context}\n\n---\n\nשאלה: {question}'
    )

    def generate():
        full_text = ''
        try:
            with client.messages.stream(
                model='claude-opus-4-6',
                max_tokens=1024,
                system=BRAIN_SYSTEM_PROMPT,
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
