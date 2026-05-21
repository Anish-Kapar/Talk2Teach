# Talk2Teach 🎬
### PDF → Hinglish Video Lecture Generator with AI Chatbot

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](YOUR_STREAMLIT_LINK_HERE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red)](https://streamlit.io)

---

## 🔍 The Problem It Solves

Most Indian students study from dense PDF textbooks — but reading alone doesn't work for everyone. Video lectures are more effective, but creating them is expensive and time-consuming for teachers.

**Talk2Teach automates this.** Upload any PDF (notes, textbook chapters, study material) and the app generates:
- A structured slide deck with key concepts
- A Hinglish audio narration (the way an Indian tutor actually explains)
- A downloadable MP4 video lecture
- A RAG-based chatbot to answer doubts from the same PDF
- An auto-generated practice quiz

This existed because no free tool converts your own study material into a personalized video lecture in minutes — especially in Hinglish, which is how most Indian students actually think and learn.

---

## 🛠️ How It Was Built

### Tech Stack
| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| AI / LLM | Google Gemini API (gemini-2.5-flash) |
| PDF Parsing | PyMuPDF (fitz) |
| Text-to-Speech | gTTS (Google Text-to-Speech) |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Video Rendering | FFmpeg (via subprocess) |
| Image Sourcing | Wikipedia REST API |

### Architecture

```
PDF Upload
    │
    ▼
PyMuPDF (text extraction)
    │
    ├──► ChromaDB (chunked embeddings for chatbot RAG)
    │
    ▼
Gemini API (generates slides: title, bullets, Hinglish script, image prompt)
    │
    ▼
gTTS (Hinglish audio per slide) + Wikipedia (auto visuals)
    │
    ▼
Pillow (slide image composition)
    │
    ▼
FFmpeg (image + audio → per-slide MP4 → concat → final video)
    │
    ▼
Streamlit (preview + download + chatbot + quiz UI)
```

### Key Decisions Made

**1. Hinglish over English TTS**
Most TTS tools generate robotic English. gTTS with `lang='hi'` produces a natural Indian accent that matches how students actually explain concepts to each other. This was a deliberate UX decision.

**2. Wikipedia for slide visuals**
Instead of paid image APIs, the app fetches relevant thumbnails from Wikipedia's free REST API using the slide's topic as a search query. Zero cost, always relevant.

**3. ChromaDB for RAG chatbot**
Rather than sending the full PDF to Gemini every time (expensive, slow), the PDF is chunked and stored in ChromaDB. Each question retrieves only the top-3 relevant chunks — faster and cheaper.

**4. FFmpeg for video assembly**
Each slide becomes an individual MP4 (image loop + audio), then all segments are concatenated using FFmpeg's concat demuxer. This keeps rendering modular — one slide failing doesn't break the whole video.

---

## 💥 What Broke & What Was Learned

**Problem 1: SDK mismatch**
`requirements.txt` had `google-genai` but the code imported `google.generativeai` — two different packages. App would crash at startup. Fixed by correcting the package name to `google-generativeai`.

**Problem 2: JSON parsing failures**
Gemini sometimes returns JSON wrapped in markdown fences (` ```json `) or with trailing commas. Built a `safe_json_from_llm()` function that strips fences, fixes trailing commas, and falls back gracefully instead of crashing.

**Problem 3: ChromaDB collection naming**
PDF filenames with spaces, dots, or special characters caused ChromaDB collection creation to fail silently. Fixed with regex sanitization on the collection name.

**Problem 4: `st.experimental_rerun()` deprecated**
Newer Streamlit versions removed this — replaced with `st.rerun()` throughout.

**Problem 5: FFmpeg not on PATH**
Video generation silently failed on systems where FFmpeg wasn't installed. Added a startup check (`check_ffmpeg()`) and disabled video features gracefully with a clear error message instead of crashing.

**What I learned:**
- LLM outputs are unpredictable — always validate and sanitize before parsing
- Modular pipelines (audio → image → video per slide) are far easier to debug than monolithic ones
- Free APIs (Wikipedia, gTTS) can replace paid ones with smart query design

---

## 🚀 Run Locally

```bash
git clone https://github.com/Anish-Kapar/Talk2Teach.git
cd Talk2Teach
pip install -r requirements.txt
streamlit run app.py
```

Add your Gemini API key in the sidebar after launch.  
Get a free key at: https://aistudio.google.com

**System requirement:** FFmpeg must be installed for video generation.  
Download: https://ffmpeg.org/download.html

---

## 📁 Project Structure

```
Talk2Teach/
├── app.py              # Full Streamlit application
├── requirements.txt    # Python dependencies
└── .gitignore
```

---

## 👤 Built By

**Anish Kapar**  
B.Tech Computer Science, Chitkara University  
GitHub: [@Anish-Kapar](https://github.com/Anish-Kapar)
