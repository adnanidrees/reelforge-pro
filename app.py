import os, time, random, tempfile
import numpy as np
import streamlit as st
from openai import OpenAI, RateLimitError
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta

# ---- Bridge Streamlit Secrets → ENV so os.getenv works everywhere ----
try:
    os.environ.setdefault("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
except Exception:
    pass

st.set_page_config(page_title="ReelForge Pro", page_icon="🎬", layout="wide")
st.title("🎬 ReelForge Pro — AI Reels Maker")

# -------------------- Session utils --------------------
def _now():
    return datetime.utcnow()

def cooldown_ok(seconds=10):
    """Prevent spam: enforce a minimum gap between API calls per session."""
    last = st.session_state.get("_last_call")
    if not last: return True, 0
    remain = seconds - (_now() - last).total_seconds()
    return (remain <= 0), max(0, int(remain))

def set_called():
    st.session_state["_last_call"] = _now()

def get_cache():
    if "script_cache" not in st.session_state:
        st.session_state["script_cache"] = {}
    return st.session_state["script_cache"]

# -------------------- Image helpers --------------------
def ensure_size(img, target=(1080, 1920)):
    im = Image.open(img).convert("RGB")
    W, H = target
    w, h = im.size
    tr = W / H
    r = w / h
    if r > tr:
        nw = int(h * tr)
        x = (w - nw) // 2
        im = im.crop((x, 0, x + nw, h)).resize((W, H))
    else:
        nh = int(w / tr)
        y = (h - nh) // 2
        im = im.crop((0, y, w, y + nh)).resize((W, H))
    return im

def draw_box(text, width=1080, height=240):
    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((20, 20, width - 20, height - 20), 20, fill=(0, 0, 0, 150))
    try:
        font = ImageFont.truetype("arial.ttf", 44)
    except Exception:
        font = ImageFont.load_default()
    y = 40
    for line in text.splitlines():
        d.text((40, y), line, font=font, fill=(255, 255, 255, 255))
        y += 48
    return im

# -------------------- AI + Fallback --------------------
def chat_with_retry(client, **kwargs):
    """Retry wrapper to handle OpenAI rate limits with backoff."""
    delay = 1.0
    last_error = None
    for attempt in range(6):  # up to 6 tries
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            last_error = e
            time.sleep(delay + random.random() * 0.7)
            delay = min(delay * 1.8, 18)
        except Exception as e:
            raise e
    if last_error:
        raise last_error

def build_script_ai(client: OpenAI, topic: str, segs: int):
    prompt = (
        f"Write exactly {segs} punchy lines for a 9:16 short video on: {topic}.\n"
        "Mix Roman Urdu + English. Max 140 chars each. Return lines separated by a blank line."
    )
    r = chat_with_retry(
        client,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=320,
    )
    txt = (r.choices[0].message.content or "").strip()
    parts = [p.strip() for p in txt.split("\n\n") if p.strip()]
    return parts[:segs]

def build_script_lite(topic: str, segs: int):
    """No-AI fallback: template-based hooks."""
    base_hooks = [
        "🔥 {topic}: Ready to level up?",
        "💡 Secret tip for {topic} you’ll wish you knew sooner!",
        "⚡ Quick wins in {topic} (no fluff).",
        "🎯 {topic} ka simple 3-step plan.",
        "🚀 {topic} results faster — let’s go!",
        "✅ {topic} mistakes to avoid today.",
        "✨ {topic}: pro hack inside.",
        "📈 {topic} ka growth formula.",
        "🧠 Smart move in {topic}: try this!",
        "🏁 {topic} — start strong, finish stronger.",
    ]
    lines = []
    for i in range(segs):
        t = random.choice(base_hooks).format(topic=topic)
        # Roman Urdu + English mix
        tweak = [
            " — abhi try karo!",
            " — easy win. 😉",
            " — sahi wala glow-up!",
            " — simple & fast.",
            " — full guide next?",
        ]
        lines.append((t + random.choice(tweak))[:135])
    return lines

# -------------------- Sidebar --------------------
with st.sidebar:
    key = st.text_input("OPENAI_API_KEY (prefer your own key)", os.getenv("OPENAI_API_KEY", ""), type="password")
    segs = st.slider("Segments", 3, 10, 5)
    fps = st.slider("FPS", 24, 60, 30)
    music = st.file_uploader("Background music (.mp3)", type=["mp3"])
    use_lite = st.toggle("Lite mode (no AI if rate-limited)", value=True)
    st.caption("Tip: Button ko bar-bar mat press karo — cooldown lagta hai.")

topic = st.text_input("Topic / Hook")
imgs = st.file_uploader("Upload 3–10 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# -------------------- Script Generation --------------------
if "segments" not in st.session_state:
    st.session_state["segments"] = None

if st.button("📝 Generate Script", type="primary", use_container_width=True):
    if not topic:
        st.error("Please enter a topic.")
    else:
        cache = get_cache()
        k = (topic.strip().lower(), int(segs))
        if k in cache:
            st.session_state["segments"] = cache[k]
            st.info("Loaded from cache (no API call).")
        else:
            ok, wait = cooldown_ok(10)
            if not ok:
                st.warning(f"Cooling down… wait {wait}s, phir try karo.")
            else:
                api_key = key or os.getenv("OPENAI_API_KEY")
                if not api_key and not use_lite:
                    st.error("OPENAI_API_KEY missing. Sidebar me apni key daalein ya Lite mode on rakhein.")
                else:
                    try:
                        if api_key:
                            client = OpenAI(api_key=api_key)
                            with st.spinner("Generating lines (AI)…"):
                                out = build_script_ai(client, topic, segs)
                                st.session_state["segments"] = out
                                cache[k] = out
                                set_called()
                        else:
                            raise RateLimitError("no-key")
                    except RateLimitError:
                        if use_lite:
                            st.info("Rate limit/no key — switching to Lite script.")
                            out = build_script_lite(topic, segs)
                            st.session_state["segments"] = out
                            cache[k] = out
                            set_called()
                        else:
                            st.warning("Rate limit hit. Sidebar me apni key use karein ya thodi der baad try karein.")
                    except Exception as e:
                        st.exception(e)

seg = st.session_state.get("segments")

if seg:
    for i, s in enumerate(seg, 1):
        st.write(f"**{i}.** {s}")

# -------------------- Reel Build --------------------
if seg:
    st.markdown("---")
    st.subheader("Build Reel")
    if st.button("🚀 Create Reel", use_container_width=True):
        if not imgs:
            st.error("Upload 3–10 images first.")
        else:
            try:
                # Lazy import so the app can still load if MoviePy isn't ready at import time
                from moviepy.editor import ImageClip, CompositeVideoClip, AudioFileClip, afx

                total_duration = 4.0 * len(seg)   # ~4s per segment
                per = total_duration / len(seg)

                clips = []
                for i, s in enumerate(seg):
                    im = ensure_size(imgs[i % len(imgs)])
                    base = ImageClip(np.array(im)).set_duration(per)
                    sub = ImageClip(np.array(draw_box(s))).set_duration(per).set_position(("center", 1920 - 240 - 48))
                    clips.append(CompositeVideoClip([base, sub]).set_start(i * per))

                vid = CompositeVideoClip(clips, size=(1080, 1920))

                if music:
                    mp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
                    with open(mp, "wb") as f:
                        f.write(music.read())
                    bg = AudioFileClip(mp).volumex(0.15).afx(afx.audio_loop, duration=vid.duration)
                    vid = vid.set_audio(bg)

                out = "reel.mp4"
                with st.spinner("Rendering video…"):
                    vid.write_videofile(out, fps=fps, codec="libx264", audio_codec="aac")
                st.video(out)
                with open(out, "rb") as f:
                    st.download_button("⬇️ Download MP4", f.read(), "reel.mp4", "video/mp4", use_container_width=True)
            except Exception as e:
                st.exception(e)
