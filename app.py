
import streamlit as st, os, numpy as np, tempfile
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, CompositeVideoClip, AudioFileClip, CompositeAudioClip, afx

st.set_page_config(page_title="ReelForge Pro", page_icon="🎬", layout="wide")
st.title("🎬 ReelForge Pro — AI Reels Maker")

def ensure_size(img, target=(1080,1920)):
    im = Image.open(img).convert("RGB")
    W,H = target; w,h = im.size; tr=W/H; r=w/h
    if r>tr:
        nw=int(h*tr); x=(w-nw)//2; im=im.crop((x,0,x+nw,h)).resize((W,H))
    else:
        nh=int(w/tr); y=(h-nh)//2; im=im.crop((0,y,w,y+nh)).resize((W,H))
    return im

def draw_box(text, width=1080, height=240):
    im = Image.new("RGBA",(width,height),(0,0,0,0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((20,20,width-20,height-20),20, fill=(0,0,0,150))
    try: font=ImageFont.truetype("arial.ttf",44)
    except: font=ImageFont.load_default()
    y=40
    for line in text.splitlines():
        d.text((40,y), line, font=font, fill=(255,255,255,255)); y+=48
    return im

with st.sidebar:
    key = st.text_input("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY",""), type="password")
    segs = st.slider("Segments", 3, 10, 5)
    fps = st.slider("FPS", 24, 60, 30)
    music = st.file_uploader("Background music (.mp3)", type=["mp3"])

topic = st.text_input("Topic / Hook")
imgs = st.file_uploader("Upload 3–10 images", type=["jpg","jpeg","png"], accept_multiple_files=True)

def build_script(client, topic, segs):
    prompt=f"Write exactly {segs} punchy lines for a 9:16 reel on: {topic}. Mix Roman Urdu + English, <=140 chars each, blank-line separated."
    r=client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}], temperature=0.7)
    txt=r.choices[0].message.content.strip()
    parts=[p.strip() for p in txt.split('\n\n') if p.strip()]
    return parts[:segs]

if st.button("📝 Generate Script"):
    if not (key and topic): st.error("Need API key + topic"); st.stop()
    client=OpenAI(api_key=key)
    st.session_state['segments']=build_script(client, topic, segs)
    for i,s in enumerate(st.session_state['segments'],1): st.write(f"**{i}.** {s}")

seg = st.session_state.get('segments')
if seg:
    st.markdown('---'); st.subheader('Build Reel')
    if st.button('🚀 Create Reel'):
        if not imgs: st.error('Upload images'); st.stop()
        dur = 4.0*len(seg)
        per = dur/len(seg)
        clips=[]
        for i,s in enumerate(seg):
            im = ensure_size(imgs[i%len(imgs)])
            base = ImageClip(np.array(im)).set_duration(per)
            sub  = ImageClip(np.array(draw_box(s))).set_duration(per).set_position(('center',1920-240-48))
            clips.append(CompositeVideoClip([base, sub]).set_start(i*per))
        vid=CompositeVideoClip(clips, size=(1080,1920))
        if music:
            mp=tempfile.NamedTemporaryFile(delete=False,suffix='.mp3').name
            open(mp,'wb').write(music.read())
            bg=AudioFileClip(mp).volumex(0.15).afx(afx.audio_loop, duration=vid.duration)
            vid=vid.set_audio(bg)
        out='reel.mp4'; vid.write_videofile(out, fps=fps, codec='libx264', audio_codec='aac')
        st.video(out); st.download_button('⬇️ Download MP4', open(out,'rb').read(), 'reel.mp4','video/mp4')
