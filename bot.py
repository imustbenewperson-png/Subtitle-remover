import os, cv2, numpy as np, subprocess, tempfile, shutil, logging
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

logging.basicConfig(level=logging.INFO)
TOKEN = os.environ.get("BOT_TOKEN")

# ── ڕووکارەکان ──
OCR_CONFIDENCE = 0.35
MASK_PADDING   = 14
FRAME_SKIP     = 8   # هەر 8 فریم OCR دەکات — باڵانسی خێرایی/کواڵێتی

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "سڵاو! 👋\n"
        "ڤیدیۆکەت بنێرە، ساب‌تایتڵەکانی دەسڕمەوە و ڤیدیۆی پاکت دەگەڕێنمەوە. ✅"
    )

async def handle_video(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    video = msg.video or msg.document

    if not video:
        await msg.reply_text("تکایە ڤیدیۆ بنێرە 🎬")
        return

    # ── داگرتنی ڤیدیۆ ──
    await msg.reply_text("⬇️ داگرتن...")
    work_dir = tempfile.mkdtemp()
    try:
        file = await ctx.bot.get_file(video.file_id)
        input_path  = os.path.join(work_dir, "input.mp4")
        output_path = os.path.join(work_dir, "output.mp4")
        frames_dir  = os.path.join(work_dir, "frames")
        os.makedirs(frames_dir)
        await file.download_to_drive(input_path)

        # ── زانیاری ڤیدیۆ ──
        cap   = cv2.VideoCapture(input_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        await msg.reply_text(f"🔍 پرۆسێس دەستپێدەکات...\n📊 {total} فریم | {fps:.0f}fps")

        # ── بارکردنی OCR ──
        import easyocr, torch
        reader = easyocr.Reader(['en', 'ar'], gpu=torch.cuda.is_available())

        # ── جیاکردنەوەی فریمەکان ──
        subprocess.run(
            f'ffmpeg -i "{input_path}" -q:v 2 "{frames_dir}/%06d.png" -hide_banner -loglevel error',
            shell=True
        )
        frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.png'))

        def get_mask(frame):
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = reader.readtext(rgb, detail=1)
            mask    = np.zeros(frame.shape[:2], dtype=np.uint8)
            for (bbox, text, conf) in results:
                if conf < OCR_CONFIDENCE:
                    continue
                pts = np.array(bbox, dtype=np.int32)
                x1  = max(0, pts[:,0].min() - MASK_PADDING)
                y1  = max(0, pts[:,1].min() - MASK_PADDING)
                x2  = min(frame.shape[1], pts[:,0].max() + MASK_PADDING)
                y2  = min(frame.shape[0], pts[:,1].max() + MASK_PADDING)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            return mask

        # ── پرۆسێسکردن ──
        last_mask = None
        done = 0
        for i, fname in enumerate(frame_files):
            path  = os.path.join(frames_dir, fname)
            frame = cv2.imread(path)
            if frame is None:
                continue
            if i % FRAME_SKIP == 0:
                last_mask = get_mask(frame)
            mask = last_mask if last_mask is not None else np.zeros(frame.shape[:2], dtype=np.uint8)
            if mask.max() > 0:
                frame = cv2.inpaint(frame, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
            cv2.imwrite(path, frame)
            done += 1
            # پرۆگرێس هەر 20% جارێک
            if done % max(1, len(frame_files) // 5) == 0:
                pct = int(done / len(frame_files) * 100)
                await msg.reply_text(f"⏳ {pct}% تەواوبوو...")

        # ── یەکخستنەوە ──
        await msg.reply_text("🎞️ یەکخستنەوەی ڤیدیۆ...")
        cmd = (
            f'ffmpeg -y -framerate {fps} -i "{frames_dir}/%06d.png" '
            f'-i "{input_path}" -map 0:v -map 1:a '
            f'-c:v libx264 -preset fast -crf 20 -c:a copy '
            f'-pix_fmt yuv420p "{output_path}" -hide_banner -loglevel error'
        )
        r = subprocess.run(cmd, shell=True)
        if r.returncode != 0:
            subprocess.run(
                f'ffmpeg -y -framerate {fps} -i "{frames_dir}/%06d.png" '
                f'-c:v libx264 -crf 20 -pix_fmt yuv420p "{output_path}" -hide_banner -loglevel error',
                shell=True
            )

        # ── ناردن ──
        await msg.reply_text("⬆️ ناردنەوە...")
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        if size_mb < 50:
            with open(output_path, 'rb') as f:
                await ctx.bot.send_video(chat_id=msg.chat_id, video=f,
                                         caption="✅ ساب‌تایتڵەکان سڕانەوە!")
        else:
            await msg.reply_text("⚠️ ڤیدیۆەکە زۆر گەورەیە (+50MB). تکایە بچووکتر بنێرە.")

    except Exception as e:
        logging.exception(e)
        await msg.reply_text(f"❌ هەڵە: {str(e)}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r'^/start'), start))
app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, handle_video))
app.run_polling()
