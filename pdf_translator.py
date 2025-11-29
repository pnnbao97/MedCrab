import gradio as gr
from transformers import AutoModel, AutoTokenizer
from medcrab import MedCrabTranslator
import torch
import os
import sys
import tempfile
import shutil
from PIL import Image, ImageOps
import fitz
import re
import time
from threading import Thread
from queue import Queue
from io import StringIO, BytesIO

# ==================== DEEPSEEK OCR SETUP ====================
OCR_MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'

ocr_tokenizer = AutoTokenizer.from_pretrained(OCR_MODEL_NAME, trust_remote_code=True)

try:
    ocr_model = AutoModel.from_pretrained(
        OCR_MODEL_NAME, 
        attn_implementation='flash_attention_2', 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True, 
        use_safetensors=True
    )
    print("✅ Using Flash Attention 2")
except (ImportError, ValueError):
    print("⚠️ Flash Attention 2 not available, using eager attention")
    ocr_model = AutoModel.from_pretrained(
        OCR_MODEL_NAME, 
        attn_implementation='eager', 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True, 
        use_safetensors=True
    )

ocr_model = ocr_model.eval().cuda()

MODEL_CONFIGS = {
    "Crab": {"base_size": 1024, "image_size": 640, "crop_mode": True},
    "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
}

# ==================== MEDCRAB TRANSLATOR SETUP ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📱 Loading MedCrab translator on {device}...")

translator = MedCrabTranslator(device=device)
print("✅ MedCrab translator loaded successfully")

# ==================== TEXT CLEANING FUNCTIONS ====================
def clean_mathrm(text):
    """Chuyển đổi LaTeX sang HTML với subscript/superscript chỉ trong môi trường toán học"""
    if not text:
        return ""
    
    # 1. Tìm và xử lý tất cả các khối toán học \(...\) và \[...\]
    def process_math_block(match):
        math_content = match.group(1)
        
        # Loại bỏ \mathrm{ }
        math_content = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', math_content)
        
        # Chuyển superscript TRƯỚC (quan trọng vì có thể có cả ^ và _ cùng lúc)
        # Xử lý ^{...} trước để tránh conflict với dấu ngoặc
        math_content = re.sub(r'\^\{([^}]+)\}', r'<sup>\1</sup>', math_content)
        # Sau đó xử lý ^X (chỉ 1 ký tự hoặc cụm ngắn, nhưng KHÔNG phải dấu ngoặc)
        math_content = re.sub(r'\^([A-Za-z0-9+\-]+)', r'<sup>\1</sup>', math_content)
        
        # Chuyển subscript SAU
        math_content = re.sub(r'_\{([^}]+)\}', r'<sub>\1</sub>', math_content)
        # Chỉ match chữ số và dấu, KHÔNG match dấu ngoặc
        math_content = re.sub(r'_([A-Za-z0-9+\-]+)', r'<sub>\1</sub>', math_content)
        
        # Xử lý các ký hiệu đặc biệt LaTeX -> Unicode
        replacements = {
            r'\times': '×',
            r'\pm': '±',
            r'\div': '÷',
            r'\cdot': '·',
            r'\approx': '≈',
            r'\leq': '≤',
            r'\geq': '≥',
            r'\neq': '≠',
            r'\rightarrow': '→',
            r'\leftarrow': '←',
            r'\Rightarrow': '⇒',
            r'\Leftarrow': '⇐',
        }
        for latex_cmd, unicode_char in replacements.items():
            math_content = math_content.replace(latex_cmd, unicode_char)
        
        return math_content
    
    # Xử lý khối \(...\) - DOTALL để match cả xuống dòng
    # Sử dụng non-greedy và xử lý nested brackets
    text = re.sub(r'\\\((.+?)\\\)', process_math_block, text, flags=re.DOTALL)
    
    # Xử lý khối \[...\] - giữ dấu ngoặc vuông bên ngoài
    def process_bracket_block(m):
        # Tạo một match object giả để tái sử dụng hàm process_math_block
        class FakeMatch:
            def __init__(self, content):
                self.content = content
            def group(self, n):
                return self.content
        
        content = process_math_block(FakeMatch(m.group(1)))
        return '[' + content + ']'
    
    text = re.sub(r'\\\[(.+?)\\\]', process_bracket_block, text, flags=re.DOTALL)
    
    # 2. Loại bỏ \mathrm{ } ở ngoài môi trường toán học (nếu còn sót)
    text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)
    
    # 3. Xử lý các ký tự backslash còn sót
    text = text.replace(r'\%', '%')
    
    # 4. Loại bỏ khoảng trắng thừa NHƯNG GIỮ XUỐNG DÒNG và dấu gạch ngang
    # Chỉ gộp khoảng trắng/tab liên tiếp trên cùng 1 dòng
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Chỉ gộp space/tab, không động vào các ký tự khác
        line = re.sub(r'[ \t]+', ' ', line).strip()
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)
    
    return text.strip()

def clean_output(text, include_images=False, remove_labels=False):
    if not text:
        return ""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    img_num = 0
    
    for match in matches:
        if '<|ref|>image<|/ref|>' in match[0]:
            if include_images:
                text = text.replace(match[0], f'\n\n**[Figure {img_num + 1}]**\n\n', 1)
                img_num += 1
            else:
                text = text.replace(match[0], '', 1)
        else:
            if remove_labels:
                text = text.replace(match[0], '', 1)
            else:
                text = text.replace(match[0], match[1], 1)
    
    # Clean \mathrm after processing
    text = clean_mathrm(text)
    
    return text.strip()

# ==================== OCR FUNCTIONS ====================
def ocr_process_image(image, mode="Crab"):
    if image is None:
        return "Error: Upload image"
    
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image = ImageOps.exif_transpose(image)
    
    config = MODEL_CONFIGS[mode]
    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    image.save(tmp.name, 'JPEG', quality=95)
    tmp.close()
    out_dir = tempfile.mkdtemp()
    
    stdout = sys.stdout
    sys.stdout = StringIO()
    
    ocr_model.infer(
        tokenizer=ocr_tokenizer, 
        prompt=prompt, 
        image_file=tmp.name, 
        output_path=out_dir,
        base_size=config["base_size"], 
        image_size=config["image_size"], 
        crop_mode=config["crop_mode"]
    )
    
    result = '\n'.join([l for l in sys.stdout.getvalue().split('\n') 
                        if not any(s in l for s in ['image:', 'other:', 'PATCHES', '====', 'BASE:', '%|', 'torch.Size'])]).strip()
    sys.stdout = stdout
    
    os.unlink(tmp.name)
    shutil.rmtree(out_dir, ignore_errors=True)
    
    if not result:
        return "No text detected"
    
    markdown = clean_output(result, True, True)
    return markdown

def ocr_process_pdf(path, mode, page_num):
    doc = fitz.open(path)
    total_pages = len(doc)
    if page_num < 1 or page_num > total_pages:
        doc.close()
        return f"Invalid page number. PDF has {total_pages} pages."
    
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False)
    img = Image.open(BytesIO(pix.tobytes("png")))
    doc.close()
    
    return ocr_process_image(img, mode)

def ocr_process_file(path, mode, page_num):
    if not path:
        return "Error: Upload file"
    if path.lower().endswith('.pdf'):
        return ocr_process_pdf(path, mode, page_num)
    else:
        return ocr_process_image(Image.open(path), mode)

# ==================== TRANSLATION FUNCTIONS ====================
def split_by_sentences(text: str, max_words: int = 100):
    def count_words(t):
        return len(t.strip().split())
    
    chunks = []
    lines = text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        empty_count = 0
        if not line.strip():
            while i < len(lines) and not lines[i].strip():
                empty_count += 1
                i += 1
            
            if chunks:
                prev_text, prev_newlines = chunks[-1]
                chunks[-1] = (prev_text, prev_newlines + empty_count)
            continue
        
        line = line.strip()
        is_last_line = (i == len(lines) - 1)
        
        if count_words(line) <= max_words:
            chunks.append((line, 0 if is_last_line else 1))
            i += 1
            continue
        
        sentences = re.split(r'(?<=[.!?])\s+', line)
        
        current_chunk = ""
        current_words = 0
        
        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_words = count_words(sentence)
            
            if sentence_words > max_words:
                if current_chunk:
                    chunks.append((current_chunk.strip(), 0))
                    current_chunk = ""
                    current_words = 0
                
                sub_parts = re.split(r',\s*', sentence)
                temp_chunk = ""
                temp_words = 0
                
                for part in sub_parts:
                    part_words = count_words(part)
                    if temp_words + part_words > max_words and temp_chunk:
                        chunks.append((temp_chunk.strip(), 0))
                        temp_chunk = part
                        temp_words = part_words
                    else:
                        if temp_chunk:
                            temp_chunk += ", " + part
                        else:
                            temp_chunk = part
                        temp_words += part_words
                
                if temp_chunk.strip():
                    current_chunk = temp_chunk.strip()
                    current_words = temp_words
                    
            elif current_words + sentence_words <= max_words:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_words += sentence_words
            else:
                chunks.append((current_chunk.strip(), 0))
                current_chunk = sentence
                current_words = sentence_words
        
        if current_chunk.strip():
            chunks.append((current_chunk.strip(), 0 if is_last_line else 1))
        
        i += 1
    
    return chunks

def translate_worker(chunk_queue, result_queue, chunk_text, chunk_index):
    try:
        translated = translator.translate(chunk_text, max_new_tokens=2048)
        result_queue.put((chunk_index, translated.strip()))
    except Exception as e:
        result_queue.put((chunk_index, f"[Lỗi dịch: {str(e)}]"))

def streaming_translate(text: str):
    if not text or not text.strip():
        yield '<div style="padding:20px; color:#ff6b6b;">⚠️ Vui lòng nhập văn bản tiếng Anh để dịch.</div>'
        return

    chunks = split_by_sentences(text, max_words=100)
    accumulated = ""
    
    result_queue = Queue()
    translation_cache = {}
    
    prefetch_count = min(2, len(chunks))
    threads = []
    
    for i in range(prefetch_count):
        chunk_text, _ = chunks[i]
        thread = Thread(target=translate_worker, args=(None, result_queue, chunk_text, i))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    for i, (chunk_text, newline_count) in enumerate(chunks):
        while i not in translation_cache:
            if not result_queue.empty():
                idx, translated = result_queue.get()
                translation_cache[idx] = translated
            else:
                time.sleep(0.01)
        
        next_chunk_idx = i + prefetch_count
        if next_chunk_idx < len(chunks):
            next_chunk_text, _ = chunks[next_chunk_idx]
            thread = Thread(target=translate_worker, args=(None, result_queue, next_chunk_text, next_chunk_idx))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        translated = translation_cache[i]
        
        # Thêm chunk vào accumulated
        if accumulated and not accumulated.endswith('\n'):
            accumulated += " " + translated
        else:
            accumulated += translated
        
        # Stream từng ký tự của chunk hiện tại
        chunk_start = len(accumulated) - len(translated)
        for j in range(len(translated)):
            current_display = accumulated[:chunk_start + j + 1]
            html_output = f'<div style="padding:20px; line-height:1.8; font-size:15px; white-space:pre-wrap; font-family:Arial,sans-serif;">{current_display}</div>'
            yield html_output
            time.sleep(0.015)
        
        # Thêm xuống dòng nếu có
        if newline_count > 0:
            actual_newlines = min(newline_count, 2)
            accumulated += "\n" * actual_newlines
            html_output = f'<div style="padding:20px; line-height:1.8; font-size:15px; white-space:pre-wrap; font-family:Arial,sans-serif;">{accumulated}</div>'
            yield html_output
    
    for thread in threads:
        thread.join(timeout=1.0)

# ==================== UI HELPER FUNCTIONS ====================
def load_image(file_path, page_num_str="1"):
    if not file_path:
        return None
    try:
        try:
            page_num = int(page_num_str)
        except (ValueError, TypeError):
            page_num = 1
            
        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            page_idx = max(0, min(page_num - 1, len(doc) - 1))
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False)
            img = Image.open(BytesIO(pix.tobytes("png")))
            doc.close()
            return img
        else:
            return Image.open(file_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def get_pdf_page_count(file_path):
    if not file_path or not file_path.lower().endswith('.pdf'):
        return 1
    try:
        doc = fitz.open(file_path)
        count = len(doc)
        doc.close()
        return count
    except Exception as e:
        print(f"Error reading PDF page count: {e}")
        return 1

def update_page_info(file_path):
    if not file_path:
        return gr.update(label="Số trang (chỉ dùng cho PDF, mặc định: 1)")
    if file_path.lower().endswith('.pdf'):
        page_count = get_pdf_page_count(file_path)
        return gr.update(
            label=f"Số trang (PDF có {page_count} trang, nhập 1-{page_count})",
            value="1"
        )
    return gr.update(
        label="Số trang (chỉ dùng cho PDF, mặc định: 1)",
        value="1"
    )

# ==================== COMBINED OCR + TRANSLATION ====================
def ocr_and_translate_streaming(file_path, mode, page_num_str):
    """Hàm kết hợp: OCR trước, sau đó dịch streaming"""
    if not file_path:
        yield '<div style="padding:20px; color:#ff6b6b;">⚠️ Vui lòng tải file lên trước!</div>'
        return
    
    # Bước 1: OCR
    yield '<div style="padding:20px; color:#4CAF50;">🔍 Đang quét OCR...</div>'
    try:
        try:
            page_num = int(page_num_str)
        except (ValueError, TypeError):
            page_num = 1
        
        markdown = ocr_process_file(file_path, mode, page_num)
        
        if not markdown or markdown.startswith("Error") or markdown.startswith("Invalid"):
            yield f'<div style="padding:20px; color:#ff6b6b;">❌ Lỗi OCR: {markdown}</div>'
            return
        
    except Exception as e:
        yield f'<div style="padding:20px; color:#ff6b6b;">❌ Lỗi OCR: {str(e)}</div>'
        return
    
    # Bước 2: Dịch streaming
    yield '<div style="padding:20px; color:#2196F3;">🦀 Đang dịch...</div>'
    time.sleep(0.5)  # Delay nhỏ để user thấy message
    
    try:
        yield from streaming_translate(markdown)
    except Exception as e:
        yield f'<div style="padding:20px; color:#ff6b6b;">❌ Lỗi dịch: {str(e)}</div>'

# ==================== GRADIO INTERFACE ====================
css = """
footer { visibility: hidden }
.main-title { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css, title="OCR + Translation") as demo:
    
    gr.Markdown("""
    <div class="main-title">
    <h1>🦀 MedCrab Translation</h1>
    <p><b>Quét PDF Y khoa → Dịch trực tiếp sang tiếng Việt (Streaming)</b></p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Tải file lên")
            file_in = gr.File(label="PDF hoặc Hình ảnh", file_types=["image", ".pdf"], type="filepath")
            input_img = gr.Image(label="Xem trước", type="pil", height=300)
            
            page_input = gr.Textbox(
                label="Số trang (chỉ dùng cho PDF, mặc định: 1)", 
                value="1",
                placeholder="Nhập số trang..."
            )
            mode = gr.Dropdown(list(MODEL_CONFIGS.keys()), value="Crab", label="Chế độ OCR")
            
            gr.Markdown("### 🦀 Quét và Dịch")
            process_btn = gr.Button("🚀 Quét OCR + Dịch tiếng Việt", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### 📄 Kết quả dịch tiếng Việt (Streaming)")
            translation_output = gr.HTML(
                label="",
                value=""
            )
    
    with gr.Accordion("ℹ️ Hướng dẫn sử dụng", open=False):
        gr.Markdown("""
        **Quy trình đơn giản:**
        1. 📤 Tải lên file PDF hoặc hình ảnh y khoa
        
        **Chế độ OCR:**
        - **Crab**: 1024 base + 640 tiles (Tốt nhất, cân bằng)
        - **Base**: 1024×1024 (Nhanh hơn)
        """)
    
    # Event handlers
    file_in.change(load_image, [file_in, page_input], [input_img])
    file_in.change(update_page_info, [file_in], [page_input])
    page_input.change(load_image, [file_in, page_input], [input_img])
    
    process_btn.click(
        ocr_and_translate_streaming, 
        [file_in, mode, page_input], 
        [translation_output]
    )

if __name__ == "__main__":
    print("🚀 Starting OCR + Translation App (Single Button)...")
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False,
        quiet=False,
    )