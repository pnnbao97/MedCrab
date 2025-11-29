import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

class MedCrabTranslator:
    """
    Translator class cho MedCrab-1.5B, dịch văn bản y khoa từ tiếng Anh sang tiếng Việt.
    """

    def __init__(self, model_name: str = "pnnbao-ump/MedCrab-1.5b", device: str = "cuda"):
        self.model_name = model_name
        self.device = device

        # Load tokenizer và model
        print(f"Loading model {model_name} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.bfloat16
        )
        self.model.eval()
    
    def _build_prompt(self, text: str) -> str:
        """Tạo prompt chuẩn cho MedCrab"""
        return (
            f"<|user|>: Translate the following medical text from English to Vietnamese:"
            f"<|ENGLISH_TEXT_START|>{text}<|ENGLISH_TEXT_END|>\n"
            "<|assistant|>:<|VIETNAMESE_TEXT_START|>"
        )

    def translate(self, text: str, max_new_tokens: int = 2048) -> str:
        """
        Dịch văn bản tiếng Anh sang tiếng Việt.
        Args:
            text: Văn bản tiếng Anh
            max_new_tokens: số token tối đa sinh ra
        Returns:
            Văn bản tiếng Việt
        """
        prompt = self._build_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Lấy EOS token, fallback nếu không có
        eos_id = self.tokenizer.get_vocab().get("<|VIETNAMESE_TEXT_END|>", self.tokenizer.eos_token_id)

        # Generate với autocast
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_id,
                do_sample=False,
                use_cache=True,
            )

        # Bỏ phần prompt
        input_len = inputs["input_ids"].shape[-1]
        decoded = self.tokenizer.decode(output_tokens[0, input_len:], skip_special_tokens=True)

        # Tách token end nếu có
        if "<|VIETNAMESE_TEXT_END|>" in decoded:
            decoded = decoded.split("<|VIETNAMESE_TEXT_END|>")[0]

        # Optional: post-processing spacing thuật ngữ y khoa
        decoded = self._post_process(decoded)

        return decoded.strip()
    
    def _post_process(self, text: str) -> str:
        """
        Sửa spacing cho các thuật ngữ y khoa bị dính.
        Ví dụ: 'phối tửthụ thể' -> 'phối tử – thụ thể'
        """
        text = text.replace("phối tửthụ thể", "phối tử – thụ thể")
        text = text.replace("miễn dịchchuyển hóa", "miễn dịch – chuyển hóa")
        text = text.replace("oxy hóakhử", "oxy hóa - khử")
        # có thể thêm các quy tắc khác
        return text
    
    def _split_into_chunks(self, text: str, max_words: int = 150) -> list[tuple[str, str]]:
        """
        Chia văn bản dài thành các đoạn nhỏ, GHI NHỚ dấu phân cách gốc.
        Return: list[(chunk_text, separator)]
        """
        # Tách câu và GIỮ LẠI dấu phân cách
        pattern = r'([.!?])\n+'
        parts = re.split(pattern, text)
        
        sentences = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i+1] and parts[i+1] in '.!?':
                # Ghép câu với dấu câu của nó
                sentences.append((parts[i] + parts[i+1], parts[i+1]))
                i += 2
            elif parts[i].strip():
                # Đoạn text không có dấu câu đặc biệt hoặc là newline
                sep = '\n' if '\n' in parts[i] else ' '
                sentences.append((parts[i].strip(), sep))
                i += 1
            else:
                i += 1
        
        chunks = []
        current_chunk = []
        word_count = 0
        last_separator = ' '

        for sentence, separator in sentences:
            words_in_sentence = sentence.split()
            
            if word_count + len(words_in_sentence) > max_words:
                if current_chunk:
                    chunks.append((" ".join(current_chunk), last_separator))
                current_chunk = words_in_sentence
                word_count = len(words_in_sentence)
            else:
                current_chunk.extend(words_in_sentence)
                word_count += len(words_in_sentence)
            
            last_separator = separator
        
        if current_chunk:
            chunks.append((" ".join(current_chunk), last_separator))
        
        return chunks

    def translate_long_text(self, text: str, max_new_tokens: int = 2048) -> str:
        """
        Dịch văn bản dài, GHỮ NGUYÊN dấu phân cách gốc khi ghép.
        """
        chunks = self._split_into_chunks(text, max_words=150)
        translated_parts = []
        
        for i, (chunk, separator) in enumerate(chunks):
            translated = self.translate(chunk, max_new_tokens=max_new_tokens)
            
            # Ghép chunk với dấu phân cách phù hợp
            if i < len(chunks) - 1:  # Không phải chunk cuối
                if separator == '\n':
                    translated_parts.append(translated + '\n')
                elif separator in '.!?':
                    translated_parts.append(translated + ' ')  # Dấu câu đã có sẵn trong translated
                else:
                    translated_parts.append(translated + ' ')
            else:  # Chunk cuối
                translated_parts.append(translated)
        
        return ''.join(translated_parts)