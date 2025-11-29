# 🦀 MedCrab

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/pnnbao97/MedCrab)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/pnnbao-ump/MedCrab-1.5B)
[![Dataset](https://img.shields.io/badge/Dataset-MedCrab-orange)](https://huggingface.co/datasets/pnnbao-ump/MedCrab)

<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/47910ff3-66bc-4218-b55b-1460e6f40b6d" />

A specialized English→Vietnamese medical translation model fine-tuned from Qwen2.5-1.5B for clinical documents, research papers, and biomedical engineering content.

## Resources

- **Model**: [pnnbao-ump/MedCrab-1.5B](https://huggingface.co/pnnbao-ump/MedCrab-1.5B)
- **Dataset**: [pnnbao-ump/MedCrab](https://huggingface.co/datasets/pnnbao-ump/MedCrab)
- **GitHub**: [pnnbao97/MedCrab](https://github.com/pnnbao97/MedCrab)

## Quick Start

```bash
# Install uv
pip install uv

# Clone and setup
git clone https://github.com/pnnbao97/MedCrab.git
cd MedCrab
uv sync

# Run translator
# If your GPU has >= 8 GB VRAM, you can run PDF/Image translation:
uv run pdf_translator.py

# Otherwise, for lower VRAM, run main translator:
uv run main.py
```

## Performance

Evaluated on complex medical passages (100-150 words) covering multi-omics, cellular biology, and pathology:

| Metric | Score |
|--------|-------|
| BLEU | 42–46 |
| COMET | 0.68–0.72 |
| METEOR | 37–40 |

**Key strengths**: High fidelity preservation of biomarkers, pathways, and technical terminology with publication-ready fluency. Maintains stability on 100+ word passages where smaller models (<0.5B) degrade.

## Example

**Input:**
> Recent integrative analyses combining single-cell RNA sequencing, spatial transcriptomics, and high-dimensional mass cytometry have identified a previously uncharacterized population of CD141⁺ dendritic cells in the fibrotic niche of patients with non-alcoholic steatohepatitis (NASH).

**Output:**
> Phân tích tích hợp gần đây kết hợp giải trình tự RNA đơn bào, phiên mã không gian, và đo khối tế bào đa chiều đã xác định quần thể tế bào tua CD141⁺ chưa rõ đặc điểm ở ổ xơ hóa của bệnh nhân viêm gan nhiễm mỡ không do rượu (NASH).

## License
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

This project is licensed under CC BY-NC 4.0. You are free to:
- Share and adapt the material for non-commercial purposes
- Proper attribution must be given

**Commercial Use:** All commercial applications require direct permission from the author. 

## Disclaimer

For research purposes only. Not intended for medical diagnosis or treatment decisions.

---

**Author**: Phạm Nguyễn Ngọc Bảo | [Facebook](https://www.facebook.com/bao.phamnguyenngoc.5/)








