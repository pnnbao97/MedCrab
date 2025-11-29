# 🦀 MedCrab-1.5B

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

# Run PDF/Image translator with OCR
uv run pdf_translator
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

Apache License 2.0 — Commercial use, modification, and distribution permitted with attribution.

## Disclaimer

For research purposes only. Not intended for medical diagnosis or treatment decisions.

---

**Author**: Phạm Nguyễn Ngọc Bảo | [Facebook](https://www.facebook.com/bao.phamnguyenngoc.5/)
