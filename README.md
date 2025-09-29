# 📑 Tender–Proposal Matcher

**Tender–Proposal Matcher** is an intelligent system to evaluate company proposals against tender requirements.  
It automates extraction, comparison, scoring, and generates detailed PDF evaluation reports.

---

## 🏗️ Workflow Overview

```text
┌──────────────────────────┐
│ Tender Document (Org)    │
└───────────┬──────────────┘
            │
      Parse & Structure (LangChain)
            │
 ┌──────────▼───────────┐
 │ Tender Requirements  │
 │ JSON Schema          │
 └──────────┬───────────┘
            │
┌───────────▼────────────────────────┐
│ Applicants' Tender Docs             │
└───────────┬───────────┬────────────┘
            │           │
    (OCR for scanned)  (Direct PDF text)   
      Tesseract OCR      PyMuPDF/pdfplumber
            │           │
            └───────┬───┘
                    │
            Parse & Chunk (LangChain)
                    │
           ┌────────▼─────────┐
           │ Applicant Data   │
           │ JSON             │
           └────────┬─────────┘
                    │
           ┌────────▼─────────┐
           │ Embeddings       │
           │ (HuggingFace/OLLAMA) │
           └────────┬─────────┘
                    │
           ┌────────▼─────────┐
           │ Vector DB        │
           │ (FAISS/Chroma)   │
           │ Store {text, metadata} │
           └────────┬─────────┘
                    │
        ┌───────────▼────────────┐
        │ Similarity Search +    │
        │ Rule Filter            │
        │ (requirements vs       │
        │ applicants)            │
        └───────────┬────────────┘
                    │
           ┌────────▼──────────┐
           │ Shortlisted       │
           │ Applicants        │
           └────────┬──────────┘
                    │
           ┌────────▼──────────┐
           │ LLM Evaluation    │
           │ - Compare with    │
           │   requirements    │
           │ - Score & Remarks │
           └────────┬──────────┘
                    │
           ┌────────▼──────────┐
           │ PDF Report        │
           │ (Jinja2 + LaTeX)  │
           │ Tabular Compliance │
           └───────────────────┘

## 🚀 Quick Start

Follow these steps to get the **Tender–Proposal Matcher** running locally:

---

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Kishorsahoo934/Tender-Matcher.git
cd Tender-Matcher

