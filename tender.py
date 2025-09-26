import os
import streamlit as st
import pdfplumber
import re
import faiss
import numpy as np
import requests
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import io
from reportlab.lib.units import cm
import asyncio
import aiohttp

# ------------------- CONFIG --------------------
API_KEY = "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"  # Replace with your OpenRouter key
MODEL = "x-ai/grok-4-fast:free"

CHUNK_SIZE, OVERLAP = 800, 150
HF_EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="gpu" if os.getenv("USE_GPU") == "1" else "cpu")

# -------- PDF utilities --------
def extract_text_from_pdf(file) -> str:
    text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = max(0, end - overlap)
        if end >= len(text):
            break
    return chunks

# -------- Embedding & FAISS --------
def embed_texts(texts):
    vectors = HF_EMBED_MODEL.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    )
    return np.array(vectors, dtype=np.float32)

class VectorIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas = []

    def add(self, vectors, metadatas):
        self.index.add(vectors)
        self.metadatas.extend(metadatas)

    def search(self, query_vec, top_k=5):
        D, I = self.index.search(query_vec, top_k)
        results = []
        for i, idx in enumerate(I[0]):
            if idx < 0:
                continue
            score = float(D[0][i])
            results.append((self.metadatas[idx], score))
        return results

# -------- Field Extraction --------
def extract_fields(text: str):
    fields = {}
    m = re.search(r'(\d+)\s+years', text, re.I)
    fields["experience"] = int(m.group(1)) if m else None
    if "valid" in text.lower() and "license" in text.lower():
        fields["license"] = "valid"
    elif "pending" in text.lower() and "license" in text.lower():
        fields["license"] = "pending"
    m = re.search(r'\$([\d,]+[kK]?)', text)
    fields["revenue"] = m.group(1) if m else None
    m = re.search(r'(\d+)\s+(months|month|years|year)', text, re.I)
    if m:
        num = int(m.group(1))
        fields["timeline_months"] = num * (12 if "year" in m.group(2).lower() else 1)
    m = re.search(r'\$([\d,]+)', text)
    fields["cost"] = m.group(1) if m else None
    return fields

# -------- Scoring (simple hybrid) --------
def score_proposal(tender_text, proposal_text, sim_score, fields):
    score, reasons = 0, []

    experience = fields.get("experience") or 0
    if experience >= 5:
        score += 20
        reasons.append("Experience ≥5 yrs ✅")
    else:
        reasons.append("Experience <5 yrs ❌")

    if fields.get("license") == "valid":
        score += 15
        reasons.append("Valid license ✅")
    else:
        reasons.append("License missing/invalid ❌")

    score += int(sim_score * 25)
    reasons.append(f"Technical similarity {sim_score:.2f}")

    timeline = fields.get("timeline_months") or 999
    if timeline <= 6:
        score += 10
        reasons.append("Timeline within 6 months ✅")
    else:
        reasons.append("Timeline too long ❌")

    if fields.get("revenue"):
        score += 10
        reasons.append("Financial info provided ✅")

    return min(100, score), reasons

# -------- OpenRouter LLM Evaluation (async) --------
async def openrouter_eval(tender_text, proposal_text, prelim_score, company_name):
    prompt = f"""
Tender requirements:
{tender_text[:600]}

Proposal:
{proposal_text[:600]}

Preliminary score: {prelim_score}

As procurement expert, give:
- Final SCORE (0-100)
- 2–3 sentence REASON
- Missing key requirements
- JSON with extracted fields (experience, license, revenue, timeline, cost)
"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}]
            }
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error {resp.status}: {await resp.text()}"

# -------- PDF Generation --------
def generate_pdf(results):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30
    )
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("Tender–Proposal Comparison Report", styles['Title']))
    elements.append(Spacer(1, 20))

    # Table header
    table_data = [["Company", "Score", "Experience", "License", "Revenue", "Timeline (months)", "Reasons"]]

    for r in results:
        reasons_text = "<br/>".join(r.get("reasons", []))
        score_val = r.get("score", r.get("prelim_score", 0))
        row = [
            Paragraph(str(r.get("company", "")), styles['Normal']),
            str(score_val),
            str(r.get("fields", {}).get("experience", "")),
            str(r.get("fields", {}).get("license", "")),
            str(r.get("fields", {}).get("revenue", "")),
            str(r.get("fields", {}).get("timeline_months", "")),
            Paragraph(reasons_text, styles['Normal'])
        ]
        table_data.append(row)

    # Column widths
    col_widths = [3.5*cm, 2*cm, 2.5*cm, 2.5*cm, 3*cm, 3*cm, 7*cm]

    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.gray),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-2,-1), 'CENTER'),
        ('ALIGN', (-1,1), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black)
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    top_company = max(results, key=lambda r: r.get("score", r.get("prelim_score", 0)))
    top_score = top_company.get("score", top_company.get("prelim_score", 0))
    suggestion_text = f"""
    <b>Suggested Proposal to Accept:</b> {top_company.get('company','')}<br/>
    <b>Score:</b> {top_score}<br/>
    <b>Reason:</b> {'; '.join(top_company.get('reasons', []))}
    """
    elements.append(Paragraph(suggestion_text, styles['Heading2']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------------- STREAMLIT APP ----------------
st.title("📑 Tender–Proposal Matcher")

# File uploads
tender_file = st.file_uploader("Upload Tender PDF", type=["pdf"])
proposal_files = st.file_uploader("Upload Proposal PDFs (max 10)", type=["pdf"], accept_multiple_files=True)

# Automatically run analysis once files are uploaded
if tender_file and proposal_files:
    with st.spinner("Processing documents..."):
        tender_text = extract_text_from_pdf(tender_file)
        tender_chunks = chunk_text(tender_text)
        tender_embs = embed_texts(tender_chunks)
        tender_vec = tender_embs.mean(axis=0, keepdims=True)

        idx = VectorIndex(tender_embs.shape[1])
        results = []

        async def process_files():
            tasks = []
            for file in proposal_files:
                p_text = extract_text_from_pdf(file)
                p_chunks = chunk_text(p_text)
                p_embs = embed_texts(p_chunks)
                metas = [{"file": file.name, "chunk": i, "text": c} for i, c in enumerate(p_chunks)]
                idx.add(p_embs, metas)
                hits = idx.search(tender_vec, top_k=10)
                hits = [h for h in hits if h[0]["file"] == file.name]
                sim = np.mean([s for _, s in hits]) if hits else 0.0
                fields = extract_fields(p_text)
                prelim, reasons = score_proposal(tender_text, p_text, sim, fields)
                task = openrouter_eval(tender_text, p_text, prelim, file.name)
                tasks.append((file.name, prelim, reasons, fields, task))
            return tasks

        tasks = asyncio.run(process_files())
        for company, prelim, reasons, fields, coro in tasks:
            gem_eval = asyncio.run(coro)
            results.append({
                "company": company,
                "prelim_score": prelim,
                "reasons": reasons,
                "fields": fields,
                "gemini_eval": gem_eval
            })

        ranked = sorted(results, key=lambda r: r["prelim_score"], reverse=True)

    st.subheader("🏆 Ranked Companies")
    for r in ranked:
        with st.expander(f"{r['company']} — Score {r['prelim_score']}"):
            st.write("**Reasons:**")
            for reason in r["reasons"]:
                st.write("- " + reason)
            st.json(r["fields"])
            st.write(r["gemini_eval"])

    pdf_buffer = generate_pdf(ranked)
    st.download_button(
        label="📥 Download Comparison PDF",
        data=pdf_buffer,
        file_name="Tender_Comparison_Report.pdf",
        mime="application/pdf"
    )
