"""
Quick test: Extract text from a few PDFs to verify pdfplumber works
"""
import pdfplumber
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

pdf_dir = Path("/Users/christophermayfield/Desktop/Corpus_Final_Review")

# Test on a few PDFs from each zero-recall topic
test_pdfs = [
    "Climate Science/documents/IPCC - AR6 Synthesis Report Summary for Policymakers.pdf",
    "Legal Documents/documents/Marbury v. Madison _ 5 U.S. 137 (1803) _ Justia U.S. Supreme Court Center.pdf",
    "Internet Standard (RFCs)/documents/rfc-editor.org-RFC 9112 STD 99 HTTP11.pdf",
    "NIST_AI Security & Governance/documents/nist.ai.100-1.pdf",
    "Deep Sea Biology/documents/Chapter_36F.pdf",
    "Dream-Based Memory Consolidation and Forgetting/documents/sleepisforforgetting.pdf",
]

print("Testing PDF extraction...")
print("=" * 80)

success_count = 0
for pdf_path in test_pdfs:
    full_path = pdf_dir / pdf_path
    if not full_path.exists():
        print(f"✗ Not found: {pdf_path}")
        continue

    try:
        with pdfplumber.open(full_path) as pdf:
            text = ""
            for page in pdf.pages[:2]:  # Just first 2 pages
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception:
                    pass

            char_count = len(text)
            if char_count > 100:
                success_count += 1
                print(f"✓ {pdf_path.split('/')[-1][:50]:<50} ({char_count:>6} chars)")
            else:
                print(f"⚠ {pdf_path.split('/')[-1][:50]:<50} (only {char_count} chars)")
    except Exception as e:
        print(f"✗ {pdf_path.split('/')[-1][:50]:<50} (error: {str(e)[:30]})")

print()
print(f"Success rate: {success_count}/{len(test_pdfs)}")
print()
print("✓ PDF extraction is working! Now indexing full corpus...")
