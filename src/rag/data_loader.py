from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pymupdf import open 

def load_pdfs(data_dir: str) -> List[Any]:
    """
    Loads the ISRO pdf data"""

    data_path = Path(data_dir).resolve()
    print(f"Data Path: {data_path}")
    documents = []

    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF Files: {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"Loading PDF: {pdf_file}")
        try:
            doc = open(str(pdf_file))
            for page_num, page in enumerate(doc):
                text = page.get_text()
                # Convert to LangChain Document
                document = Document(
                    page_content=text,
                    metadata={
                        'source': str(pdf_file),
                        'page': page_num
                    }
                )
                documents.append(document)
            doc.close()
        except Exception as e:
            print(f"Failed to load PDF {pdf_file}:{e}")

    return documents
