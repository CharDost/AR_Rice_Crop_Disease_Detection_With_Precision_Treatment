import zipfile
import xml.etree.ElementTree as ET

def extract_text_from_docx(docx_path):
    try:
        with zipfile.ZipFile(docx_path, 'r') as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            
            # The namespace for word XML
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            # Extract paragraphs
            paragraphs = []
            for p in tree.iterfind('.//w:p', ns):
                texts = []
                for t in p.iterfind('.//w:t', ns):
                    if t.text:
                        texts.append(t.text)
                if texts:
                    paragraphs.append(''.join(texts))
            
            return '\n'.join(paragraphs)
    except Exception as e:
        return str(e)

with open(r"d:\Code\extracted_report.txt", "w", encoding="utf-8") as f:
    f.write(extract_text_from_docx(r"d:\Code\Project Report Content.docx"))
