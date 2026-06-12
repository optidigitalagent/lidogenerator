# Временный скрипт: извлечь текст из docx без python-docx
import zipfile, re, html, sys

with zipfile.ZipFile(r"C:\Users\Admin\lidogenerator\lead_hunter_free_ukraine.docx") as z:
    xml = z.read("word/document.xml").decode("utf-8")

xml = re.sub(r"<w:p\b[^>]*>", "\n", xml)
xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
xml = re.sub(r"<w:br[^>]*/>", "\n", xml)
text = re.sub(r"<[^>]+>", "", xml)
text = html.unescape(text)
sys.stdout.reconfigure(encoding="utf-8")
print(text)
