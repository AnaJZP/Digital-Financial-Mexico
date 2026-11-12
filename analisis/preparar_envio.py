"""
Prepare the Springer submission package from the manuscript exported to Word.

Input : envio_springer/Manuscript_Changing_Stage_Not_Tier.docx (Word export of the
        "Manuscript" tab of the Claude Doc)
Output: envio_springer/
          Manuscript_Changing_Stage_Not_Tier.docx   (LaTeX blocks -> native Word equations,
                                                     Times New Roman, 1.5 spacing)
          figures_tiff/Fig1.tif ... Fig11.tif       (600 dpi, LZW)
          figures_pdf/Fig1.pdf  ... Fig11.pdf       (vector, embedded fonts)
          Alt_text_figures.xlsx                     (Springer: alt text per figure)

Requires latex2mathml (pip) and Microsoft Word's mathml2omml.xsl.
Usage: PYTHONPATH=<dir with latex2mathml> python3 analisis/preparar_envio.py
"""
import copy
import re
import shutil
from pathlib import Path

import latex2mathml.converter as l2m
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt
from lxml import etree
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "envio_springer"
DOCX = OUT / "Manuscript_Changing_Stage_Not_Tier.docx"
FIG = ROOT / "analisis" / "figuras"
XSL = Path("/Applications/Microsoft Word.app/Contents/Resources/mathml2omml.xsl")

# Equations, in order of appearance. Each LaTeX code block of the doc maps to one or more.
EQUATIONS = {
    "z_{itk}": [
        r"z_{itk} = \frac{x_{itk} - \bar{x}_k}{s_k}",
        r"\min_{C_1,\ldots,C_K} \sum_{j=1}^{K} \sum_{(i,t) \in C_j} \| \mathbf{z}_{it} - \mathbf{\mu}_j \|^2",
    ],
    "T = ": [
        r"T = \sum_{i} p_i \frac{y_i}{\mu} \ln \frac{y_i}{\mu} = T_{B} + T_{W}",
        r"T_{B} = \sum_{r} p_r \frac{y_r}{\mu} \ln \frac{y_r}{\mu}, \qquad T_{W} = \sum_{r} p_r \frac{y_r}{\mu} T_r",
    ],
    r"\ln Y": [
        r"\ln Y_{it} = \alpha_i + \gamma_t + \sum_{k \neq 2021Q4} \beta_k \mathbf{1}[t = k] I_i + \varepsilon_{it}",
    ],
}

ALT = {
    1: "Nine small line charts, one per access channel, showing the median Mexican state with the "
       "interquartile and min-max ranges across states from 2018 to 2026. Mobile-banking contracts and "
       "development-bank branches rise steeply; commercial and Sofipo branches decline; agents, ATMs and "
       "POS terminals rise moderately.",
    2: "Nine stacked area charts of the population-weighted Theil index by channel, split into "
       "between-region and within-region components, 2018-2026. Inequality falls for ATMs, agents and "
       "commercial branches and rises for development-bank branches and mobile banking, where almost all "
       "inequality lies between regions.",
    3: "Four line charts of cluster-validity criteria for K from 2 to 10: within-cluster sum of squares, "
       "silhouette, Davies-Bouldin index and bootstrap stability. K = 7 is highlighted; it maximises the "
       "silhouette and keeps high stability.",
    4: "Heatmap of the seven K-means centroids (rows, from entrenched laggards to frontier innovators) by "
       "nine access indicators (columns), with the value per 10,000 adults printed in each cell.",
    5: "Diverging heatmap of standardised centroid values (z-scores) for the seven adoption stages and "
       "nine indicators.",
    6: "Grid of 32 states (rows, grouped by region) by 33 quarters (columns) shaded by adoption stage. "
       "Most states keep the same stage throughout; southern and central states change from entrenched "
       "laggards to laggards in transition between 2022 and 2025.",
    7: "Stacked area chart of the share of Mexico's adult population living in each adoption stage, "
       "2018-2026. The bottom band of entrenched laggards (about 52%) is replaced after 2022 by laggards in "
       "transition (about 53% in 2026).",
    8: "Three heatmaps of quarter-to-quarter transition probabilities between the seven stages for the "
       "pre-COVID, COVID and post-COVID periods. Almost all probability lies on the diagonal; off-diagonal "
       "moves appear only after 2022.",
    9: "Three panels: the sorted 10-nearest-neighbour distance curve with the chosen epsilon, and two "
       "scatter plots of state-quarters on the first two principal components shaded by K-means stage and "
       "by DBSCAN group. DBSCAN separates a dense core, a northern border belt, a metropolitan-tourist "
       "frontier and Mexico City outliers.",
    10: "Scatter plot of all state-quarters on the first two principal components shaded by stage, with "
        "lines tracing the 2018-2026 paths of seven states. Southern states move vertically toward public "
        "branches and mobile banking; Mexico City moves far to the lower right.",
    11: "Four event-study plots of coefficients with 95% confidence bands by quarter relative to 2021Q4, "
        "for mobile-banking contracts, banking agents, ATMs and commercial branches. Pre-period "
        "coefficients drift away from zero; only ATMs show a clear rise after 2022.",
}
FIG_FILES = ["fig01_evolucion_canales", "fig02_desigualdad_theil", "fig03_seleccion_k",
             "fig04_heatmap_centroides", "fig05_perfiles_z", "fig06_trayectorias_estado",
             "fig07_poblacion_por_etapa", "fig08_matrices_transicion", "fig09_dbscan",
             "fig10_mapa_pca_trayectorias", "fig11_event_study_bienestar"]


def omml(latex, xslt):
    mml = etree.fromstring(l2m.convert(latex).encode())
    return xslt(mml).getroot()


def equation_paragraph(template_par, latex, number, xslt):
    """New paragraph (cloned from a body paragraph) holding a centred equation and its number."""
    p = copy.deepcopy(template_par)
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)
    p.append(omml(latex, xslt))
    r = etree.SubElement(p, qn("w:r"))
    t = etree.SubElement(r, qn("w:t"))
    t.text = f"  ({number})"
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p


NUMERIC = re.compile(r"^[\s\d.,%<>=+\-−–()]*(of\s+\d+)?[\s\d.,%]*$")


def _border(parent, edge, sz):
    el = etree.SubElement(parent, qn(f"w:{edge}"))
    el.set(qn("w:val"), "single" if sz else "nil")
    if sz:
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")


def apa_table(table, font_pt=10):
    """APA 7 look: horizontal rules only (top, under header, bottom), no shading, no bold,
    smaller single-spaced text, text columns left-aligned and numeric columns centred."""
    tbl = table._tbl
    tblPr = tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    borders = etree.SubElement(tblPr, qn("w:tblBorders"))
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        _border(borders, edge, 0)
    # full text width
    w = tblPr.find(qn("w:tblW"))
    w.set(qn("w:type"), "pct")
    w.set(qn("w:w"), "5000")
    # tblBorders must precede tblLayout / tblLook in schema order
    for tag in ("tblLayout", "tblCellMar", "tblLook"):
        el = tblPr.find(qn(f"w:{tag}"))
        if el is not None:
            tblPr.remove(el)
            tblPr.append(el)

    rows = table.rows
    ncol = len(table.columns)
    numeric_col = []
    for c in range(ncol):
        body = [rows[r].cells[c].text.strip() for r in range(1, len(rows))]
        body = [b for b in body if b]
        numeric_col.append(bool(body) and sum(bool(NUMERIC.match(b)) for b in body) / len(body) >= 0.7)

    for r_i, row in enumerate(rows):
        for c_i, cell in enumerate(row.cells):
            tcPr = cell._tc.get_or_add_tcPr()
            for shd in tcPr.findall(qn("w:shd")):
                tcPr.remove(shd)
            for old in tcPr.findall(qn("w:tcBorders")):
                tcPr.remove(old)
            tcb = etree.Element(qn("w:tcBorders"))
            if r_i == 0:
                _border(tcb, "top", 8)        # 1 pt rule above the header
                _border(tcb, "bottom", 4)     # 0.5 pt rule under the header
            if r_i == len(rows) - 1:
                _border(tcb, "bottom", 8)     # 1 pt rule closing the table
            # tcBorders goes right after tcW in tcPr
            tcW = tcPr.find(qn("w:tcW"))
            (tcW.addnext(tcb) if tcW is not None else tcPr.insert(0, tcb))
            for p in cell.paragraphs:
                pf = p.paragraph_format
                pf.space_before = Pt(2)
                pf.space_after = Pt(2)
                pf.line_spacing = 1.0
                if r_i == 0 or numeric_col[c_i] and c_i > 0:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for run in p.runs:
                    run.font.size = Pt(font_pt)
                    run.font.bold = False


def main():
    doc = Document(DOCX)
    xslt = etree.XSLT(etree.parse(str(XSL)))
    body_template = next(p for p in doc.paragraphs if p.style.name == "Normal" and p.text.strip())

    # 1. LaTeX code blocks -> numbered native equations
    n = 0
    for p in list(doc.paragraphs):
        if p.style.name != "Code":
            continue
        key = next((k for k in EQUATIONS if p.text.strip().startswith(k)), None)
        if key is None:
            raise SystemExit(f"Unmapped code block: {p.text[:60]}")
        anchor = p._p
        for latex in EQUATIONS[key]:
            n += 1
            new = equation_paragraph(body_template._p, latex, n, xslt)
            anchor.addnext(new)
            anchor = new
        p._p.getparent().remove(p._p)
    for p in doc.paragraphs:
        if p._p.find(qn("m:oMath")) is not None:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    print(f"Equations converted: {n}")

    # 1b. APA-style tables and their notes ("Note." paragraphs right below a table)
    for t in doc.tables:
        apa_table(t)
        nxt = t._tbl.getnext()
        while nxt is not None and nxt.tag == qn("w:p"):
            from docx.text.paragraph import Paragraph
            par = Paragraph(nxt, doc)
            if not par.text.strip().startswith("Note."):
                break
            par.paragraph_format.line_spacing = 1.0
            par.paragraph_format.space_before = Pt(4)
            for run in par.runs:
                run.font.size = Pt(10)
            nxt = nxt.getnext()
    # table label ("Table n", bold) and title (italic) lines: keep with the table, no extra gap
    for p in doc.paragraphs:
        if re.fullmatch(r"Table \d+", p.text.strip()):
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.space_after = Pt(0)
            title = p._p.getnext()
            if title is not None and title.tag == qn("w:p"):
                from docx.text.paragraph import Paragraph
                tp = Paragraph(title, doc)
                tp.paragraph_format.keep_with_next = True
                tp.paragraph_format.space_after = Pt(4)
    print(f"Tables formatted (APA): {len(doc.tables)}")

    # 2. Typography for review: Times New Roman 12 pt, 1.5 line spacing
    for name in ("Normal", "Heading 1", "Heading 2", "Heading 3"):
        st = doc.styles[name]
        st.font.name = "Times New Roman"
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    doc.styles["Normal"].font.size = Pt(12)
    doc.styles["Normal"].paragraph_format.line_spacing = 1.5

    # 2b. The export writes some style pPr children out of schema order (pBdr/shd after
    #     spacing/ind); Word tolerates it but strict validators do not. Reorder them.
    order = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl",
             "numPr", "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens",
             "kinsoku", "wordWrap", "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN",
             "bidi", "adjustRightInd", "snapToGrid", "spacing", "ind", "contextualSpacing",
             "mirrorIndents", "suppressOverlap", "jc", "textDirection", "textAlignment",
             "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr", "sectPr", "pPrChange"]
    rank = {qn(f"w:{t}"): i for i, t in enumerate(order)}
    for ppr in doc.styles.element.iter(qn("w:pPr")):
        kids = list(ppr)
        for k in kids:
            ppr.remove(k)
        for k in sorted(kids, key=lambda e: rank.get(e.tag, len(order))):
            ppr.append(k)
    doc.save(DOCX)

    # 3. Figures as separate files, numbered as in the text
    for sub in ("figures_tiff", "figures_pdf"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(FIG_FILES, start=1):
        shutil.copy(FIG / "tiff" / f"{stem}.tif", OUT / "figures_tiff" / f"Fig{i}.tif")
        shutil.copy(FIG / "pdf" / f"{stem}.pdf", OUT / "figures_pdf" / f"Fig{i}.pdf")

    # 4. Alt-text workbook (captions read from the manuscript)
    captions = {}
    for p in doc.paragraphs:
        m = re.match(r"Fig\. (\d+)\s+(.*)", p.text.strip())
        if m:
            captions[int(m.group(1))] = m.group(2)
    wb = Workbook()
    ws = wb.active
    ws.title = "Alt text"
    ws.append(["Figure", "File", "Caption", "Alt text"])
    for i in range(1, 12):
        ws.append([f"Fig. {i}", f"Fig{i}.tif / Fig{i}.pdf", captions.get(i, ""), ALT[i]])
    for col, w in zip("ABCD", (9, 22, 60, 90)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows():
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for c in ws[1]:
        c.font = Font(bold=True)
    wb.save(OUT / "Alt_text_figures.xlsx")
    print(f"Captions found: {len(captions)}; package written to {OUT}")


if __name__ == "__main__":
    main()
