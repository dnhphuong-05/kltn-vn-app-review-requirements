"""Word export of thesis tables and figures (python-docx), following 09_thesis/guidelines/yeu_cau_tom_tat.md and the
table look of the thesis draft: Times New Roman, 'Table Grid' borders, header row shaded FCE4D6, highlighted cells
F4B183, rows never split across pages, header row repeated.

Captions: "Bảng {x.y}: Tên. (Nguồn: …)" ABOVE a table, "Hình {x.y}: Tên. (Nguồn: …)" BELOW a figure, italic, centred.
The number is left as a yellow-highlighted {x.y} placeholder (the thesis draft's convention for unfinished parts), since
it depends on the chapter the item ends up in.
"""
import re

import pandas as pd

FONT, SIZE = 'Times New Roman', 13
HEADER_FILL, HL_FILL = 'FCE4D6', 'F4B183'
_NUMERIC = re.compile(r'[\s\d.,±%+\-−–/$()≥≤~x]+')


class Report:
    def __init__(self, title=None):
        from docx import Document
        from docx.shared import Cm, Pt
        self.doc = Document()
        s = self.doc.sections[0]
        s.page_height, s.page_width = Cm(29.7), Cm(21)
        s.top_margin, s.bottom_margin, s.left_margin, s.right_margin = Cm(2), Cm(2), Cm(3), Cm(2)
        normal = self.doc.styles['Normal']
        normal.font.name, normal.font.size = FONT, Pt(SIZE)
        normal.element.rPr.rFonts.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia', FONT)
        self.n_tables = self.n_figures = 0
        if title:
            p = self.doc.add_paragraph()
            self._runs(p, title, bold=True)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _shade(cell, fill):
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        tcpr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), fill)
        tcpr.append(shd)

    @staticmethod
    def _runs(p, text, bold=False, italic=False, size=SIZE):
        """Text with '{...}' parts highlighted yellow (placeholders)."""
        from docx.enum.text import WD_COLOR_INDEX
        from docx.shared import Pt
        for part in re.split(r'(\{[^{}]*\})', str(text)):
            if not part:
                continue
            r = p.add_run(part)
            r.font.name, r.font.size, r.bold, r.italic = FONT, Pt(size), bold, italic
            if part.startswith('{') and part.endswith('}'):
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW

    def _caption(self, text, keep_with_next):
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Pt
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(6), Pt(6)
        p.paragraph_format.keep_with_next = keep_with_next
        self._runs(p, text, italic=True)

    def heading(self, text):
        from docx.shared import Pt
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(12), Pt(6)
        p.paragraph_format.keep_with_next = True
        self._runs(p, text, bold=True)

    def note(self, text):
        from docx.shared import Pt
        p = self.doc.add_paragraph()
        p.paragraph_format.space_after = Pt(9)
        self._runs(p, text, italic=True, size=11)

    # ---------------------------------------------------------------- table
    def table(self, df, title, source='Tác giả', note=None, highlight=(), widths_cm=None, font_size=11):
        """df: already formatted strings (see fmt_frame). highlight: {(row, column name)} of body cells to shade."""
        from docx.enum.table import WD_TABLE_ALIGNMENT
        from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.shared import Cm, Pt
        self.n_tables += 1
        self._caption(f'Bảng {{x.y}}: {title}. (Nguồn: {source})', keep_with_next=True)
        cols = list(df.columns)
        t = self.doc.add_table(rows=1 + len(df), cols=len(cols))
        t.style = 'Table Grid'
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        widths = widths_cm or [16.0 / len(cols)] * len(cols)
        hl = {(r, cols.index(c)) for r, c in highlight}
        rows = [cols] + df.astype(str).values.tolist()
        for ri, row in enumerate(rows):
            tr = t.rows[ri]
            trpr = tr._tr.get_or_add_trPr()
            trpr.append(OxmlElement('w:cantSplit'))
            if ri == 0:
                th = OxmlElement('w:tblHeader')
                th.set(qn('w:val'), 'true')
                trpr.append(th)
            for ci, val in enumerate(row):
                cell = tr.cells[ci]
                cell.width = Cm(widths[ci])
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                p.paragraph_format.keep_with_next = ri < len(rows) - 1
                val = '' if val in ('nan', 'None') else val
                is_num = bool(_NUMERIC.fullmatch(val)) and val.strip() != ''
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (ri == 0 or is_num) else WD_ALIGN_PARAGRAPH.LEFT
                self._runs(p, val, bold=(ri == 0), size=font_size)
                if ri == 0:
                    self._shade(cell, HEADER_FILL)
                elif (ri - 1, ci) in hl:
                    self._shade(cell, HL_FILL)
        if note:
            self.note(note)
        else:
            self.doc.add_paragraph().paragraph_format.space_after = Pt(0)

    # ---------------------------------------------------------------- figure
    def figure(self, path, title, source='Tác giả', width_cm=16.0):
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Cm
        self.n_figures += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(path, width=Cm(width_cm))
        self._caption(f'Hình {{x.y}}: {title}. (Nguồn: {source})', keep_with_next=False)

    def save(self, path):
        """Save; if the file is open in Word (locked), save next to it as '<name>_moi.docx' instead of failing."""
        try:
            self.doc.save(path)
        except PermissionError:
            root, ext = path.rsplit('.', 1)
            path = f'{root}_moi.{ext}'
            self.doc.save(path)
            print(f'File gốc đang mở trong Word — đã lưu bản mới thành {path}')
        return path


# ---------------------------------------------------------------- formatting a DataFrame into display strings
def _vn(x, decimals):
    from src.reporting.style import vn
    return vn(x, decimals)


def fmt_frame(df, decimals=None, percent=(), integer=(), pm=(), keep=()):
    """Strings for display (Vietnamese number format). decimals: {col: n} (default 3 for floats);
    percent: shares 0–1 → '12,3%'; pm: 'a ± b' strings with '.' decimals → ','; keep: columns left as they are."""
    decimals = decimals or {}
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c]
        if c in keep:
            out[c] = s.astype(str)
        elif c in pm:
            out[c] = s.astype(str).str.replace('.', ',', regex=False)
        elif c in percent:
            out[c] = s.map(lambda v: '' if pd.isna(v) else _vn(v * 100, decimals.get(c, 1)) + '%')
        elif c in integer or pd.api.types.is_integer_dtype(s):
            out[c] = s.map(lambda v: '' if pd.isna(v) else _vn(float(v), 0))
        elif pd.api.types.is_bool_dtype(s):
            out[c] = s.map({True: 'Có', False: 'Không'})
        elif pd.api.types.is_float_dtype(s):
            out[c] = s.map(lambda v, d=decimals.get(c, 3): '' if pd.isna(v) else _vn(v, d))
        else:
            out[c] = s.fillna('').astype(str)
    return out
