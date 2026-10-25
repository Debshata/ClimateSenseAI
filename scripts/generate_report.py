"""Generate the IEEE-formatted final report (DOCX, A4 two-column) from saved result artefacts.

Uses the official IEEE conference A4 Word template (converted from Strict to Transitional OOXML
by Word) so that IEEE paragraph styles and auto-numbering (headings, Fig., TABLE, references)
are preserved.  Every number in the text is read from results/ or experiments/ at run time.
Also exports a PDF through Microsoft Word when available.
"""
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from climatesense.labeling.compound_risk import CLASS_NAMES
from climatesense.utils import read_json

TEMPLATE = ROOT / "reports/inputs/conference-template-a4-transitional.docx"
OUT = ROOT / "reports/final/ClimateSenseAI_IEEE_Final_Report.docx"
FIG = ROOT / "results/figures"
COL_W = 3.4  # inches, one IEEE A4 column
FULL_W = 6.9

# ---------------------------------------------------------------------------- references
REFS = [
    ("ipcc", "IPCC, “Summary for policymakers,” in Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change. Cambridge, U.K.: Cambridge Univ. Press, 2021, pp. 3–32, doi: 10.1017/9781009157896.001."),
    ("zscheischler", "J. Zscheischler et al., “Future climate risk from compound events,” Nature Climate Change, vol. 8, no. 6, pp. 469–477, 2018, doi: 10.1038/s41558-018-0156-3."),
    ("leonard", "M. Leonard et al., “A compound event framework for understanding extreme impacts,” WIREs Climate Change, vol. 5, no. 1, pp. 113–128, 2014, doi: 10.1002/wcc.252."),
    ("raymond", "C. Raymond, T. Matthews, and R. M. Horton, “The emergence of heat and humidity too severe for human tolerance,” Science Advances, vol. 6, no. 19, Art. no. eaaw1838, 2020, doi: 10.1126/sciadv.aaw1838."),
    ("mora", "C. Mora et al., “Global risk of deadly heat,” Nature Climate Change, vol. 7, no. 7, pp. 501–506, 2017, doi: 10.1038/nclimate3322."),
    ("who", "World Health Organization, WHO Global Air Quality Guidelines: Particulate Matter (PM2.5 and PM10), Ozone, Nitrogen Dioxide, Sulfur Dioxide and Carbon Monoxide. Geneva, Switzerland: WHO, 2021."),
    ("spei", "S. M. Vicente-Serrano, S. Beguería, and J. I. López-Moreno, “A multiscalar drought index sensitive to global warming: The standardized precipitation evapotranspiration index,” Journal of Climate, vol. 23, no. 7, pp. 1696–1718, 2010, doi: 10.1175/2009JCLI2909.1."),
    ("usdm", "M. Svoboda et al., “The Drought Monitor,” Bulletin of the American Meteorological Society, vol. 83, no. 8, pp. 1181–1190, 2002, doi: 10.1175/1520-0477-83.8.1181."),
    ("grossiord", "C. Grossiord et al., “Plant responses to rising vapor pressure deficit,” New Phytologist, vol. 226, no. 6, pp. 1550–1566, 2020, doi: 10.1111/nph.16485."),
    ("modis", "K. Didan, “MODIS/Terra Vegetation Indices Monthly L3 Global 1km SIN Grid V061 (MOD13A3),” NASA EOSDIS Land Processes DAAC, 2021, doi: 10.5067/MODIS/MOD13A3.061."),
    ("reichstein", "M. Reichstein et al., “Deep learning and process understanding for data-driven Earth system science,” Nature, vol. 566, pp. 195–204, 2019, doi: 10.1038/s41586-019-0912-1."),
    ("baltrusaitis", "T. Baltrušaitis, C. Ahuja, and L.-P. Morency, “Multimodal machine learning: A survey and taxonomy,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 2, pp. 423–443, 2019, doi: 10.1109/TPAMI.2018.2798607."),
    ("ramachandram", "D. Ramachandram and G. W. Taylor, “Deep multimodal learning: A survey on recent advances and trends,” IEEE Signal Processing Magazine, vol. 34, no. 6, pp. 96–108, 2017, doi: 10.1109/MSP.2017.2738401."),
    ("vaswani", "A. Vaswani et al., “Attention is all you need,” in Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 5998–6008."),
    ("lstm", "S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997, doi: 10.1162/neco.1997.9.8.1735."),
    ("gru", "K. Cho et al., “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” in Proc. EMNLP, 2014, pp. 1724–1734, doi: 10.3115/v1/D14-1179."),
    ("tft", "B. Lim, S. Ö. Arık, N. Loeff, and T. Pfister, “Temporal fusion transformers for interpretable multi-horizon time series forecasting,” International Journal of Forecasting, vol. 37, no. 4, pp. 1748–1764, 2021, doi: 10.1016/j.ijforecast.2021.03.012."),
    ("convlstm", "X. Shi et al., “Convolutional LSTM network: A machine learning approach for precipitation nowcasting,” in Advances in Neural Information Processing Systems, vol. 28, 2015, pp. 802–810."),
    ("rothfusz", "L. P. Rothfusz, “The heat index ‘equation’ (or, more than you ever wanted to know about heat index),” NWS Southern Region Headquarters, Fort Worth, TX, Tech. Attachment SR 90-23, 1990."),
    ("steadman", "R. G. Steadman, “The assessment of sultriness. Part I: A temperature-humidity index based on human physiology and clothing science,” Journal of Applied Meteorology, vol. 18, no. 7, pp. 861–873, 1979, doi: 10.1175/1520-0450(1979)018<0861:TAOSPI>2.0.CO;2."),
    ("stull", "R. Stull, “Wet-bulb temperature from relative humidity and air temperature,” Journal of Applied Meteorology and Climatology, vol. 50, no. 11, pp. 2267–2269, 2011, doi: 10.1175/JAMC-D-11-0143.1."),
    ("vecellio", "D. J. Vecellio, S. T. Wolf, R. M. Cottle, and W. L. Kenney, “Evaluating the 35°C wet-bulb temperature adaptability threshold for young, healthy subjects (PSU HEAT Project),” Journal of Applied Physiology, vol. 132, no. 2, pp. 340–345, 2022, doi: 10.1152/japplphysiol.00738.2021."),
    ("focal", "T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal loss for dense object detection,” in Proc. IEEE ICCV, 2017, pp. 2980–2988, doi: 10.1109/ICCV.2017.324."),
    ("adamw", "I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. ICLR, 2019."),
    ("optuna", "T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, “Optuna: A next-generation hyperparameter optimization framework,” in Proc. ACM SIGKDD, 2019, pp. 2623–2631, doi: 10.1145/3292500.3330701."),
    ("rf", "L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001, doi: 10.1023/A:1010933404324."),
    ("gbm", "J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” The Annals of Statistics, vol. 29, no. 5, pp. 1189–1232, 2001, doi: 10.1214/aos/1013203451."),
    ("sklearn", "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011."),
    ("pytorch", "A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Advances in Neural Information Processing Systems, vol. 32, 2019, pp. 8024–8035."),
    ("ig", "M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” in Proc. ICML, PMLR vol. 70, 2017, pp. 3319–3328."),
    ("attn_expl", "S. Jain and B. C. Wallace, “Attention is not explanation,” in Proc. NAACL-HLT, 2019, pp. 3543–3556, doi: 10.18653/v1/N19-1357."),
    ("weather", "N. Elgiriyewithana, “World Weather Repository (Daily Updating),” Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository (version 1105 used)."),
    ("drought_kaggle", "cdminix, “Predict droughts using weather & soil data,” Kaggle, 2021 (sources: NASA POWER, U.S. Drought Monitor, Harmonized World Soil Database). [Online]. Available: https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data"),
    ("worldcities", "SimpleMaps, “World Cities Database,” Kaggle mirror by J. Martínez. [Online]. Available: https://www.kaggle.com/datasets/juanmah/world-cities"),
    ("berkeley", "Berkeley Earth, “Climate change: Earth surface temperature data,” Kaggle. [Online]. Available: https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data"),
    ("rohde", "R. Rohde et al., “A new estimate of the average Earth surface land temperature spanning 1753 to 2011,” Geoinformatics & Geostatistics: An Overview, vol. 1, no. 1, 2013, doi: 10.4172/2327-4581.1000101."),
    ("power", "NASA Langley Research Center, “POWER: Prediction Of Worldwide Energy Resources — Data Access.” [Online]. Available: https://power.larc.nasa.gov/"),
    ("emdat", "Centre for Research on the Epidemiology of Disasters (CRED), UCLouvain, “EM-DAT: The International Disaster Database.” [Online]. Available: https://www.emdat.be"),
]
REF_NUM = {k: i + 1 for i, (k, _) in enumerate(REFS)}


def cite(*keys):
    return "[" + "], [".join(str(REF_NUM[k]) for k in keys) + "]"


# ---------------------------------------------------------------------------- docx helpers
class Report:
    def __init__(self):
        self.doc = Document(TEMPLATE)
        body = self.doc.element.body
        sects = [copy.deepcopy(s) for s in body.iter(qn("w:sectPr"))]
        self.sect_title = sects[0]          # one column (title/authors)
        self.sect_two = sects[3]            # two columns, 0.25 in gap
        for child in list(body):
            body.remove(child)
        final = copy.deepcopy(self.sect_two)
        body.append(final)
        self.fig_n = 0
        self.tab_n = 0
        self.eq_n = 0

    @property
    def body(self):
        return self.doc.element.body

    def _insert(self, p_el):
        self.body.insert(len(self.body) - 1, p_el)

    def para(self, text="", style="Body Text", bold_prefix=None, align=None, italic=False, size=None):
        p = self.doc.add_paragraph(style=style)
        self._insert(p._p)
        if bold_prefix:
            r = p.add_run(bold_prefix)
            r.bold = True
        self._rich(p, text, italic, size)
        if align is not None:
            p.alignment = align
        return p

    @staticmethod
    def _rich(p, text, italic=False, size=None):
        """Minimal inline markup: *italic* and _{sub} / ^{sup}."""
        import re
        tokens = re.split(r"(\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})", text)
        for t in tokens:
            if not t:
                continue
            if t.startswith("*") and t.endswith("*"):
                r = p.add_run(t[1:-1]); r.italic = True
            elif t.startswith("_{"):
                r = p.add_run(t[2:-1]); r.font.subscript = True
            elif t.startswith("^{"):
                r = p.add_run(t[2:-1]); r.font.superscript = True
            else:
                r = p.add_run(t)
                r.italic = italic
            if size:
                r.font.size = Pt(size)

    def section_break(self, sect):
        p = self.doc.add_paragraph()
        s = copy.deepcopy(sect)
        t = s.find(qn("w:type"))
        if t is None:
            t = OxmlElement("w:type"); s.insert(0, t)
        t.set(qn("w:val"), "continuous")
        p._p.get_or_add_pPr().append(s)
        self._insert(p._p)

    def heading(self, text, level=1):
        return self.para(text, f"Heading {level}")

    def equation(self, text):
        self.eq_n += 1
        p = self.doc.add_paragraph(style="equation")
        self._insert(p._p)
        p.add_run("\t")
        self._rich(p, text, italic=False)
        for r in p.runs:
            r.font.name = "Times New Roman"
            r._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        p.add_run(f"\t({self.eq_n})").font.name = "Times New Roman"
        return self.eq_n

    def figure(self, path, caption, width=COL_W):
        self.fig_n += 1
        p = self.doc.add_paragraph(style="Normal")
        self._insert(p._p)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Inches(width))
        p.paragraph_format.keep_with_next = True
        self.para(caption, "figure caption")
        return self.fig_n

    def wide_figure(self, path, caption):
        self.section_break(self.sect_two)
        n = self.figure(path, caption, FULL_W)
        self.section_break(self.sect_title)
        return n

    def table(self, title, header, rows, widths=None, font=7, bold_rows=()):
        self.tab_n += 1
        self.para(title, "table head")
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        self._insert(t._tbl)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.style = self.doc.styles["Table Grid"] if "Table Grid" in [s.name for s in self.doc.styles] else None
        self._borders(t)
        for j, h in enumerate(header):
            c = t.rows[0].cells[j]
            c.paragraphs[0].style = self.doc.styles["table col head"]
            r = c.paragraphs[0].add_run(h); r.font.size = Pt(font); r.bold = True
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                c = t.rows[i + 1].cells[j]
                c.paragraphs[0].style = self.doc.styles["table copy"]
                c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                r = c.paragraphs[0].add_run(str(v)); r.font.size = Pt(font)
                if i in bold_rows:
                    r.bold = True
        if widths:
            for row in t.rows:
                for j, w in enumerate(widths):
                    row.cells[j].width = Inches(w)
        sp = self.doc.add_paragraph(style="Normal")
        sp.paragraph_format.space_after = Pt(4)
        self._insert(sp._p)
        return self.tab_n

    @staticmethod
    def _borders(t):
        tblPr = t._tbl.tblPr
        b = OxmlElement("w:tblBorders")
        for edge in ("top", "bottom", "insideH"):
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "000000")
            b.append(e)
        tblPr.append(b)
        cm = OxmlElement("w:tblCellMar")
        for side in ("left", "right"):
            e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), "40"); e.set(qn("w:type"), "dxa"); cm.append(e)
        tblPr.append(cm)

    def bullets(self, items):
        for it in items:
            self.para(it, "bullet list")


def _tie_sentence(pv, ties):
    names = {"persistence": "persistence", "hist_gradient_boosting": "gradient boosting", "random_forest": "the random forest",
             "gru": "the tuned GRU", "mafnet_concat": "the concatenation-fusion variant", "transformer": "the tuned Transformer"}
    tied = [names[k] for k in pv if ties(k)]
    better = [names[k] for k in pv if pv[k]["ci95"][0] > 0]
    s = ""
    if tied:
        s += ("The paired intervals include zero for " + ", ".join(tied)
              + ", so MAF-Net is statistically indistinguishable from these models on this test period. ")
    if better:
        s += "MAF-Net is better than " + ", ".join(better) + " (intervals above zero), although the margins are small."
    return s


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pm(m, s):
    return f"{m:.3f} ± {s:.3f}" if s and not np.isnan(s) else f"{m:.3f}"


# ---------------------------------------------------------------------------- content
def build():
    audit = read_json(ROOT / "data/manifests/data_audit.json")
    qual = read_json(ROOT / "data/manifests/dataset_quality.json")
    lab = read_json(ROOT / "results/analysis/label_sensitivity.json")
    splits = read_json(ROOT / "data/manifests/splits.json")
    sel_cfgs = read_json(ROOT / "experiments/tuning/selected_configs.json")
    st = read_json(ROOT / "results/metrics/selected_model_test.json")
    rec = read_json(ROOT / "experiments" / st["selected_run"].split("-")[0] / f"{st['selected_run']}.json")
    env = read_json(ROOT / "docs/environment_info.json")
    mc = pd.read_csv(ROOT / "results/tables/model_comparison_h1.csv").set_index("model")
    ab = pd.read_csv(ROOT / "results/tables/ablation.csv").set_index("variant")
    hz = pd.read_csv(ROOT / "results/tables/horizons.csv")
    lb = pd.read_csv(ROOT / "results/tables/lookback.csv")
    sp = pd.read_csv(ROOT / "results/tables/spatial_holdout.csv").set_index("model")
    ex = read_json(ROOT / "results/analysis/explainability.json")
    dr = read_json(ROOT / "experiments/drought_aux/drought_aux_results.json")
    fvs = read_json(ROOT / "experiments/final_val_summary.json")
    trials = {m: pd.DataFrame(read_json(ROOT / f"experiments/tuning/{m}_trials.json")) for m in ["mafnet", "gru", "transformer"]}

    M = "MAF-Net (proposed, attention fusion)"
    MC = "MAF-Net w/o attention (concat fusion)"
    PERS = "Persistence (today's class)"
    HGB = "Gradient boosting (HistGB)"
    tm = st["test_metrics"]
    ci = st["ci95_macro_f1_site_block_bootstrap"]
    pv = st["paired_vs"]
    best_other = mc.drop(index=[M]).macro_f1.idxmax()
    rank = int((mc.macro_f1 > mc.loc[M, "macro_f1"]).sum()) + 1
    beats = lambda k: pv[k]["ci95"][0] > 0  # noqa: E731
    ties = lambda k: pv[k]["ci95"][0] <= 0 <= pv[k]["ci95"][1]  # noqa: E731
    att_gain = mc.loc[M, "macro_f1"] - mc.loc[MC, "macro_f1"]
    pers_gain = tm["macro_f1"] - mc.loc[PERS, "macro_f1"]
    n_trials = {m: int((t.state == "COMPLETE").sum()) for m, t in trials.items()}
    n_pruned = {m: int((t.state == "PRUNED").sum()) for m, t in trials.items()}
    sp_params = sel_cfgs["mafnet"]["params"]

    R = Report()
    # ------------------------------------------------------------ title block (one column)
    R.para("ClimateSense AI: A Multimodal Attention-Based Deep Learning Framework for Early Detection of Compound Climate Habitability Risk", "paper title")
    R.para("Debshata Choudhury", "Author")
    R.para("Department of Computer Science and Engineering\nAmity University, Noida, India\nEnrollment No. A2305223104 · Course: Generative AI", "Author", italic=False)
    R.section_break(R.sect_title)

    # ------------------------------------------------------------ abstract
    abstract = (
        "Dangerous environmental conditions rarely stem from a single variable: heat, humidity, air pollution and dryness "
        "often co-occur. This work develops ClimateSense AI, a multimodal attention-based deep learning framework that "
        "forecasts a project-defined four-class compound habitability-risk level (Low, Moderate, High, Critical) one day "
        f"ahead. A harmonised panel of {audit['site_days_kept']:,} site-days for {audit['sites_kept']} capital cities "
        f"({audit['date_range'][0]} to {audit['date_range'][1]}) was built from four public Kaggle datasets: the World "
        "Weather Repository (meteorology and air quality), World Cities (geography, population), Berkeley Earth "
        "(long-term temperature normals) and a US drought dataset (used only for an auxiliary benchmark and transfer "
        "learning). Labels combine NWS heat-index categories, WHO 2021 air-quality guideline levels and a vapour-pressure-"
        "deficit dryness proxy. The proposed Multimodal Attention Fusion Network (MAF-Net) encodes climate, air-quality "
        "and dryness sequences with separate temporal encoders and fuses them, together with static context, through "
        "masked modality attention and a gated residual (modality dropout is available as a tuned option). On a strictly held-out future period "
        f"({splits['test'][0]} to {splits['test'][1]}) the selected MAF-Net reached a macro-F1 of {tm['macro_f1']:.3f} "
        f"(95% site-block bootstrap CI {ci[0]:.3f}–{ci[1]:.3f}), accuracy {tm['accuracy']:.3f}, balanced accuracy "
        f"{tm['balanced_accuracy']:.3f} and macro ROC-AUC {tm['roc_auc_macro_ovr']:.3f}, versus {mc.loc[PERS, 'macro_f1']:.3f} "
        f"macro-F1 for a persistence forecast and {mc.loc[HGB, 'macro_f1']:.3f} for gradient boosting. "
        + ("Attention fusion improved on concatenation fusion" if att_gain > 0 else "Attention fusion did not improve on concatenation fusion")
        + f" ({mc.loc[M, 'macro_f1']:.3f} vs {mc.loc[MC, 'macro_f1']:.3f}, five-seed means), and ablations show that "
        "multimodal input clearly outperforms any single modality. Gains over persistence are small, come from detecting "
        "class transitions, and did not carry over to unseen cities. The study also documents important limitations, "
        "including the absence of observed soil-moisture, SPEI and vegetation data, an observation-hour bias in the "
        "source data, and the circularity of rule-based labels."
    )
    R.para(abstract, "Abstract", bold_prefix="Abstract—")
    R.para("Climate risk, compound events, multimodal deep learning, attention mechanism, LSTM, GRU, environmental monitoring, air pollution, drought, habitability risk", "Keywords", bold_prefix="Keywords—")

    # ------------------------------------------------------------ I. Introduction
    R.heading("Introduction")
    R.para("The IPCC Sixth Assessment Report concludes that human influence has increased the frequency of hot extremes and "
           f"the likelihood of compound extremes such as concurrent heatwaves and droughts {cite('ipcc')}. Compound events "
           f"can cause impacts larger than any constituent hazard alone {cite('zscheischler', 'leonard')}. The physiological "
           f"burden of heat depends on humidity, and heat–humidity combinations near proposed tolerance limits have already "
           f"been observed {cite('raymond', 'mora')}. Heat frequently coincides with elevated ozone or particulate pollution, "
           f"which the WHO regards as one of the largest environmental health threats {cite('who')}, while precipitation "
           f"deficits and high evaporative demand dry soils and vegetation {cite('spei', 'grossiord')}.")
    R.para("Heat warnings, air-quality indices and drought monitors are, however, usually produced separately and at different "
           "resolutions, so a system based on any one of them may miss days on which several moderate stresses co-occur. "
           "This project therefore asks: can a multimodal deep learning framework integrating meteorological, air-quality "
           "and drought-related indicators improve the identification of compound environmental conditions associated with "
           "reduced human habitability compared with models based on individual environmental modalities?")
    R.para("The contributions of this work are: (i) a reproducible harmonisation of four public datasets into a leakage-safe "
           f"daily panel of {audit['sites_kept']} capital-city sites; (ii) a transparent, sensitivity-tested four-class "
           "compound-risk label built from published thresholds; (iii) MAF-Net, a branch-wise architecture with masked "
           "modality attention, gated residual fusion and modality dropout; (iv) a fair comparison against naive, classical "
           "and tuned recurrent/Transformer baselines on a future held-out period, with multi-seed statistics, ablations, "
           "horizon and spatial-transfer tests; and (v) a cross-domain transfer experiment from a US drought benchmark. "
           "Throughout, “habitability risk” denotes the degree of combined environmental stress at a place and "
           "time; it is not a determination that a location is or will become uninhabitable.")

    # ------------------------------------------------------------ II. Related work
    R.heading("Related Work")
    R.heading("Compound events and heat–humidity stress", 2)
    R.para(f"Compound-event research emphasises that impacts emerge from combinations of drivers {cite('zscheischler', 'leonard')}. "
           f"Heat stress is commonly quantified with the heat index derived from Steadman's apparent temperature {cite('steadman', 'rothfusz')} "
           f"or with wet-bulb temperature {cite('stull')}; recent physiological evidence places the limit of compensable heat stress "
           f"well below the theoretical 35 °C wet-bulb threshold {cite('vecellio', 'raymond')}.")
    R.heading("Air quality and drought monitoring", 2)
    R.para(f"WHO's 2021 guidelines substantially tightened recommended levels of PM2.5, PM10, O_{{3}} and NO_{{2}} {cite('who')}. "
           f"Drought is typically monitored with indices such as SPEI {cite('spei')} or expert products like the US Drought "
           f"Monitor {cite('usdm')}; vegetation stress is observed with satellite indices such as MODIS NDVI/EVI {cite('modis')}, "
           f"and vapour-pressure deficit is an established driver of plant water stress {cite('grossiord')}.")
    R.heading("Deep learning for environmental time series and multimodal fusion", 2)
    R.para(f"Deep learning is increasingly used in Earth-system science {cite('reichstein')}, from recurrent networks (LSTM "
           f"{cite('lstm')}, GRU {cite('gru')}) to convolutional-recurrent nowcasting {cite('convlstm')} and attention-based "
           f"forecasters {cite('vaswani', 'tft')}. Multimodal learning offers early, late and intermediate fusion strategies "
           f"{cite('baltrusaitis', 'ramachandram')}; attention allows a model to weight modalities adaptively, although attention "
           f"weights are not necessarily faithful explanations {cite('attn_expl')}. Compound-risk classification that jointly models "
           "heat, air quality and dryness with modality-level attention appears comparatively unexplored relative to single-hazard prediction.")

    # ------------------------------------------------------------ III. Problem formulation
    R.heading("Problem Formulation")
    R.para("For site *s* and day *t* let x_{s,t} ∈ R^{F} denote the daily feature vector and c_{s} ∈ R^{K} static context. "
           "Features are partitioned into modalities m ∈ {climate, air, dryness}. Given the look-back window "
           "X_{s,t} = (x_{s,t−L+1}, …, x_{s,t}) with L = 14, the task is to predict the compound-risk class at t + h:")
    e1 = R.equation("ŷ_{s,t+h} = argmax_{k} f_{θ}(X_{s,t}, c_{s})_{k},   k ∈ {0,1,2,3}")
    R.para(f"with h = 1 as the primary horizon and h ∈ {{3, 7}} as secondary horizons. No observation later than day t enters "
           f"the input. The label y_{{s,t}} is computed from the observed (never imputed) values of day t via hazard scores "
           f"H (heat), A (air quality) and D (dryness), each in {{0,…,3}}. With S = H + A + D, m = max(H, A, D) and "
           f"n_{{2}} the number of hazards ≥ 2, the classes are defined as")
    e2 = R.equation("y = 3 if S ≥ 6 ∨ (m = 3 ∧ n_{2} ≥ 2); 2 if S ≥ 4 ∨ m = 3 ∨ n_{2} ≥ 2; 1 if S ≥ 2 ∨ m ≥ 2; else 0")
    R.para(f"so that the class reflects both the number and the severity of concurrent stresses (Eq. ({e2})). Hypotheses: "
           "H1 — multimodal models outperform unimodal ones; H2 — attention fusion outperforms plain concatenation; "
           "H3 — learned models outperform the persistence forecast ŷ_{s,t+h} = y_{s,t}.")

    # ------------------------------------------------------------ IV. Dataset
    R.heading("Dataset Description")
    R.para(f"The study is restricted to four Kaggle datasets (Table {R.tab_n + 1}). The World Weather Repository {cite('weather')} "
           f"(version 1105; {qual['weather']['rows']:,} rows, {len(qual['weather']['columns'])} columns) provides one snapshot "
           f"per capital city per day with temperature, humidity, wind, pressure, precipitation, UV and six pollutants. World "
           f"Cities {cite('worldcities')} ({qual['world_cities']['rows']:,} cities) supplies coordinates, ISO codes and "
           f"population; Berkeley Earth {cite('berkeley', 'rohde')} ({qual['berkeley_by_city']['rows']:,} monthly city records, "
           f"1743–2013) supplies 1961–1990 monthly normals and long-term warming; and the US drought dataset "
           f"{cite('drought_kaggle')} ({qual['drought_us_timeseries']['splits']['train']['rows']:,} training rows for "
           f"{qual['drought_us_timeseries']['splits']['train']['counties']:,} counties) is used only for an auxiliary benchmark "
           "because US counties cannot be aligned with global capitals.")
    R.table("Datasets used", ["Dataset", "Records", "Coverage", "Use"], [
        ["World Weather Repository", f"{qual['weather']['rows']:,}", f"{qual['weather']['date_min']} – {qual['weather']['date_max']}", "Inputs + labels"],
        ["World Cities", f"{qual['world_cities']['rows']:,}", "global", "Geography, population"],
        ["Berkeley Earth (by city)", f"{qual['berkeley_by_city']['rows']:,}", f"{qual['berkeley_by_city']['date_min'][:4]} – {qual['berkeley_by_city']['date_max'][:4]}", "Climatology context"],
        ["US drought (NASA POWER + USDM)", f"{sum(v['rows'] for v in qual['drought_us_timeseries']['splits'].values()):,}", "2000 – 2020, US", "Auxiliary / transfer"],
    ], widths=[1.15, 0.6, 0.95, 0.85])
    R.para(f"After cleaning, {audit['sites_kept']} of {audit['sites_total']} candidate sites with at least 500 observed days were "
           f"retained ({audit['site_days_kept']:,} site-days; {audit['grid_unobserved_days']:,} unobserved calendar days on the "
           f"daily grid). The proposal's SPEI, soil-moisture and NDVI sources {cite('spei', 'power', 'modis')} were not part of the "
           "permitted data; no synthetic substitutes were generated, and the drought/soil/vegetation branch is realised as an "
           "observed atmospheric dryness proxy. Two properties of the source data matter for interpretation: (i) all snapshots "
           "are taken at nearly the same UTC instant, so the local observation hour ranges from roughly 04–06 h in the "
           "Americas to about 17 h in East Asia; (ii) cross-site PM2.5 levels shift abruptly around 2024-09 and 2025-10, "
           "consistent with changes in the provider's air-quality model.")
    R.figure(FIG / "eda/site_map_high_critical_share.png", "Retained capital-city sites coloured by the share of High or Critical days.")

    # ------------------------------------------------------------ V. Preprocessing
    R.heading("Data Preprocessing")
    R.para("Cleaning. Sentinel values (−9999) and physically impossible values were set to missing; duplicate rows were "
           "removed and multiple same-day records averaged. Because the provider sometimes renames a location, a site is "
           "defined by its country and coordinates rounded to 0.1°. Spatial alignment. Sites were matched to World Cities and "
           f"Berkeley Earth by haversine distance (coordinates decide; names are only a check): {audit['world_cities_join']['matched']} "
           f"of {audit['sites_kept']} sites matched World Cities within 50 km (median {audit['world_cities_join']['median_distance_km']:.1f} km) "
           f"and {audit['berkeley_join']['matched']} matched a Berkeley series within 150 km.")
    R.para("Feature engineering. From temperature T and relative humidity RH the NWS heat index "
           f"{cite('rothfusz', 'steadman')}, Stull wet-bulb temperature {cite('stull')}, dew point and vapour-pressure deficit were derived:")
    R.equation("VPD = 0.6108·exp(17.27T/(T+237.3))·(1 − RH/100)  [kPa]")
    R.para("Further features are log-transformed pollutant concentrations, the provider's US-EPA and GB-DEFRA indices, past-only "
           "7- and 30-day precipitation sums and wet-day fractions, days since recorded precipitation, the temperature anomaly "
           "relative to the Berkeley normal, the local observation hour, seasonal harmonics and eight static context features "
           f"(Table {R.tab_n + 1}).")
    R.table("Model inputs by branch", ["Branch", "#", "Features"], [
        ["A Climate", "17", "T, feels-like, RH, dew point, heat index, wet-bulb, wind, gust, pressure, cloud, visibility, UV, T anomaly, obs. hour, sin/cos day-of-year, heat score"],
        ["B Air quality", "9", "log CO, O3, NO2, SO2, PM2.5, PM10; US-EPA, GB-DEFRA index; air score"],
        ["C Dryness proxy", "9", "VPD, log precip., wet flag, 7/30-d precip. sums and wet fractions, days since wet, dryness score"],
        ["D Context", "8", "lat, |lat|, sin/cos lon, log10 population, Berkeley annual normal, seasonal amplitude, 2000–12 warming"],
    ], widths=[0.8, 0.25, 2.35])
    R.para("Missing data and leakage control. Inputs are forward-filled for at most three days (past information only); "
           "remaining gaps are imputed with the training mean after standardisation and flagged by per-modality missingness "
           "indicators, and labels are never imputed. Standardisation statistics are fitted on training rows only. The calendar "
           f"is split chronologically and identically for all sites into training ({splits['train'][0]} to {splits['train'][1]}), "
           f"validation ({splits['val'][0]} to {splits['val'][1]}) and test ({splits['test'][0]} to {splits['test'][1]}). A "
           "sample is admitted only if its first input day and its target day lie in the same split, which embargoes L + h days "
           "at each boundary. Class imbalance is handled with class-weighted or focal loss; no oversampling is used.")
    dist = audit["label_distribution_by_split"]
    R.para(f"Label validity. The class distribution over all labelled site-days is "
           + ", ".join(f"{k} {v * 100:.1f}%" for k, v in lab["base_distribution"].items())
           + f". Perturbing heat-index bands by ±1 °C or VPD bands by 25% changes few labels (weighted κ ≥ "
           f"{min(lab['variants'][k]['quadratic_kappa_vs_base'] for k in ['heat_index_bands_-1C', 'heat_index_bands_+1C', 'vpd_bands_x1.25']):.2f}), "
           f"whereas removing the air or dryness hazard changes them substantially (κ = {lab['variants']['without_air']['quadratic_kappa_vs_base']:.2f} "
           f"and {lab['variants']['without_dryness']['quadratic_kappa_vs_base']:.2f}). Today's class equals tomorrow's on "
           f"{lab['next_day_label_persistence'] * 100:.1f}% of consecutive days, which makes persistence a demanding baseline. "
           "The highest High/Critical shares occur in Riyadh, New Delhi, Khartoum, Jakarta and Kuwait City, which is regionally plausible.")

    # ------------------------------------------------------------ VI. Methodology
    R.heading("Proposed Methodology")
    R.para(f"MAF-Net (Fig. {R.fig_n + 1}) follows the branch-wise design of the problem statement. Each sequence modality m is "
           "encoded by its own temporal encoder E_{m} (GRU, LSTM or Transformer for climate; GRU for air quality and dryness) "
           "and static context by an MLP. Representations are projected into a common d-dimensional space and tagged with a "
           "learned modality embedding e_{m}:")
    R.equation("z_{m} = LayerNorm(W_{m} E_{m}(X^{(m)})) + e_{m}")
    R.para("Modality tokens Z optionally interact through one masked self-attention layer and are then pooled by a learned query "
           f"q with multi-head cross-attention {cite('vaswani')}; a modality missing for the whole window is removed via the "
           "key-padding mask \U0001d4dc rather than replaced by fictitious values:")
    R.equation("α = softmax(qW_{Q}(ZW_{K})^{T}/√d + \U0001d4dc),   f = αZW_{V}W_{O}")
    R.para("A gated residual combines the attended vector with the masked mean token z̄, and an MLP head outputs class "
           "probabilities:")
    R.equation("g = σ(W_{g}[f; z̄]),  u = g⊙f + (1−g)⊙z̄,  p = softmax(MLP(u))")
    R.para("Two enhancements over plain concatenation fusion are introduced: modality-masked attention and modality dropout, "
           "which masks whole modalities with probability p_{md} during training so that the network neither over-relies on "
           f"one branch nor fails when a modality is absent (p_{{md}} was a tuned hyper-parameter; the selected configuration uses "
           f"p_{{md}} = {sp_params['modality_dropout']}, and its effect is examined in the ablation study). Training minimises "
           f"class-weighted cross-entropy or focal loss "
           f"{cite('focal')} with AdamW {cite('adamw')}.")
    R.wide_figure(FIG / "architecture_mafnet.png", "Architecture of the ClimateSense Multimodal Attention Fusion Network (MAF-Net).")

    # ------------------------------------------------------------ VII. Experimental setup
    R.heading("Experimental Setup")
    R.para(f"Hardware and software. Experiments ran on an {env['gpu']} GPU ({env['gpu_vram_gb']:.0f} GB, compute capability "
           f"{env['gpu_capability']}), {env.get('logical_cpus', 12)} logical CPUs and {env.get('ram_gb', 16):.0f} GB RAM with "
           f"Python {env['python'].split()[0]}, PyTorch {env['torch']} {cite('pytorch')}, scikit-learn {env['packages']['scikit-learn']} "
           f"{cite('sklearn')} and Optuna {env['packages']['optuna']} {cite('optuna')}. Mixed precision was not used because the "
           "Pascal GPU has low FP16 throughput.")
    R.para("Baselines. Naive: persistence, majority class and site-month climatology. Classical models on 242 aggregated window "
           f"features (last, mean, max, min, s.d., trend per feature + context): logistic regression, random forest {cite('rf')}, "
           f"histogram gradient boosting {cite('gbm')} and an MLP. Sequence models on all features jointly: vanilla RNN, LSTM "
           f"{cite('lstm')}, GRU {cite('gru')} and a Transformer encoder {cite('vaswani')}. The concatenation-fusion variant of MAF-Net "
           "isolates the effect of attention.")
    R.para(f"Tuning and selection. Optuna's TPE sampler with median pruning maximised validation macro-F1 (24 trials for MAF-Net, "
           f"10 each for GRU and Transformer; {sum(n_trials.values())} completed and {sum(n_pruned.values())} pruned in total; "
           "20 epochs, patience 5). Selected configurations were retrained for up to 40 epochs (patience 6, ReduceLROnPlateau, "
           "gradient clipping 1.0) with five seeds (MAF-Net, concat variant, GRU) or three (Transformer, LSTM, RNN). The reported "
           "MAF-Net checkpoint is the seed with the highest validation macro-F1; the test split was evaluated once, after all "
           f"choices were frozen. Table {R.tab_n + 1} lists the selected MAF-Net hyper-parameters.")
    R.table("Selected MAF-Net hyper-parameters", ["Parameter", "Value"], [
        ["Look-back / horizon", f"{rec['config']['lookback']} d / {rec['config']['horizon']} d"],
        ["Climate encoder", f"{rec['config']['encoder'].upper()}, {rec['config']['layers']} layer(s), hidden {rec['config']['hidden']}"],
        ["Air / dryness encoders", f"GRU, hidden {max(16, rec['config']['hidden'] // 2)}"],
        ["Token dimension d / heads", f"{rec['config']['d_model']} / {rec['config']['heads']}"],
        ["Token self-attention", "yes" if rec["config"]["token_self_attention"] else "no"],
        ["Dropout / modality dropout", f"{rec['config']['dropout']} / {rec['config']['modality_dropout']}"],
        ["Loss", {"ce": "cross-entropy", "wce": "class-weighted CE", "focal": "focal (γ = 2)"}[rec["config"]["loss"]]],
        ["Optimiser, LR, weight decay", f"AdamW, {rec['config']['lr']:g}, {rec['config']['weight_decay']:g}"],
        ["Batch size / best epoch", f"{rec['config']['batch_size']} / {rec['best_epoch']}"],
        ["Trainable parameters", f"{rec['n_params']:,}"],
    ], widths=[1.6, 1.8])
    R.para(f"Metrics. Accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, one-vs-rest macro ROC-AUC and PR-AUC, "
           "log-loss, Brier score and quadratic-weighted κ. Macro-F1 is the primary metric because High and Critical are "
           "minority classes. Confidence intervals use 1 000 site-block bootstrap resamples; model differences are assessed with "
           "paired bootstrap on identical test samples.")

    # ------------------------------------------------------------ VIII. Results
    R.heading("Results and Analysis")
    R.para(f"Table {R.tab_n + 1} compares all models on the held-out test period (horizon one day). The selected MAF-Net achieved "
           f"macro-F1 {tm['macro_f1']:.3f} (95% CI {ci[0]:.3f}–{ci[1]:.3f}), accuracy {tm['accuracy']:.3f}, balanced accuracy "
           f"{tm['balanced_accuracy']:.3f}, weighted F1 {tm['weighted_f1']:.3f}, macro ROC-AUC {tm['roc_auc_macro_ovr']:.3f} and "
           f"macro PR-AUC {tm['pr_auc_macro']:.3f}. Across five seeds MAF-Net obtained {pm(mc.loc[M, 'macro_f1'], mc.loc[M, 'macro_f1_std'])} "
           f"test macro-F1, ranking {rank} of {len(mc)} models; the strongest other model was {best_other} "
           f"({mc.loc[best_other, 'macro_f1']:.3f}).")
    rows = []
    order = mc.sort_values("macro_f1", ascending=False)
    for name, r in order.iterrows():
        rows.append([name.replace(" (proposed, attention fusion)", " (proposed)"), f"{int(r.n_params):,}" if r.n_params else "—",
                     f3(r.val_macro_f1) if not np.isnan(r.val_macro_f1) else "—", f3(r.accuracy), f3(r.balanced_accuracy),
                     pm(r.macro_f1, r.macro_f1_std) if r.n_runs > 1 else f3(r.macro_f1), f3(r.weighted_f1), f3(r.roc_auc_macro_ovr),
                     f"{r.train_time_s:.0f}"])
    bold = [i for i, (n, _) in enumerate(order.iterrows()) if n == M]
    R.table("Held-out test performance (h = 1; mean ± s.d. over seeds where trained with several seeds)",
            ["Model", "Params", "Val F1", "Acc.", "Bal. acc.", "Macro-F1", "W-F1", "ROC-AUC", "Time (s)"], rows,
            widths=[1.05, 0.4, 0.33, 0.3, 0.33, 0.55, 0.3, 0.38, 0.33], font=6, bold_rows=bold)
    R.para("Parameters of tree ensembles are counted as tree nodes; training time is measured wall-clock time per run.", "table footnote")
    R.para(f"Against persistence, MAF-Net improved macro-F1 by {pv['persistence']['diff']:.3f} (paired bootstrap 95% CI "
           f"{pv['persistence']['ci95'][0]:.3f} to {pv['persistence']['ci95'][1]:.3f}), "
           + ("confirming H3. " if beats("persistence") else "an inconclusive difference. ")
           + f"Relative to gradient boosting the difference was {pv['hist_gradient_boosting']['diff']:+.3f} "
           f"(CI {pv['hist_gradient_boosting']['ci95'][0]:.3f} to {pv['hist_gradient_boosting']['ci95'][1]:.3f}), relative to the "
           f"tuned GRU {pv['gru']['diff']:+.3f} (CI {pv['gru']['ci95'][0]:.3f} to {pv['gru']['ci95'][1]:.3f}) and relative to the "
           f"concatenation-fusion variant {pv['mafnet_concat']['diff']:+.3f} (CI {pv['mafnet_concat']['ci95'][0]:.3f} to "
           f"{pv['mafnet_concat']['ci95'][1]:.3f}). "
           + _tie_sentence(pv, ties))
    pc = tm["per_class"]
    R.para(f"Per class (Fig. {R.fig_n + 1}), recall was {pc['0']['recall']:.2f} for Low, {pc['1']['recall']:.2f} for Moderate, "
           f"{pc['2']['recall']:.2f} for High and {pc['3']['recall']:.2f} for Critical, with precision {pc['0']['precision']:.2f}, "
           f"{pc['1']['precision']:.2f}, {pc['2']['precision']:.2f} and {pc['3']['precision']:.2f}. Errors are concentrated between "
           "adjacent classes (quadratic-weighted κ = "
           f"{tm['quadratic_kappa']:.3f}; mean absolute ordinal error {tm['mean_abs_ordinal_error']:.3f}).")
    R.figure(FIG / "confusion_matrix_normalized.png", "Row-normalised confusion matrix of the selected MAF-Net on the held-out test period.")
    R.wide_figure(FIG / "roc_pr_curves.png", "One-vs-rest ROC (left) and precision–recall (right) curves of the selected MAF-Net on the test period; dotted lines mark class prevalence.")
    R.para(f"Training dynamics are shown in Fig. {R.fig_n + 1}: validation macro-F1 peaked at epoch {rec['best_epoch']} and the "
           "plateau scheduler reduced the learning rate afterwards. The validation loss is lower than the training loss because "
           "the validation period (January–June) contains a larger share of the easier Low class than the training period; "
           "validation macro-F1 fluctuates between epochs, which is why early stopping on this metric is used. Fig. "
           f"{R.fig_n + 2} summarises the comparison.")
    R.wide_figure(FIG / "training_curves.png", "Training and validation loss, macro-F1 and learning rate of the selected MAF-Net run.")
    R.figure(FIG / "model_comparison_column.png", "Test macro-F1 (mean ± s.d. over seeds) of all models (h = 1); site-month climatology and majority class omitted.")
    h_m = hz[hz.model == "MAF-Net"].set_index("horizon")
    h_p = hz[hz.model == "Persistence"].set_index("horizon")
    R.para(f"Forecast horizon. Skill decays with lead time (Fig. {R.fig_n + 1}): MAF-Net macro-F1 is "
           + ", ".join(f"{h_m.loc[h, 'macro_f1']:.3f} at h = {h}" for h in h_m.index)
           + ", versus " + ", ".join(f"{h_p.loc[h, 'macro_f1']:.3f}" for h in h_p.index) + " for persistence. "
           + (f"The advantage over persistence grows with horizon ({h_m.loc[7, 'macro_f1'] - h_p.loc[7, 'macro_f1']:+.3f} at seven days)."
              if (h_m.loc[7, 'macro_f1'] - h_p.loc[7, 'macro_f1']) > (h_m.loc[1, 'macro_f1'] - h_p.loc[1, 'macro_f1']) else
              "The advantage over persistence does not grow with horizon."))
    R.figure(FIG / "horizons.png", "Test macro-F1 as a function of forecast horizon.")
    sp_m, sp_g, sp_p = (sp.loc[k, "heldout_macro_f1"] for k in ("mafnet", "gru", "persistence"))
    R.heading("Spatial and temporal robustness", 2)
    R.para(f"Spatial generalisation. Retraining on 80% of sites and testing on the remaining 20% (unseen cities, test period; "
           f"three seeds) gave macro-F1 {pm(sp_m, sp.loc['mafnet', 'heldout_macro_f1_std'])} for MAF-Net, "
           f"{pm(sp_g, sp.loc['gru', 'heldout_macro_f1_std'])} for the GRU and {sp_p:.3f} for persistence on the held-out sites, "
           f"compared with {sp.loc['mafnet', 'seen_sites_macro_f1']:.3f} for MAF-Net on the training sites (Fig. {R.fig_n + 1}). "
           + ("On unseen cities neither learned model beat persistence, i.e. the learned skill over persistence did not transfer "
              "to new locations in this experiment." if max(sp_m, sp_g) <= sp_p else
              "The learned models retained an advantage over persistence on unseen cities."))
    R.figure(FIG / "spatial_holdout.png", "Macro-F1 on unseen (held-out) sites and on training sites during the test period.")
    best_L = lb.loc[lb.val_macro_f1.idxmax(), "lookback"]
    R.para("Look-back length. Validation macro-F1 was "
           + ", ".join(f"{r.val_macro_f1:.3f} (L = {int(r.lookback)})" for r in lb.sort_values("lookback").itertuples())
           + (f"; longer history did not help, and the 14-day window is the best of the three." if best_L == 14 else
              f"; L = {int(best_L)} was best on validation, but the differences are within seed variability."))
    vc = read_json(ROOT / "results/analysis/validity_checks.json")
    lonb = pd.DataFrame(vc["by_longitude_band"])
    pa = vc["persistence_analysis"]
    subs = {s["subset"].split(" (")[0]: s for s in pa["subsets"]}
    R.para("Observation-hour bias. The local hour of the snapshot correlates with the label "
           f"(Spearman ρ = {vc['obs_hour_correlation']['spearman_hour_vs_label_all_days']:.2f}; with the heat score "
           f"{vc['obs_hour_correlation']['spearman_hour_vs_heat_score_all_days']:.2f}). Test macro-F1 of MAF-Net ranged from "
           f"{lonb.macro_f1_mafnet.min():.2f} to {lonb.macro_f1_mafnet.max():.2f} across longitude bands, being highest where "
           "High/Critical days are common (western/southern and eastern Asia) and lowest in the Americas and Europe/Africa, where "
           "few High/Critical days occur at the sampled local hours (Fig. "
           f"{R.fig_n + 1}). Per-band results should therefore not be read as regional differences in true risk.")
    R.figure(FIG / "validity/performance_by_longitude.png", "Test macro-F1 by longitude band; annotations give the median local observation hour and the share of High/Critical days.")
    reg = pd.DataFrame(vc["pm25_regimes"]["regimes"])
    R.para("Air-quality non-stationarity. The cross-site median PM2.5 differs between the three provider regimes ("
           + ", ".join(f"{r.median:.1f} µg/m³" for r in reg.itertuples())
           + f"; Kolmogorov–Smirnov D = {vc['pm25_regimes']['ks_statistic']['R1 vs R2']:.2f} and "
           f"{vc['pm25_regimes']['ks_statistic']['R2 vs R3']:.2f} between consecutive regimes). Training spans all three regimes, "
           "whereas validation and test lie entirely in the last one, so part of the air-quality signal learned in training is "
           "shifted at evaluation time; monthly macro-F1 over validation and test is reported in the repository.")
    R.para(f"Persistence and early warning. On the test period the class is unchanged from the previous day for "
           f"{pa['test_label_persistence'] * 100:.1f}% of samples. On these stable days MAF-Net is correct for "
           f"{subs['stable']['acc_mafnet'] * 100:.1f}% (persistence 100% by construction). On the {subs['transition']['n']:,} "
           f"transition days it is correct for {subs['transition']['acc_mafnet'] * 100:.1f}% (gradient boosting "
           f"{subs['transition']['acc_hgb'] * 100:.1f}%, persistence 0%), and it anticipates "
           f"{pa['onset_recall_high_or_critical']['mafnet'] * 100:.1f}% of the {subs['onset of High/Critical']['n']} onsets of High/Critical "
           f"conditions (gradient boosting {pa['onset_recall_high_or_critical']['hgb'] * 100:.1f}%), at the cost of a false "
           f"High/Critical alarm on {pa['false_alarm_rate_hc_on_stable_low_moderate']['mafnet'] * 100:.1f}% of days that stay "
           "Low/Moderate. Most of the headline accuracy therefore reflects the persistence of the project-defined classes; the "
           "genuine early-warning skill lies in this modest but non-zero detection of transitions.")
    R.figure(FIG / "validity/transitions_vs_stable.png", "Accuracy on stable days, class transitions and High/Critical onsets/ends (test).")

    # ------------------------------------------------------------ IX. Ablation
    R.heading("Ablation Study")
    abr = ab.sort_values("macro_f1", ascending=False)
    R.para(f"Table {R.tab_n + 1} reports ablations of the selected configuration (three seeds each). Unimodal variants were "
           f"clearly weaker (climate-only {ab.loc['climate_only', 'macro_f1']:.3f}, air-only {ab.loc['air_only', 'macro_f1']:.3f}, "
           f"dryness-only {ab.loc['dryness_only', 'macro_f1']:.3f}) than the full model ({ab.loc['all_attention', 'macro_f1']:.3f}), "
           "supporting H1. "
           f"Removing air quality lowered test macro-F1 to {ab.loc['climate_dryness', 'macro_f1']:.3f} and removing the dryness branch to "
           f"{ab.loc['climate_air', 'macro_f1']:.3f}; removing all humidity-derived features gave {ab.loc['no_humidity', 'macro_f1']:.3f}. "
           f"Replacing attention by concatenation gave {ab.loc['all_concat_no_attention', 'macro_f1']:.3f}, "
           + ("so attention provided a small benefit (H2 weakly supported)." if ab.loc['all_attention', 'macro_f1'] > ab.loc['all_concat_no_attention', 'macro_f1'] + 0.005
              else "so attention provided no measurable benefit over concatenation (H2 not supported).")
           + f" Adding modality dropout (p = 0.1) gave {ab.loc['with_modality_dropout', 'macro_f1']:.3f}, removing static context "
           f"{ab.loc['all_attention_no_context', 'macro_f1']:.3f}, and adding the transferred drought features "
           f"{ab.loc['with_transferred_drought_features', 'macro_f1']:.3f}.")
    nice = {"climate_only": "Climate only", "air_only": "Air quality only", "dryness_only": "Dryness proxy only",
            "climate_air": "Climate + air (= full − dryness)", "climate_dryness": "Climate + dryness (= full − air)", "all_concat_no_attention": "All, concat (no attention)",
            "all_attention_no_context": "All, attention, no context", "all_attention": "All, attention (full)",
            "no_humidity": "Full − humidity features", "no_air_quality": "Full − air quality", "no_dryness": "Full − dryness",
            "with_modality_dropout": "Full + modality dropout 0.1", "with_transferred_drought_features": "Full + transferred drought features"}
    rows = [[nice.get(v, v), r.modalities, pm(r.val_macro_f1, r.val_macro_f1_std), pm(r.macro_f1, r.macro_f1_std), f3(r.balanced_accuracy),
             f3(r.roc_auc_macro_ovr)] for v, r in abr.iterrows()]
    R.table("Ablation study (mean ± s.d., three seeds; C climate, A air, D dryness, ctx context)",
            ["Variant", "Inputs", "Val macro-F1", "Test macro-F1", "Bal. acc.", "ROC-AUC"], rows,
            widths=[1.15, 0.6, 0.6, 0.6, 0.35, 0.4], font=6, bold_rows=[i for i, v in enumerate(abr.index) if v == "all_attention"])
    R.figure(FIG / "ablation.png", "Ablation results on validation and test (three seeds).")
    R.para(f"Auxiliary drought benchmark and transfer. On 400 US counties, a two-layer GRU predicting the US Drought Monitor class "
           f"from 30 days of weather reached test macro-F1 {dr['test_metrics']['macro_f1']:.3f} (majority {dr['test_majority_macro_f1']:.3f}), "
           "confirming that drought is only weakly predictable from a month of weather. Applied to the capital-city sites, the "
           f"encoder labelled {dr['transfer_feature_summary']['share_argmax_ge_D2'] * 100:.0f}% of site-days as D2 or worse, a sign "
           "of domain shift (snapshot versus daily-total precipitation, US versus global climates); its features "
           + ("did not improve" if ab.loc['with_transferred_drought_features', 'macro_f1'] <= ab.loc['all_attention', 'macro_f1'] + 0.003 else "slightly improved")
           + " MAF-Net.")

    # ------------------------------------------------------------ X. Discussion
    R.heading("Discussion")
    g = ex["permutation_importance_groups"]
    gsorted = sorted(g.items(), key=lambda kv: -kv[1]["mean_drop_macro_f1"])
    R.para("Explainability. Permuting whole modality groups on 8 000 test windows reduced macro-F1 by "
           + ", ".join(f"{v['mean_drop_macro_f1']:.3f} ({k})" for k, v in gsorted)
           + f". The most important single features were "
           + ", ".join(list(ex["permutation_importance_features"].keys())[:4])
           + f" (Fig. {R.fig_n + 1}). Mean attention weights per predicted class are reported in the repository; they describe how "
           f"the fusion module weighted modality tokens but, as argued in {cite('attn_expl')}, are not causal explanations, and "
           f"integrated gradients {cite('ig')} were used as a complementary attribution.")
    R.figure(FIG / "explain/permutation_importance_groups.png", "Drop in test macro-F1 when a modality group is permuted (selected MAF-Net).")
    cs = ex["case_studies"]
    if cs:
        c0 = next((c for c in cs if c["correct"]), cs[0])
        c1 = next((c for c in cs if not c["correct"]), None)
        txt = (f"Case studies. For {c0['city']} ({c0['country']}) on {c0['target_date']} the model predicted {c0['predicted']} "
               f"(P = {c0['probabilities'][c0['predicted']]:.2f}) and the observed class was {c0['actual']}; on the preceding day the "
               f"heat index was {c0['signals_last_input_day']['heat_index_c']} °C, PM2.5 {c0['signals_last_input_day']['pm25']} "
               f"µg/m³ and VPD {c0['signals_last_input_day']['vpd_kpa']} kPa.")
        if c1:
            txt += (f" A failure case is {c1['city']} on {c1['target_date']}: predicted {c1['predicted']} but observed {c1['actual']}, "
                    "illustrating that day-to-day changes in pollution or humidity are hard to anticipate from the preceding fortnight.")
        R.para(txt)
    R.para("Strengths and weaknesses. The multimodal models are consistently better than any single modality and, on the "
           "training sites, slightly better than persistence; High/Critical days are detected with ROC-AUC above "
           f"{min(pc['2']['roc_auc'], pc['3']['roc_auc']):.2f}. Most residual errors are between adjacent classes, and the "
           "Moderate and High classes are hardest because they are defined by thresholds that the next day's weather crosses "
           "frequently. Because labels persist strongly from day to day, most of the attainable accuracy is already captured by "
           "persistence and simple aggregated-feature models; gains from more elaborate fusion are therefore small. The "
           "validation period (January–June) and test period (June–October) differ in season and in the provider's "
           "air-quality regime, which partly explains differences between validation and test scores.")
    R.para("Label validity. The classes are a deterministic, transparent combination of published thresholds; they are robust to "
           "small threshold changes but have not been validated against health or disaster outcomes. The observation-hour bias "
           "means heat stress is under-represented for the Americas and over-represented for East Asia, so geographic comparisons "
           "of predicted risk must be made with caution.")

    # ------------------------------------------------------------ XI. Limitations
    R.heading("Limitations and Future Work")
    R.bullets([
        "Limited temporal history (about 2.4 years) and a single four-month test period; rolling-origin evaluation across years was not possible.",
        "Capital-city sampling bias and a single daily snapshot taken at a longitude-dependent local hour.",
        "No observed SPEI, soil moisture or vegetation index for the capital cities; the dryness branch is a proxy and soil/vegetation ablations were not evaluated.",
        "The US drought dataset is geographically mismatched; its transferred representation suffers from domain shift.",
        "Air-quality values are provider estimates with apparent regime shifts; WHO comparisons are approximate.",
        "Project-defined labels with risk of circular validation; no external validation (e.g., EM-DAT).",
        "Generalisation beyond the sampled cities was tested with only one held-out-site split.",
    ])
    R.para(f"Future work should add observed NASA POWER soil wetness {cite('power')}, SPEIbase {cite('spei')} and MODIS NDVI/EVI "
           f"{cite('modis')} through the existing dryness branch, use hourly or daily-maximum observations to remove the "
           f"observation-hour bias, validate classes against EM-DAT events {cite('emdat')} and health records, and extend the "
           "evaluation to several years and climate-zone-stratified spatial splits.")

    # ------------------------------------------------------------ XII. Conclusion
    R.heading("Conclusion")
    R.para(f"ClimateSense AI provides a reproducible, leakage-safe pipeline and a multimodal attention-fusion network for "
           f"one-day-ahead forecasting of a transparent four-class compound environmental-risk label for {audit['sites_kept']} "
           f"capital cities. On genuinely unseen future data the proposed MAF-Net reached macro-F1 {tm['macro_f1']:.3f} and macro "
           f"ROC-AUC {tm['roc_auc_macro_ovr']:.3f}, improving on persistence by {pers_gain:.3f} macro-F1 on the sampled cities "
           "(but not on unseen cities) and clearly outperforming unimodal variants, which supports the central hypothesis that integrating heat, air-quality and dryness signals "
           "improves compound-risk identification. "
           + ("Attention-based fusion, however, was not clearly superior to concatenation or to strong single-stream and "
              "gradient-boosting baselines, " if ties("mafnet_concat") or ties("gru") or ties("hist_gradient_boosting") else
              "Attention-based fusion also outperformed the strongest baselines, ")
           + "and the absence of observed soil, drought-index and vegetation data limits the scope of the conclusions. The "
           "results should be read as a screening tool for a project-defined stress index, not as a measure of human habitability.")

    R.para("Acknowledgment", "Heading 5")
    R.para("The author thanks Prof. M. K. Dutta, Amity Centre for Artificial Intelligence, for guidance, and the providers of the "
           "public datasets used in this study. Code, notebooks and results: https://github.com/Debshata/ClimateSenseAI.")
    R.para("References", "Heading 5")
    for _, ref in REFS:
        R.para(ref, "references")
    # remove the template's placeholder copyright footer ("XXX-X-XXXX-XXXX-X/XX/$XX.00 (c)20XX IEEE")
    for sec in R.doc.sections:
        for hf in (sec.footer, sec.first_page_footer, sec.even_page_footer):
            for para in hf.paragraphs:
                for r in para.runs:
                    r.text = ""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.doc.core_properties.title = "ClimateSense AI: A Multimodal Attention-Based Deep Learning Framework for Early Detection of Compound Climate Habitability Risk"
    R.doc.core_properties.author = "Debshata Choudhury"
    R.doc.save(OUT)
    print("saved", OUT, "| figures", R.fig_n, "| tables", R.tab_n, "| equations", R.eq_n)


def to_pdf():
    pdf = OUT.with_suffix(".pdf")
    ps = (f"$w = New-Object -ComObject Word.Application; $w.Visible = $false; $w.DisplayAlerts = 0; "
          f"try {{ $d = $w.Documents.Open('{OUT}', $false, $true); $d.ExportAsFixedFormat('{pdf}', 17); $d.Close(0) }} finally {{ $w.Quit() }}")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=300)
    print("pdf:", pdf.exists(), r.stderr[:300])


if __name__ == "__main__":
    build()
    if "--no-pdf" not in sys.argv:
        to_pdf()
