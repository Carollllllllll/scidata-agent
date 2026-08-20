from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def create_csv_fixture() -> Path:
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    path = EXAMPLES / "perovskite_metrics.csv"
    dataframe = pd.DataFrame(
        [
            {
                "Paper Title": "Demo perovskite solar cells",
                "Material": "MAPbI3",
                "Method": "spin coating",
                "PCE (%)": 21.3,
                "Condition": "AM 1.5G illumination",
            },
            {
                "Paper Title": "Demo perovskite solar cells",
                "Material": "FAPbI3",
                "Method": "annealing",
                "PCE (%)": 23.1,
                "Condition": "after 1000 h stability test",
            },
            {
                "Paper Title": "Demo benchmark",
                "Material": "Diffusion baseline",
                "Method": "diffusion",
                "FID": 8.7,
                "Condition": "test set",
            },
        ]
    )
    dataframe.to_csv(path, index=False, encoding="utf-8")
    return path


def create_pdf_fixture() -> Path:
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    path = EXAMPLES / "demo_scientific_paper.pdf"
    if path.exists():
        return path
    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(72, 740, "Demo Perovskite Solar Cell Study")
    pdf.setFont("Helvetica", 10)
    lines = [
        "Abstract",
        "The MAPbI3 device prepared by spin coating achieved a PCE of 21.3% under AM 1.5G illumination.",
        "A FAPbI3 sample showed a stability retention of 92% after 1000 h in nitrogen.",
        "The SnO2 transport layer reduced the RMSE of the calibration model to 0.12 eV on the test set.",
        "Method",
        "Devices were fabricated by spin coating and annealing with SnO2 as the transport layer.",
    ]
    y = 710
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 18
    pdf.showPage()
    pdf.setFont("Helvetica", 10)
    pdf.drawString(72, 740, "Results")
    pdf.drawString(72, 710, "The chromophore showed an absorption wavelength of 336 nm in acetonitrile.")
    pdf.save()
    return path


if __name__ == "__main__":
    print(create_csv_fixture())
    print(create_pdf_fixture())
