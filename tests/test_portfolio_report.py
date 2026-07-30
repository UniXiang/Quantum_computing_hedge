"""PDF report smoke test for the US n=24 run artifact."""
import json
import os
from pathlib import Path
import sys

from pypdf import PdfReader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from portfolio_report import generate_portfolio_report
from real_portfolio import load_config


def test_generate_portfolio_report(tmp_path):
    result_path = Path(__file__).parent.parent / "results" / (
        "us_n24_portfolio.json"
    )
    if not result_path.exists():
        return
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    config = load_config(
        Path(__file__).parent.parent / "configs" / "portfolio_us_n24.yaml"
    )
    output = tmp_path / "report.pdf"
    generate_portfolio_report(
        payload,
        config,
        output,
        {
            "generated_at_utc": "2026-07-28T10:00:00+00:00",
            "host": "test-host",
            "platform": "test-platform",
            "python": "3.10",
            "portfolio_solver_device": "CPU",
            "qaoa_gpu_note": "测试状态",
            "brsmi_snapshot": "GPU 0 test snapshot",
        },
    )
    reader = PdfReader(output)
    assert len(reader.pages) >= 4
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "n=24" in text
    assert "Beta" in text
    assert "SNDK" in text
