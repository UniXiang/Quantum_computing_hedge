"""Generate the audited PDF report accompanying each US n=24 run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    PageBreak,
    Spacer,
    Table,
    TableStyle,
)


COMPANY_PROFILES = {
    "SNDK": (
        "Sandisk Corporation",
        "闪存与数据存储公司，核心方向包括 NAND 闪存和存储产品。该持仓使组合暴露于"
        "AI、数据中心和终端存储需求，同时也承受存储芯片价格周期风险。",
        "https://www.sandisk.com/company/about-us",
    ),
    "MU": (
        "Micron Technology, Inc.",
        "存储半导体厂商，产品覆盖 DRAM、NAND、SSD 与高带宽内存。组合由此获得"
        "AI 基础设施和内存景气周期暴露。",
        "https://www.micron.com/about/company",
    ),
    "DFTX": (
        "Definium Therapeutics, Inc.",
        "原 MindMed，聚焦精神疾病药物研发的临床阶段生物科技公司。潜在收益受临床、"
        "监管和融资事件驱动，个股风险显著高于成熟企业。",
        "https://ir.definiumtx.com/about",
    ),
    "TWST": (
        "Twist Bioscience Corporation",
        "以硅基平台生产合成 DNA，服务医药、农业、诊断和工业生物等领域。属于高成长"
        "合成生物学暴露，估值和商业化节奏敏感。",
        "https://www.twistbioscience.com/company/about",
    ),
    "IRDM": (
        "Iridium Communications Inc.",
        "提供全球卫星语音、数据和定位导航授时服务，客户涉及航空、航海、政府、应急和"
        "关键基础设施。可带来不同于传统互联网网络的卫星通信暴露。",
        "https://www.iridium.com/company/",
    ),
    "CSCO": (
        "Cisco Systems, Inc.",
        "网络、安全、可观测性与协作基础设施公司。相较组合中的高波动成长股，它代表"
        "更成熟的企业 IT 与网络基础设施暴露。",
        "https://www.cisco.com/site/us/en/about/index.html",
    ),
    "CSX": (
        "CSX Corporation",
        "北美大型货运铁路运营商，网络覆盖美国东部主要经济区域。它为组合加入实体运输"
        "和工业周期暴露，降低纯科技/生物科技主题集中度。",
        "https://www.csx.com/index.cfm/about-us/company-overview/",
    ),
}


def _register_fonts() -> tuple[str, str]:
    candidates = [
        (
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyhbd.ttc",
        ),
        (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        ),
        (
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ),
    ]
    for regular, bold in candidates:
        if Path(regular).exists() and Path(bold).exists():
            try:
                pdfmetrics.registerFont(TTFont("ReportCN", regular))
                pdfmetrics.registerFont(TTFont("ReportCN-Bold", bold))
                return "ReportCN", "ReportCN-Bold"
            except Exception:
                continue
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light", "STSong-Light"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def collect_runtime_environment(gpu_note: str = "") -> dict[str, Any]:
    snapshot = ""
    brsmi = shutil.which("brsmi")
    if brsmi:
        try:
            snapshot = subprocess.run(
                [brsmi],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            ).stdout.strip()
        except Exception as exc:
            snapshot = f"BR-SMI 查询失败: {exc}"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "portfolio_solver_device": "CPU（模拟退火 + SLSQP）",
        "qaoa_gpu_note": gpu_note or (
            "本命令未执行 QAOA；金融筛选、模拟退火和连续权重分配均在 CPU 完成。"
        ),
        "brsmi_snapshot": snapshot or "当前环境未检测到 brsmi。",
    }


def _styles(font: str, bold: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName=bold,
            fontSize=22,
            leading=29,
            textColor=colors.HexColor("#102A43"),
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName=font,
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#486581"),
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName=bold,
            fontSize=15,
            leading=21,
            textColor=colors.HexColor("#0B7285"),
            spaceBefore=8,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName=bold,
            fontSize=11.5,
            leading=16,
            textColor=colors.HexColor("#334E68"),
            spaceBefore=5,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9.2,
            leading=15,
            textColor=colors.HexColor("#243B53"),
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName=font,
            fontSize=7.5,
            leading=11,
            textColor=colors.HexColor("#486581"),
        ),
        "table": ParagraphStyle(
            "Table",
            parent=base["BodyText"],
            fontName=font,
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#243B53"),
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            parent=base["BodyText"],
            fontName=bold,
            fontSize=7.5,
            leading=10,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9.2,
            leading=15,
            textColor=colors.HexColor("#102A43"),
            backColor=colors.HexColor("#E6FCF5"),
            borderColor=colors.HexColor("#20C997"),
            borderWidth=0.8,
            borderPadding=8,
            spaceAfter=10,
        ),
    }


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    escaped = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(
        ">", "&gt;"
    )
    return Paragraph(escaped, style)


def _table(
    rows: list[list[Any]],
    widths: list[float],
    styles: dict[str, ParagraphStyle],
    header: bool = True,
) -> Table:
    converted: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        style = styles["table_head"] if header and row_index == 0 else styles[
            "table"
        ]
        converted.append([_p(cell, style) for cell in row])
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BCCCDC")),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B7285")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white, colors.HexColor("#F0F4F8")
                ]),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def _footer(canvas, doc, font: str) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
    canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
    canvas.setFont(font, 7)
    canvas.setFillColor(colors.HexColor("#627D98"))
    canvas.drawString(18 * mm, 8.5 * mm, "Quantum Hedge - US n=24 运行报告")
    canvas.drawRightString(
        A4[0] - 18 * mm, 8.5 * mm, f"第 {doc.page} 页"
    )
    canvas.restoreState()


def generate_portfolio_report(
    payload: dict[str, Any],
    config: dict[str, Any],
    output: str | Path,
    runtime: dict[str, Any] | None = None,
) -> Path:
    font, bold = _register_fonts()
    styles = _styles(font, bold)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runtime = runtime or collect_runtime_environment()
    doc = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="Quantum Hedge US n=24 运行报告",
        author="Quantum Hedge",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="normal",
    )
    doc.addPageTemplates(
        PageTemplate(
            id="report",
            frames=[frame],
            onPage=lambda canvas, current_doc: _footer(
                canvas, current_doc, font
            ),
        )
    )
    story: list[Any] = []
    meta = payload["meta"]
    allocation = payload["allocation"]
    selected = payload["selected_directions"]
    qaoa = config["qaoa"]
    objective = config["objective"]
    selection = config["selection"]
    universe = config["universe"]

    story.extend(
        [
            _p("Quantum Hedge 美股双向组合运行报告", styles["title"]),
            _p(
                f"报告时间：{runtime['generated_at_utc']} ｜ "
                f"数据截止：{payload['as_of']} ｜ n={selection['qubits']} ｜ "
                f"金融求解器：{payload['solver']}",
                styles["subtitle"],
            ),
            _p(
                "结论摘要：本次模型从全市场美股中筛选 9 个候选标的，与 BTC、原油和黄金"
                "共同构造 12 个标的的 long/short 双向 24-qubit QUBO。修正后的组合以"
                "美股多头为 Alpha 主体，以小规模另类资产空头作风险调节；净敞口 40%，"
                "样本估计 Beta 为 0.584，接近 0.6 的目标。",
                styles["callout"],
            ),
            _p("1. 运行状态与测试参数", styles["h1"]),
        ]
    )
    run_rows = [
        ["项目", "本次值", "含义"],
        [
            "组合运行时间（UTC）",
            (
                f"{runtime.get('started_at_utc', '未记录')} 至 "
                f"{runtime.get('completed_at_utc', '未记录')}"
            ),
            f"总耗时 {float(runtime.get('duration_seconds', 0.0)):.3f} 秒",
        ],
        ["报告时间（UTC）", runtime["generated_at_utc"], "本地 PDF 生成时间"],
        ["组合运行主机", runtime["host"], "远端金融组合求解所在主机"],
        [
            "本地报告环境",
            f"Python {runtime['python']}",
            runtime["platform"],
        ],
        ["n / qubits", selection["qubits"], "QUBO 二进制变量数量；这里固定为 24"],
        [
            "变量布局",
            meta["variable_layout"],
            "每个标的各占 long/short 两个位，同一标的不可同时多空",
        ],
        [
            "金融求解",
            f"SA seed={qaoa['seed']}，预算={qaoa['sa_budget_seconds']} 秒",
            "模拟退火选择方向，随后用 SLSQP 分配连续权重",
        ],
        [
            "QAOA 参数",
            f"p={qaoa['layers']}，{qaoa['dtype']}，top_k={qaoa['top_k']}",
            "GPU QAOA 使用的层数、复数精度和最终候选数",
        ],
        ["计算设备", runtime["portfolio_solver_device"], "本报告的组合结果不是 GPU QAOA 输出"],
        ["GPU/QAOA 状态", runtime["qaoa_gpu_note"], "失败状态也必须记录，不能当作成功结果"],
    ]
    story.append(
        _table(run_rows, [36 * mm, 63 * mm, 75 * mm], styles)
    )
    story.extend(
        [
            Spacer(1, 6),
            _p("GPU 快照", styles["h2"]),
            _p(runtime["brsmi_snapshot"], styles["small"]),
            _p("2. 数据范围与股票筛选", styles["h1"]),
        ]
    )
    data_rows = [
        ["指标", "数值", "说明"],
        ["源数据行数", f"{meta['source_rows_through_as_of']:,}", "截止日之前通过基础有效性过滤的日线记录"],
        ["截止日有效股票", f"{meta['source_symbols_through_as_of']:,}", "文件标称 3401 只；截止日可用且有效的为 3380 只"],
        ["120 日历史合格", f"{meta['history_eligible_symbols']:,}", "具有最少历史观测数的股票"],
        ["流动性池", f"{meta['liquid_pool_symbols']:,}", "按近 20 日平均成交额选取的前 500 只"],
        ["最终股票候选", meta["finalists"], "因子排序后再执行每行业最多 3 只的上限"],
        ["股票历史窗口", f"{meta['history_start']} 至 {meta['history_end']}", "用于全市场横截面筛选"],
        ["共同建模窗口", f"{meta['common_window_start']} 至 {meta['common_window_end']}", f"股票与 BTC/CL/XAU 共同有效的 {meta['common_observations']} 个观测"],
    ]
    story.append(_table(data_rows, [40 * mm, 42 * mm, 92 * mm], styles))

    story.extend(
        [
            PageBreak(),
            _p("3. 最终组合结果", styles["h1"]),
        ]
    )
    position_rows = [[
        "代码", "公司 / 标的", "类别", "方向", "权重", "因子分", "样本 Beta"
    ]]
    sector_labels = {
        "Technology": "科技",
        "Health Care": "医疗健康",
        "Telecommunications": "通信",
        "Industrials": "工业",
        "Alternative": "另类资产",
    }
    for item in selected:
        direction = "多头" if item["direction"] == "long" else "空头"
        position_rows.append(
            [
                item["underlying"],
                item.get("name", COMPANY_PROFILES.get(
                    item["underlying"], (item["underlying"], "", "")
                )[0]),
                sector_labels.get(item["sector"], item["sector"]),
                direction,
                f"{100 * float(item['weight']):.2f}%",
                f"{float(item['factor_score']):.3f}",
                f"{float(item['beta']):.3f}",
            ]
        )
    story.append(
        _table(
            position_rows,
            [17 * mm, 43 * mm, 29 * mm, 16 * mm, 18 * mm, 22 * mm, 22 * mm],
            styles,
        )
    )
    long_weight = sum(max(float(row["weight"]), 0.0) for row in selected)
    short_weight = sum(max(-float(row["weight"]), 0.0) for row in selected)
    metric_rows = [
        ["组合指标", "数值", "解释"],
        ["多头敞口", f"{long_weight:.2%}", "所有正权重之和"],
        ["空头敞口", f"{short_weight:.2%}", "所有负权重绝对值之和"],
        ["总敞口（Gross）", f"{allocation['gross_exposure']:.2%}", "多头敞口 + 空头敞口，衡量总风险资本使用量"],
        ["净敞口（Net）", f"{allocation['net_exposure']:.2%}", "多头敞口 - 空头敞口；40% 表示仍保持明确净多头"],
        ["组合 Beta", f"{allocation['portfolio_beta']:.3f}", "相对项目等权美股基准的样本系统性敏感度"],
        ["QUBO 能量", f"{payload['qubo_energy']:.6f}", "仅用于同一实例内比较，越低越优；不是收益率"],
    ]
    story.extend(
        [
            Spacer(1, 8),
            _table(metric_rows, [43 * mm, 28 * mm, 103 * mm], styles),
            _p("4. 关键参数解释", styles["h1"]),
        ]
    )
    parameter_rows = [
        ["参数", "设定", "含义"],
        [
            "beta_target",
            objective["beta_target"],
            "目标组合 Beta。0.6 表示希望市场每变动 1%，组合的系统性部分约同向变动 0.6%；"
            "它不是保证值，实际结果允许在风险收益权衡后接近目标。",
        ],
        [
            "min_net_exposure",
            config["allocation"]["min_net_exposure"],
            "净敞口下限。0.4 要求多头权重至少比空头权重多 40 个百分点，防止退化成纯空头组合。",
        ],
        [
            "max_gross_exposure",
            config["allocation"]["max_gross_exposure"],
            "总敞口上限 1.30，限制杠杆与多空仓位绝对值之和。",
        ],
        [
            "max_stock_weight",
            config["allocation"]["max_stock_weight"],
            "单只股票最大 8%，控制个股集中风险。",
        ],
        [
            "max_contract_abs_weight",
            config["allocation"]["max_contract_abs_weight"],
            "单个 BTC/CL/XAU 方向最大绝对权重 12%。",
        ],
        [
            "downside_risk_weight",
            objective["downside_risk_weight"],
            "下行条件协方差惩罚权重；数值越高，模型越重视基准下跌日的共同损失。",
        ],
        [
            "factor weights",
            (
                "动量 30%；下行调整收益 30%；低波动 20%；"
                "流动性 20%"
            ),
            "动量、下行调整收益、低波动和流动性在股票初筛中的横截面权重。",
        ],
        [
            "sector cap",
            config["preselection"]["max_per_sector"],
            "初筛阶段每个行业最多进入 3 只，避免 9 个候选全部来自同一主题。",
        ],
    ]
    story.append(
        _table(parameter_rows, [40 * mm, 35 * mm, 99 * mm], styles)
    )

    story.extend(
        [
            PageBreak(),
            _p("5. 入选公司与业务含义", styles["h1"]),
        ]
    )
    for item in selected:
        code = item["underlying"]
        if code not in COMPANY_PROFILES:
            continue
        name, description, source = COMPANY_PROFILES[code]
        story.append(
            KeepTogether(
                [
                    _p(f"{code} - {name}", styles["h2"]),
                    _p(description, styles["body"]),
                    _p(f"公司资料：{source}", styles["small"]),
                    Spacer(1, 4),
                ]
            )
        )
    story.extend(
        [
            _p("6. 从市场角度理解这套组合", styles["h1"]),
            _p(
                "组合的核心是“成长与基础设施多头 + 小规模跨资产空头”。SNDK 与 MU 提供"
                "存储半导体及 AI 数据基础设施暴露；DFTX 与 TWST 提供高波动、高事件驱动"
                "的生物科技暴露；IRDM 和 CSCO 对应卫星通信及企业网络；CSX 则加入货运"
                "铁路和实体经济周期。这使组合不完全依赖单一科技主题，但科技和医疗仍是"
                "主要风险来源。",
                styles["body"],
            ),
            _p(
                "样本 Beta 为 0.584，意味着在这段历史窗口的线性估计中，若等权美股基准"
                "变动 1%，组合的系统性部分大约同向变动 0.584%。它比全市场多头的系统性"
                "敏感度低，但不是市场中性，也不能保证未来仍保持同样关系。",
                styles["body"],
            ),
            _p(
                "BTC、CL 和 XAU 的空头总计约 5.94%，作用是根据样本内 Alpha、Beta 和"
                "下行协方差对股票多头做小幅调节。这里不应简单理解为“黄金一定能对冲"
                "股票”或“原油一定与股票同向”：模型使用的是本项目文件在 84 个共同观测"
                "中的统计关系。尤其 CL_USDT 与 XAU_USDT 是价格代理，报告尚未纳入标准"
                "期货合约乘数、保证金、展期、基差、资金费率、手续费和滑点，因此不能直接"
                "当作可实现交易收益。",
                styles["body"],
            ),
            _p(
                "总敞口 51.88% 表示约一半资本对应风险仓位绝对值，净敞口 40% 表示组合"
                "仍明显看多股票，同时保留较多现金/未使用风险预算。CSX 的因子分略为负但"
                "仍入选，说明最终 QUBO 并非只按因子分排序，它还同时考虑 Beta 目标、下行"
                "协方差和组合分散化。",
                styles["body"],
            ),
            _p("7. 局限与复现边界", styles["h1"]),
        ]
    )
    limitations = [
        "本结果是模拟退火基线 + 连续权重优化结果，不是已完成的 GPU QAOA 结果。",
        "基准是流动性合格美股池的等权收益，不是 S&P 500 或 Nasdaq-100。",
        "因子、Beta 和协方差均为样本估计；84 个共同观测较短，参数可能不稳定。",
        "当前结果未计手续费、滑点、资金费率、保证金、强平、合约乘数、展期和基差。",
        "QUBO 能量只能在相同实例与参数下横向比较，不能解释为年化收益或 Sharpe。",
        "该报告用于研究与比赛复现，不构成投资建议。",
    ]
    for index, text in enumerate(limitations, 1):
        story.append(_p(f"{index}. {text}", styles["body"]))

    story.extend(
        [
            _p("附录 A：完整运行参数", styles["h1"]),
        ]
    )
    appendix_rows = [["配置路径", "参数值"]]
    for section_name, section in config.items():
        if not isinstance(section, dict):
            appendix_rows.append([section_name, section])
            continue
        for key, value in section.items():
            appendix_rows.append([f"{section_name}.{key}", value])
    story.append(
        _table(appendix_rows, [70 * mm, 104 * mm], styles)
    )
    story.extend(
        [
            Spacer(1, 10),
            _p("附录 B：资料来源", styles["h1"]),
        ]
    )
    for code, (name, _, source) in COMPANY_PROFILES.items():
        story.append(_p(f"{code} - {name}: {source}", styles["small"]))
    story.append(
        _p(
            "行情输入：data/market_daily.parquet；字段契约为 Nasdaq 日线 close/volume、"
            "asset_type、sector、tradable 等。报告中的组合结果来自同目录 JSON。",
            styles["small"],
        )
    )
    doc.build(story)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-note", default="")
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    if args.config:
        import yaml

        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    elif "parameters" in payload:
        config = payload["parameters"]
    else:
        raise ValueError(
            "result JSON has no embedded parameters; pass --config"
        )
    runtime = collect_runtime_environment(args.gpu_note)
    runtime.update(payload.get("run", {}))
    output = generate_portfolio_report(
        payload,
        config,
        args.output,
        runtime,
    )
    print(f"Saved PDF report: {output}")


if __name__ == "__main__":
    main()
