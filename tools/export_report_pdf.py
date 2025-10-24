#!/usr/bin/env python3
"""
Render calibration_report.html to PDF using headless Chromium (pyppeteer).

Usage:
  python tools/export_report_pdf.py [--html calibration_report.html] [--out calibration_report.pdf]

If pyppeteer is not installed, install it:
  pip install pyppeteer
"""
import argparse
import asyncio
import os
from pathlib import Path

try:
    from pyppeteer import launch
except Exception as e:
    launch = None


async def render_pdf(html_path: Path, out_path: Path):
    browser = await launch(args=['--no-sandbox'])
    page = await browser.newPage()
    url = html_path.resolve().as_uri()
    await page.goto(url, {'waitUntil': 'networkidle0'})
    # Give images time to load (base64 already inline, but just in case)
    await page.waitFor(250)
    await page.pdf({
        'path': str(out_path),
        'format': 'A4',
        'printBackground': True,
        'margin': {'top': '15mm', 'right': '15mm', 'bottom': '15mm', 'left': '15mm'},
        # Prefer portrait
        'landscape': False,
    })
    await browser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--html', default='calibration_report.html')
    ap.add_argument('--out', default='calibration_report.pdf')
    args = ap.parse_args()
    html = Path(args.html)
    out = Path(args.out)

    if not html.exists():
        raise SystemExit(f'HTML not found: {html}')

    if launch is None:
        raise SystemExit('pyppeteer is not installed. Run: pip install pyppeteer')

    asyncio.get_event_loop().run_until_complete(render_pdf(html, out))
    print('PDF saved to', out)


if __name__ == '__main__':
    main()

