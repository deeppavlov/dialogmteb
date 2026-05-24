const fs = require("node:fs/promises");
const path = require("node:path");
const { chromium } = require("playwright");

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith("--")) continue;

    const [rawKey, rawValue] = token.split("=", 2);
    const key = rawKey.slice(2);
    let value = rawValue;

    if (value === undefined && argv[i + 1] && !argv[i + 1].startsWith("--")) {
      value = argv[i + 1];
      i += 1;
    }

    args[key] = value ?? "true";
  }
  return args;
}

function toInt(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.round(parsed) : fallback;
}

function normalizeFormat(rawFormat) {
  const format = String(rawFormat || "pdf")
    .toLowerCase()
    .replace(/^\./, "");
  if (!["pdf", "svg", "png"].includes(format)) {
    throw new Error(`Unsupported format "${rawFormat}". Use pdf, svg, or png.`);
  }
  return format;
}

function escapeAttr(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

async function waitForPageToSettle(page) {
  await page.waitForLoadState("networkidle");
  await page.evaluate(async () => {
    if (document.fonts && document.fonts.ready) {
      await document.fonts.ready;
    }
  });
}

async function buildIsolatedExportDocument(page, selector, padding, timeout) {
  const payload = await page.evaluate((selectorValue) => {
    const el = document.querySelector(selectorValue);
    if (!el) {
      return { error: `Element "${selectorValue}" not found` };
    }

    const styleAndLinks = Array.from(
      document.querySelectorAll('style, link[rel="stylesheet"]')
    )
      .map((node) => node.outerHTML)
      .join("\n");

    return {
      elementHtml: el.outerHTML,
      styleAndLinks,
      baseUri: document.baseURI,
    };
  }, selector);

  if (payload.error) {
    throw new Error(payload.error);
  }

  const html = `<!doctype html>
<html>
  <head>
    <meta charset="UTF-8" />
    <base href="${escapeAttr(payload.baseUri)}" />
    ${payload.styleAndLinks}
    <style>
      html, body {
        margin: 0;
        padding: 0;
        background: white;
      }
      #__export_root__ {
        display: inline-block;
        box-sizing: border-box;
        padding: ${padding}px;
        background: white;
      }
    </style>
  </head>
  <body>
    <div id="__export_root__">${payload.elementHtml}</div>
  </body>
</html>`;

  await page.setContent(html, { waitUntil: "networkidle", timeout });
  await waitForPageToSettle(page);
}

async function measureRoot(page) {
  const size = await page.$eval("#__export_root__", (el) => {
    const rect = el.getBoundingClientRect();
    return {
      width: Math.max(1, Math.ceil(rect.width)),
      height: Math.max(1, Math.ceil(rect.height)),
    };
  });
  return size;
}

async function exportPdf(page, outputPath, width, height) {
  await page.addStyleTag({
    content: `@page { size: ${width}px ${height}px; margin: 0; }`,
  });

  await page.pdf({
    path: outputPath,
    printBackground: true,
    width: `${width}px`,
    height: `${height}px`,
    margin: { top: "0", right: "0", bottom: "0", left: "0" },
    preferCSSPageSize: true,
    pageRanges: "1",
  });
}

async function exportPng(page, outputPath, width, height) {
  await page.screenshot({
    path: outputPath,
    clip: { x: 0, y: 0, width, height },
  });
}

async function exportSvg(page, outputPath) {
  const svgText = await page.evaluate(() => {
    const root = document.getElementById("__export_root__");
    if (!root) {
      throw new Error('Element "#__export_root__" not found in isolated document');
    }

    const rect = root.getBoundingClientRect();
    const width = Math.max(1, Math.ceil(rect.width));
    const height = Math.max(1, Math.ceil(rect.height));
    const serializer = new XMLSerializer();
    const serializedRoot = serializer.serializeToString(root);

    const css = [];
    for (const sheet of Array.from(document.styleSheets)) {
      try {
        for (const rule of Array.from(sheet.cssRules)) {
          css.push(rule.cssText);
        }
      } catch {
        // Ignore cross-origin or restricted stylesheets.
      }
    }

    const safeCss = css.join("\n").replaceAll("]]>", "]]]]><![CDATA[>");
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <style><![CDATA[
${safeCss}
  ]]></style>
  <foreignObject x="0" y="0" width="100%" height="100%">
    <div xmlns="http://www.w3.org/1999/xhtml">${serializedRoot}</div>
  </foreignObject>
</svg>`;
  });

  await fs.writeFile(outputPath, svgText, "utf8");
}

function printUsage() {
  console.log(`Usage:
  node scripts/export-pdf.cjs [--url <url>] [--selector <css-selector>] [--padding <px>] [--format pdf|svg|png] [--out <path>]

Examples:
  node scripts/export-pdf.cjs --url http://localhost:4173 --format pdf --out dialogmteb-figure.pdf
  node scripts/export-pdf.cjs --format svg --out dialogmteb-figure.svg

Environment variables:
  DIALOGMTEB_URL, DIALOGMTEB_SELECTOR, DIALOGMTEB_PADDING, DIALOGMTEB_FORMAT, DIALOGMTEB_OUT, DIALOGMTEB_TIMEOUT_MS`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help === "true" || args.h === "true") {
    printUsage();
    return;
  }

  const url = args.url || process.env.DIALOGMTEB_URL || "http://localhost:4173/";
  const selector = args.selector || process.env.DIALOGMTEB_SELECTOR || "#dialogmteb-figure";
  const padding = toInt(args.padding || process.env.DIALOGMTEB_PADDING || 24, 24);
  const timeout = toInt(args.timeout || process.env.DIALOGMTEB_TIMEOUT_MS || 60_000, 60_000);
  const format = normalizeFormat(args.format || process.env.DIALOGMTEB_FORMAT || "pdf");
  const defaultOut = `dialogmteb-figure.${format}`;
  const outputPath = path.resolve(args.out || process.env.DIALOGMTEB_OUT || defaultOut);

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    viewport: { width: 2560, height: 1600 },
    deviceScaleFactor: 1,
  });

  try {
    await page.goto(url, { waitUntil: "domcontentloaded", timeout });
    await waitForPageToSettle(page);

    await buildIsolatedExportDocument(page, selector, padding, timeout);
    const { width, height } = await measureRoot(page);
    await page.setViewportSize({ width, height });

    if (format === "pdf") {
      await exportPdf(page, outputPath, width, height);
    } else if (format === "png") {
      await exportPng(page, outputPath, width, height);
    } else {
      await exportSvg(page, outputPath);
    }

    console.log(`Exported ${format.toUpperCase()} to ${outputPath} (${width}x${height}px)`);
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
