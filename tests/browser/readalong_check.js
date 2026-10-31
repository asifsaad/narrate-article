// Drives a generated read-along page in headless Chromium and reports JSON.
// Used by tests/test_readalong.py::test_page_plays_in_a_browser; skipped
// when Node or Playwright is not installed.
//
//   node tests/browser/readalong_check.js /abs/path/index.html
const { chromium } = require("playwright");

(async () => {
  const pagePath = process.argv[2];
  const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
  const page = await browser.newPage({ viewport: { width: 390, height: 640 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  await page.goto("file://" + pagePath);

  const layout = await page.evaluate(() => ({
    units: document.querySelectorAll(".w").length,
    text: document.getElementById("text").textContent,
    overflowX: document.documentElement.scrollWidth > window.innerWidth,
  }));

  const state = () => page.evaluate(() => {
    const R = window.narrateReadAlong, a = document.getElementById("audio");
    const t = a.currentTime - R.data.audio_offset_s;
    const w = R.activeWord(t);
    const expected = w < 0 ? -1 : R.units.findIndex((u) => u.firstWord <= w && w <= u.lastWord);
    return { t, expected, lit: Number(document.body.dataset.unit ?? -1),
             litText: document.querySelector(".w.current")?.textContent ?? null,
             sentence: Number(document.body.dataset.sentence ?? -1), scrollY: window.scrollY };
  });

  // Play from the start: the lit unit must follow the clock.
  await page.click("#play");
  const samples = [];
  for (let i = 0; i < 14; i++) { await page.waitForTimeout(150); samples.push(await state()); }
  const moved = new Set(samples.map((s) => s.lit).filter((u) => u >= 0)).size;
  const outOfStep = samples.filter((s) => Math.abs(s.lit - s.expected) > 1).length;
  await page.click("#play");

  // Click a word: it lights up and the clock moves to it.
  const target = page.locator(".w").nth(5);
  const targetText = await target.textContent();
  await target.click();
  const afterClick = await state();

  // Keyboard: next sentence, then next paragraph.
  const before = afterClick.sentence;
  await page.keyboard.press("ArrowRight");
  const afterNext = await state();
  await page.keyboard.press("Shift+ArrowRight");
  const afterParagraph = await state();

  // Auto-scroll: jumping to the last sentence brings it into view.
  await page.evaluate(() => {
    const R = window.narrateReadAlong, s = R.data.sentences;
    R.seek(s[s.length - 1].start);
  });
  await page.waitForTimeout(900);
  const scrolled = await page.evaluate(() => {
    const el = document.querySelector(".s.current-sentence");
    const r = el.getBoundingClientRect();
    return { scrollY: window.scrollY, visible: r.top >= 0 && r.bottom <= window.innerHeight };
  });

  // Manual scrolling suspends auto-scroll: scroll to the top just before
  // playback crosses into the last sentence, which would normally follow it.
  await page.evaluate(() => {
    const R = window.narrateReadAlong, s = R.data.sentences;
    R.seek(s[s.length - 1].start - 0.4);
  });
  await page.mouse.wheel(0, -100000);
  await page.waitForTimeout(300);
  await page.click("#play");
  await page.waitForTimeout(900);
  const suspended = await page.evaluate(() => ({
    scrollY: window.scrollY, sentence: Number(document.body.dataset.sentence ?? -1),
    last: window.narrateReadAlong.data.sentences.length - 1,
  }));
  await page.click("#play");

  console.log(JSON.stringify({
    layout, moved, outOfStep, samples: samples.length,
    targetText, afterClick, before, afterNext, afterParagraph, scrolled, suspended, errors,
  }));
  await browser.close();
})().catch((error) => { console.error(error); process.exit(1); });
