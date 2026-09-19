import { chromium } from "playwright";

const frontend = (process.env.FRONTEND_URL || "https://ai-learnmate-frontend.onrender.com").replace(/\/$/, "");
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
const consoleErrors = [];
const failedRequests = [];
page.on("console", msg => { if (msg.type() === "error") consoleErrors.push(msg.text()); });
page.on("requestfailed", req => failedRequests.push(req.url() + " :: " + (req.failure()?.errorText || "request failed")));

try {
  await page.goto(frontend, { waitUntil: "networkidle", timeout: 90000 });
  await page.getByPlaceholder("Enter learner name").fill("Browser Smoke");
  await page.getByRole("button", { name: /Enter Workspace/i }).click();
  const chat = page.getByPlaceholder(/Message AI LearnMate/i);
  await chat.waitFor({ state: "visible", timeout: 30000 });

  const subjectSelect = page.getByRole("combobox", { name: /Select subject for MCQ generation/i });
  const subjectOptions = await subjectSelect.locator("option").evaluateAll(options => options.map(option => option.textContent?.trim()).filter(Boolean));
  const expectedSubjects = ["C", "C++", "Java", "Python", "Data Structures", "Algorithms", "DBMS", "SQL", "Operating Systems", "Computer Networks", "Digital Logic", "Computer Organization", "Discrete Mathematics", "Mathematics", "Software Engineering", "Web Development", "Artificial Intelligence", "Machine Learning", "Cybersecurity", "Cloud Computing"];
  if (subjectOptions.length !== expectedSubjects.length || expectedSubjects.some(subject => !subjectOptions.includes(subject))) {
    throw new Error("Frontend subject selector is incomplete: " + JSON.stringify(subjectOptions));
  }
  console.log("SUBJECT SELECTOR PASSED:", subjectOptions.length, "subjects");

  await chat.fill("Hi");
  await page.getByTitle("Send message").click();
  await page.locator(".chat-messages .message.assistant:not(.typing)").last().waitFor({ state: "visible", timeout: 90000 });
  const first = await page.locator(".chat-messages .message.assistant").last().innerText();
  if (!first.trim() || /could not reach the AI assistant/i.test(first)) throw new Error("Invalid first AI response: " + first);

  await chat.fill("What is 2 + 3?");
  await page.getByTitle("Send message").click();
  await page.locator(".chat-messages .message.assistant:not(.typing)").last().waitFor({ state: "visible", timeout: 90000 });
  const second = await page.locator(".chat-messages .message.assistant").last().innerText();
  if (!/\b5\b/.test(second)) throw new Error("Browser AI Tutor did not return 5: " + second);

  if (consoleErrors.length) throw new Error("Browser console errors: " + JSON.stringify(consoleErrors));
  if (failedRequests.length) throw new Error("Failed browser requests: " + JSON.stringify(failedRequests));

  console.log("BROWSER SMOKE PASSED");
  console.log("First response:", first.slice(0, 180).replace(/\n/g, " "));
  console.log("Second response:", second.slice(0, 180).replace(/\n/g, " "));
} finally {
  await browser.close();
}
