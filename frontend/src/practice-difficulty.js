(() => {
  if (window.__aiLearnMatePracticeDifficultyInstalled) return;
  window.__aiLearnMatePracticeDifficultyInstalled = true;

  const STORAGE_KEY = 'aiLearnMate.practiceDifficulty';
  const DIFFICULTIES = [
    ['adaptive', 'Adaptive'],
    ['easy', 'Easy'],
    ['medium', 'Medium'],
    ['hard', 'Difficult'],
  ];
  let selectedDifficulty = localStorage.getItem(STORAGE_KEY) || 'adaptive';
  if (!DIFFICULTIES.some(([value]) => value === selectedDifficulty)) {
    selectedDifficulty = 'adaptive';
  }

  // Intercept only the subject-practice request. The protected AI Assistant
  // /api/chat route and its payload are left untouched.
  const originalFetch = window.fetch.bind(window);
  window.fetch = (input, init = {}) => {
    const url = typeof input === 'string' ? input : input?.url || '';
    if (url.includes('/api/assessment/start') && init.body) {
      try {
        const body = JSON.parse(init.body);
        body.difficulty = selectedDifficulty;
        return originalFetch(input, { ...init, body: JSON.stringify(body) });
      } catch {
        // Preserve the original request when the body is not JSON.
      }
    }
    return originalFetch(input, init);
  };

  const findPracticePanel = () => {
    const panels = [...document.querySelectorAll('.grid .panel')];
    return panels.find((panel) =>
      [...panel.querySelectorAll('button')].some((button) =>
        button.textContent.includes('Generate Adaptive MCQs')
      )
    );
  };

  const installDifficultyControl = () => {
    const panel = findPracticePanel();
    if (!panel || panel.querySelector('.practice-difficulty-wrap')) return;

    const subjectSelect = panel.querySelector('select');
    if (!subjectSelect) return;

    const wrapper = document.createElement('label');
    wrapper.className = 'practice-difficulty-wrap';

    const label = document.createElement('span');
    label.textContent = 'Difficulty';

    const select = document.createElement('select');
    select.className = 'practice-difficulty';
    select.setAttribute('aria-label', 'Practice difficulty');

    for (const [value, text] of DIFFICULTIES) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = text;
      select.appendChild(option);
    }

    select.value = selectedDifficulty;
    select.addEventListener('change', () => {
      selectedDifficulty = select.value;
      localStorage.setItem(STORAGE_KEY, selectedDifficulty);
    });

    wrapper.append(label, select);
    subjectSelect.insertAdjacentElement('afterend', wrapper);
  };

  const observer = new MutationObserver(installDifficultyControl);
  observer.observe(document.documentElement, { childList: true, subtree: true });
  installDifficultyControl();
})();
