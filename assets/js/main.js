document.addEventListener('DOMContentLoaded', () => {
  const bg = document.getElementById('background-container');
  const img = document.getElementById('landing-image');
  const intro = document.getElementById('intro-text');
  const firstCue = document.getElementById('first-scroll-cue');
  const rest = document.getElementById('rest-of-text');
  const secondCue = document.getElementById('second-scroll-cue');
  const buttonContainer = document.getElementById('button-container');
  const button = document.getElementById('get-started');

  // Set up background image
  if (img && bg) {
    const imgUrl = img.getAttribute('src');
    bg.style.backgroundImage = `url(${imgUrl})`;
  }

  // Initial state: show intro and first cue
  intro.classList.add('visible');
  firstCue.classList.remove('hidden');
  rest.classList.remove('visible');
  secondCue.classList.add('hidden');
  buttonContainer.classList.remove('visible');
  button.classList.remove('visible');

  // Fade background and image after short delay
  setTimeout(() => {
    bg.classList.add('background-mode');
    if (img) img.style.opacity = 0;
  }, 500);

  // Handle scroll events
  window.addEventListener('scroll', () => {
    const scrollPosition = window.scrollY;
    const windowHeight = window.innerHeight;

    // Section 1: Intro
    if (scrollPosition < windowHeight * 0.33) {
      intro.classList.add('visible');
      rest.classList.remove('visible');
      buttonContainer.classList.remove('visible');
      firstCue.classList.remove('hidden');
      secondCue.classList.add('hidden');
      button.classList.remove('visible');
    }
    // Section 2: Rest-of-text
    else if (scrollPosition < windowHeight * 0.66) {
      intro.classList.remove('visible');
      rest.classList.add('visible');
      buttonContainer.classList.remove('visible');
      firstCue.classList.add('hidden');
      secondCue.classList.remove('hidden');
      button.classList.remove('visible');
    }
    // Section 3: Button
    else {
      intro.classList.remove('visible');
      rest.classList.remove('visible');
      buttonContainer.classList.add('visible');
      firstCue.classList.add('hidden');
      secondCue.classList.add('hidden');
      button.classList.add('visible');
    }
  });
});