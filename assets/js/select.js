/**
 * assets/js/select.js
 * Loaded in select.html via:
 *   <script src="{{ site.baseurl }}/assets/js/select.js"></script>
 */

const moodData = [
  { img: '/bayes_beats/assets/images/moods/Image1.jpg',         audio: '/bayes_beats/assets/audio/moods/Image1_Music.mp3' },
  { img: '/bayes_beats/assets/images/moods/Image2.jpg',          audio: '/bayes_beats/assets/audio/moods/Image2_Music.mp3' },
  { img: '/bayes_beats/assets/images/moods/Image3.jpg',            audio: '/bayes_beats/assets/audio/moods/Image3_Music.wav' }
];

function getRandomMoods() {
  return moodData.slice().sort(() => 0.5 - Math.random()).slice(0, 3);
}

function renderImages() {
  const container = document.getElementById('mood-images');
  container.innerHTML = '';                  // clear old
  getRandomMoods().forEach((mood,i) => {
    const img = document.createElement('img');
    img.src   = mood.img;
    img.alt   = `Mood ${i+1}`;
    img.className = 'mood-img';
    img.addEventListener('click', () => showFocus(mood, img));
    container.appendChild(img);
  });
}

function showFocus(mood, clickedImg) {
  // 1) dim the page behind
  document.body.classList.add('darkened-background');

  // 2) dim the two non-clicked thumbs
  document.querySelectorAll('.mood-img').forEach(el => {
    if (el !== clickedImg) el.classList.add('dim');
  });

  // 3) populate the overlay
  const focusArea = document.getElementById('focus-area');
  document.getElementById('focus-img').src = mood.img;
  const audio = document.getElementById('mood-audio');
  audio.src = mood.audio;

  // 4) show the overlay
  focusArea.style.display = 'flex';
  focusArea.scrollIntoView({ behavior: 'smooth', block: 'center' });

  // 5) autoplay
  audio.load();
  audio.play().catch(_ => {
    audio.autoplay = true;
    audio.play().catch(()=>{});
  });

  // 6) inject Back button
  addBackButton();
}

function addBackButton() {
  const focusArea = document.getElementById('focus-area');
  if (document.getElementById('back-button')) return;

  const btn = document.createElement('button');
  btn.id = 'back-button';
  btn.textContent = 'Back to Selection';

  btn.addEventListener('click', () => {
    // 1) undo dims + overlay + audio
    document.body.classList.remove('darkened-background');
    document.querySelectorAll('.mood-img.dim').forEach(i => i.classList.remove('dim'));
    document.getElementById('mood-audio').pause();
    focusArea.style.display = 'none';

    // 2) re-render & scroll up
    renderImages();
    document.getElementById('mood-select-container')
            .scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  focusArea.appendChild(btn);
}

document.addEventListener('DOMContentLoaded', () => {
  // 1) ensure overlay is hidden on load
  document.getElementById('focus-area').style.display = 'none';

  // 2) wire shuffle button
  document.getElementById('shuffle-btn').onclick = renderImages;

  // 3) first render
  renderImages();
});
