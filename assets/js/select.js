// Replace with your actual image and audio URLs
const moodData = [
  {
    img: 'assets/images/moods/back_to_school.png',
    audio: 'assets/audio/moods/back_to_school.mp3'
  },
  {
    img: 'assets/images/moods/desert_gardens.jpg',
    audio: 'assets/audio/moods/desert_gardens.mp3'
  },
  {
    img: 'assets/images/moods/green_grass.jpeg',
    audio: 'assets/audio/moods/How_Do_I_Get_My_Grass_Really_Green.mp3'
  },
  {
    img: 'assets/images/moods/smoke_break.jpeg',
    audio: 'assets/audio/moods/smoke_break.mp3'
  },
  {
    img: 'assets/images/moods/take_me_down_the_hills.jpeg',
    audio: 'assets/audio/moods/take_me_down_the_hills.mp3'
  },
  {
    img: 'assets/images/moods/bass.jpg',
    audio: 'assets/audio/moods/feel_the_bass.mp3'
  },
  {
    img: 'assets/images/moods/gonna_be_fine.jpeg',
    audio: 'assets/audio/moods/gonna_be_fine.mp3'
  },
  {
    img: 'assets/images/moods/paris.jpg',
    audio: 'assets/audio/moods/paris_streets.mp3'
  },
  {
    img: 'assets/images/moods/space_cyber_punk.jpg',
    audio: 'assets/audio/moods/space_cyberpunk.mp3'
  }
];

let available = [...moodData];
let shown = [];
let shuffles = 0;

function sampleImages() {
  // Sample 3 without replacement
  let pool = available.filter(item => !shown.includes(item));
  let pick = [];
  for (let i = 0; i < 3 && pool.length > 0; i++) {
    let idx = Math.floor(Math.random() * pool.length);
    pick.push(pool[idx]);
    pool.splice(idx, 1);
  }
  shown = shown.concat(pick);
  return pick;
}

function renderImages(images) {
  const container = document.getElementById('mood-images');
  container.innerHTML = '';
  images.forEach((item, idx) => {
    const img = document.createElement('img');
    img.src = item.img;
    img.className = 'mood-img';
    img.alt = `Mood ${idx+1}`;
    img.onclick = () => selectMood(item);
    container.appendChild(img);
  });
}

function selectMood(item) {
  document.querySelector('.mood-select-container').style.display = 'none';
  const focus = document.getElementById('focus-area');
  const focusImg = document.getElementById('focus-img');
  const audio = document.getElementById('mood-audio');
  focusImg.src = item.img;
  audio.src = item.audio;
  focus.style.display = 'flex';
  audio.play();
}

function showError(msg) {
  const err = document.getElementById('shuffle-error');
  err.textContent = msg;
}

function clearError() {
  showError('');
}

function shakeButton() {
  const btn = document.getElementById('shuffle-btn');
  btn.classList.add('shake');
  setTimeout(() => btn.classList.remove('shake'), 400);
}

document.addEventListener('DOMContentLoaded', () => {
  shown = [];
  shuffles = 0;
  renderImages(sampleImages());

  document.getElementById('shuffle-btn').onclick = () => {
    if (shown.length >= moodData.length) {
      shakeButton();
      showError('No more shuffles allowed. Please select an image.');
      return;
    }
    clearError();
    renderImages(sampleImages());
    shuffles += 1;
  };
});