const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const gridEl = document.getElementById('grid');
const modalEl = document.getElementById('imageModal');
const modalImgEl = document.getElementById('modalImage');
const modalCloseBtn = document.getElementById('modalClose');

function openModal(data) {
  if (!modalEl) return;
  modalImgEl.src = data.image_url;
  modalImgEl.alt = data.image_id || 'Satellite image';
  modalEl.classList.remove('hidden');
  modalEl.classList.add('show');
  document.body.classList.add('modal-open');
}

function closeModal() {
  if (!modalEl) return;
  modalEl.classList.remove('show');
  modalEl.classList.add('hidden');
  modalImgEl.src = '';
  document.body.classList.remove('modal-open');
}

if (modalCloseBtn && modalEl) {
  modalCloseBtn.addEventListener('click', closeModal);
  modalEl.addEventListener('click', (event) => {
    if (event.target === modalEl) {
      closeModal();
    }
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && modalEl.classList.contains('show')) {
      closeModal();
    }
  });
}

function addMsg(text, cls) {
  const div = document.createElement('div');
  div.className = `msg ${cls}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function send() {
  const q = inputEl.value.trim();
  if (!q) return;
  inputEl.value = '';
  addMsg(q, 'user');
  sendBtn.disabled = true;
  gridEl.innerHTML = '';
  addMsg('검색 중...', 'bot');

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: q, top_k: 24 })
    });
    const data = await resp.json();
    const last = messagesEl.lastChild;
    if (last && last.classList.contains('bot')) last.remove();
    addMsg(data.answer || '(응답 없음)', 'bot');

    const results = data.results || [];

    if (!results.length) {
      const empty = document.createElement('div');
      empty.className = 'empty';
      empty.textContent = '검색된 이미지가 없습니다.';
      gridEl.appendChild(empty);
      return;
    }

    results.forEach(r => {
      const card = document.createElement('div');
      card.className = 'card';
      card.tabIndex = 0;
      card.setAttribute('role', 'button');
      card.setAttribute('aria-label', `${r.image_id || 'Satellite image'} 보기`);
      const img = document.createElement('img');
      img.src = r.image_url;
      img.alt = r.image_id;
      const cap = document.createElement('div');
      cap.className = 'cap';
      cap.textContent = r.caption;
      const sim = document.createElement('div');
      sim.className = 'sim';
      sim.textContent = `similarity: ${Number(r.similarity || 0).toFixed(4)}`;
      card.appendChild(img);
      card.appendChild(cap);
      card.appendChild(sim);
      card.addEventListener('click', () => openModal(r));
      card.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          openModal(r);
        }
      });
      gridEl.appendChild(card);
    });
  } catch (e) {
    addMsg('에러가 발생했습니다.', 'bot');
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener('click', send);
inputEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });

