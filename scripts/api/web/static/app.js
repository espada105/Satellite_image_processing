const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const gridEl = document.getElementById('grid');

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

    (data.results || []).forEach(r => {
      const card = document.createElement('div');
      card.className = 'card';
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

