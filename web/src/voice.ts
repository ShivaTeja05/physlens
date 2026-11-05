// One voice for the whole app. Every spoken line goes through here, so leaving a mode, hiding the browser tab or
// turning Voice off silences it, and a chain of lines (a narration) stops instead of going on to the next line:
// cancelling speech fires the line's onend/onerror, which used to start the next line.

let generation = 0

/** Changes whenever speech is stopped; a narration started under an older value must not continue. */
export const voiceGeneration = () => generation

export function stopVoice() {
  generation++
  if ('speechSynthesis' in window) speechSynthesis.cancel()
}

export function pickVoice(): SpeechSynthesisVoice | undefined {
  const voices = speechSynthesis.getVoices()
  const preferred = ['Samantha', 'Daniel', 'Google UK English Female', 'Karen', 'Moira', 'Serena']
  return voices.find((v) => preferred.some((p) => v.name.includes(p))) ?? voices.find((v) => v.lang.startsWith('en'))
}

/** Say one line. `onEnd` runs once when it finishes (or after a safety timeout), never after stopVoice(). */
export function speakLine(text: string, { rate = 1.03, onEnd }: { rate?: number; onEnd?: () => void } = {}) {
  const gen = generation
  let done = false
  const end = () => { if (!done && gen === generation) { done = true; onEnd?.() } }
  if (!('speechSynthesis' in window)) { setTimeout(end, 1500 + text.length * 30); return }
  const u = new SpeechSynthesisUtterance(text)
  const v = pickVoice()
  if (v) u.voice = v
  u.rate = rate
  u.onend = end
  u.onerror = end
  speechSynthesis.speak(u)
  setTimeout(end, 4000 + text.length * 90)   // some browsers never fire onend
}

// Nobody is listening to a hidden or closed tab.
if (typeof document !== 'undefined') {
  document.addEventListener('visibilitychange', () => { if (document.hidden) stopVoice() })
  addEventListener('pagehide', stopVoice)
}
