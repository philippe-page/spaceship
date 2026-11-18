export const STORY_OPENINGS = [
  "Once upon a time",
  "It was a dark and stormy night when",
  "Long ago, in a forgotten corner of the world,",
  "It all started the morning",
  "Nobody ever expected that",
  "Deep beneath the surface of the city,",
  "The alarm went off, but",
  "I never intended to become",
  "There was an old legend about",
  "From the moment the letter arrived,",
  "High above the clouds,",
  "Years after the war ended,",
  "She knew it was a mistake the second",
  "In the quiet hours just before dawn,",
  "They told us never to open the door to",
  "Everything changed the day",
]

/** Picks a random story opening that begins with the same words as `text`. */
export function expandToStoryStart(text: string) {
  const words = text.match(/\S+/g) ?? []
  const candidates = STORY_OPENINGS.filter((start) => {
    const startWords = start.split(/\s+/)
    return startWords.length >= words.length && words.every((word, i) => word === startWords[i])
  })
  return candidates[Math.floor(Math.random() * candidates.length)] ?? text
}
