// The backend serializes naive UTC datetimes (datetime.utcnow(), no 'Z' or
// offset) -- `new Date(...)` would otherwise read that string as local time
// and shift it by the viewer's UTC offset.
export function parseUtc(isoString: string): Date {
  const hasTimezone = /[zZ]|[+-]\d\d:\d\d$/.test(isoString)
  return new Date(hasTimezone ? isoString : `${isoString}Z`)
}

const TIME_FORMAT = new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit' })
const DATE_FORMAT = new Intl.DateTimeFormat('en-US', { month: 'long', day: 'numeric' })

export function formatTime(isoString: string): string {
  return TIME_FORMAT.format(parseUtc(isoString))
}

export function formatDate(isoString: string): string {
  return DATE_FORMAT.format(parseUtc(isoString))
}

function isSameDay(a: Date, b: Date): boolean {
  return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate()
}

// Sentence case ("Today", "Yesterday", "September 18").
function relativeDay(isoString: string): string {
  const date = parseUtc(isoString)
  const now = new Date()
  if (isSameDay(date, now)) return 'Today'

  const yesterday = new Date(now)
  yesterday.setDate(now.getDate() - 1)
  if (isSameDay(date, yesterday)) return 'Yesterday'

  return DATE_FORMAT.format(date)
}

// All-caps ("TODAY", "YESTERDAY", "SEPTEMBER 18") -- for section headers.
export function dayLabel(isoString: string): string {
  return relativeDay(isoString).toUpperCase()
}
