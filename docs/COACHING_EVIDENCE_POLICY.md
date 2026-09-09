# AI Coach Evidence Policy

The AI coach may use CS2 domain knowledge to choose what a player should
review, but it must not turn a generic principle into a verdict unless the
video pipeline establishes the required context.

This policy incorporates the useful coaching topics from the user-provided
CS2 overview article: counter-strafing, crosshair/head-line placement, cover,
peek discipline, utility, sound information, economy, roles, and trade support.

## Evidence gates

| Coaching topic | Evidence currently available | Allowed output | Not allowed without more evidence |
|---|---|---|---|
| First enemy contact | Opposing player visible, timestamp, nearby shot/damage candidates | Locate a review window; ask the player to inspect movement stop, expected head line, exposure, cover, utility, and trade support | Calling the player passive, rushed, or slow; comparing contact time with the observed round duration |
| Map-control pace | None of map, side, spawn, route, objective phase is reliably established | State that pace cannot yet be judged | Early/late route verdicts or fixed "contest in 15-20 seconds" advice |
| Counter-strafe | A visual shot candidate may be present; camera motion is not player movement | Ask whether movement had stopped before the first shot | Claiming that counter-strafing failed |
| Crosshair placement | Enemy visibility and review frames | Ask whether the crosshair covered the expected enemy head line | Claiming good/bad aim or exact crosshair error from one frame |
| Cover and peeking | Review clip around contact/death | Ask about body exposure, retreat cover, repeated peek, information gained, and trade support | Claiming a peek was wrong without geometry and tactical context |
| Flash exposure | Central viewport whiteout episode and duration | Locate the episode and recommend turning away/using cover before re-peeking | Assigning ownership of the flash or calling it a team flash |
| Scope hold | Continuous scoped viewport episode | Review peripheral-information loss during long holds | Calling the hold incorrect without weapon, angle, and teammate coverage |
| POV death | Stable native health HUD followed by a sustained disappearance | Review last cover, first exposure, repeated line, retreat, and trade potential | Naming the cause of death without confirming the clip |
| Armour on one screenshot | Numeric OCR of the native shield value | Treat values below 60 as low and suggest a next-round refill/helmet check; no warning at 100 | Treating the armour-presence boolean as 0/1, or criticising a normal 100-armour pistol-round purchase for lacking a helmet |
| Economy | Not yet parsed reliably | No economy verdict | Eco/force-buy/save criticism |
| Sound and communication | No audio/voice-information pipeline | No sound/comms verdict | Claiming missed footsteps, weak calls, or poor team roles |

## Timing rule

`first_contact_elapsed / observed_round_duration` is explicitly forbidden as a
pace score. The denominator depends on when the round happened to end, so a
normal spawn-to-contact route can look artificially late in a short round.

Pace analysis may be added only after the system can establish at least the
map, side, spawn context, route or position progression, and the relevant round
phase. Until then, encounter timing is descriptive evidence only.
