### 2. LESSONS_LEARNED.md 

```markdown
# LESSONS_LEARNED.md

## Definition of Done (DoD) – PilotWeave-Skin

Jedes Feature / Commit / Pull Request muss folgende Kriterien erfüllen:

1. **Code läuft reproduzierbar**  
   - Seed fixiert (z. B. seed=42)  
   - Keine silent failures (alle Exceptions gefangen + geloggt)  

2. **Energy bookkeeping invariant**  
   - Gesamtenergie bleibt ≥ 0 (kein negativer Energie-Output)  
   - Bounds & monotonicity wo physikalisch sinnvoll (z. B. Tc nicht unter 0 K, Kohärenz 0–1)  

3. **No silent failures**  
   - Jede Operation prüft Input-Validität (clip, assert, log)  
   - Reports enthalten `run_status` (success / partial_fail / error) + `error_details`

4. **Vertical Slice Pattern**  
   - Jede größere Änderung hat mindestens einen vollständigen End-to-End-Test (scenario JSON → run → report JSON + summary MD)  

5. **Documentation first**  
   - README + LESSONS_LEARNED up-to-date vor Merge  
   - Jede neue Funktion / Klasse hat Docstring + Beispiel  

6. **Hybrid-Kopplung**  
   - PilotWeave-Reservoir + unsere Projektionsarchitektur (CFC, noise-assisted, Tc-proxy) müssen gekoppelt sein  
   - Kein reiner PW- oder reiner Projection-Code ohne Brücke

## Collaboration Guardrails
- **Issue-first**: Neue Idee → Issue erstellen (mit Use-Case, Ziel, DoD)  
- **Commit small**: Atomic Commits (1 Feature / Fix)  
- **Review**: Mindestens 1 Approval vor Merge (bei Solo: self-review + checklist)  
- **Tests**: Jeder Run erzeugt Report (JSON + MD) – kein Run ohne Trace

Dieses DoD ist **unser API-Contract**.  
Es verhindert Scope-Creep, Garbage-in-Garbage-out und versteckte Bugs.
