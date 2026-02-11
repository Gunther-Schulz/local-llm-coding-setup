# ARCH REVIEW - Architecture Quality Analysis Protocol

**Instructions for AI assistant.** User triggers when requesting architecture quality analysis and fixes.

---

## PROTOCOL OVERVIEW

**CORE PRINCIPLE:** Systematic architecture quality verification and improvement.

**APPROACH:** Three-phase workflow:
- **A1.1 ANALYZE:** Scan codebase, find architecture violations (always explore NEW areas)
- **A1.2 FIX:** Fix the issues found
- **A1.3 VERIFY:** Check fixes follow architecture

**LOOP STRUCTURE:** A1.1 continues until user chooses "f" to proceed to A1.2.

**DEFAULT ACTION IN A1.1:**
- **Default:** Continue analyzing ("c" command is implied)
- **STOP condition:** ONLY when user explicitly says "f" (fix command)
- **If user doesn't say "f":** Continue with next analysis cycle
- **CRITICAL:** Each analysis cycle MUST explore NEW areas of the codebase

**ANALYSIS CYCLES - DEFAULT BEHAVIOR:**

⚠️ **CRITICAL:** Each cycle must analyze NEW components/areas not yet checked.

- **Scope expansion:** Each "c" iteration widens to NEW areas/components
- **Track analyzed areas:** Maintain list of what's been analyzed to avoid re-checking
- **Progress is incremental:** Each cycle adds findings from new areas
- **STOP condition:** ONLY stop analyzing when user explicitly says "f" (fix command)
- **[PENDING] items are normal:** They represent transparency about what still needs analysis

---

## GAP: WHY CHECKLIST-ONLY ANALYSIS MISSES REAL BUGS

**Observed failure:** Multiple analysis cycles can complete with "architecture good" while **real bugs** (dead code, unvalidated input, wrong parsing) stay undiscovered until the user prompts to "look for more." The protocol then finds them quickly. So the workflow is biased toward **pattern compliance** (does the codebase match the categories?) and **breadth** (new areas per cycle), and away from **correctness** (does this code path run? is this input validated? does parsing match the schema?).

**Root causes:**

1. **Breadth over depth.** "Explore NEW areas" is interpreted as add new *components* each cycle. Analysts move on instead of staying in one component to trace execution paths, follow config keys to their use, or audit one parser against its file format.

2. **Checklist shapes what is searched for.** Violations are framed as C1.6 (duplication), C2.1 (error handling), etc. Bugs like "method never called," "config key ignored," "header not validated," "parse assumes wrong number of columns" do not map cleanly to a single category, so they are not systematically searched for.

3. **No mandatory "contract vs reality" checks.** The protocol verifies that patterns *exist* (e.g. error handling present, config loaded). It does not require: for each **config key** → find code that **reads** it; for each **public or important method** → find **call sites**; for each **external input** (headers, body, file format) → confirm it is **validated** or **parsed** in line with the documented schema.

4. **Verification is "read 2–3 components."** That confirms cross-component consistency (e.g. same style of error handling). It does not confirm "this function is ever invoked" or "this branch is reachable" or "this parser handles all columns."

5. **Output rules (no snippets, concise tracker)** reward summarising findings. They do not force the analyst to ask one more question: "Where is X called?" or "What if this value is negative?"

**Therefore:** To find the kind of bugs that were missed (e.g. `reset_turn` never called, Content-Length not validated, scenario parsing wrong column count), the protocol must **explicitly require** behavior and correctness audits, not only checklist coverage.

---

## MANDATORY: BEHAVIOR AND CORRECTNESS AUDIT (A1.1)

**When:** During A1.1 ANALYZE. At least one full pass of these checks must be done (e.g. one cycle dedicated to it, or one item per cycle until done). Do not rely on checklist categories alone to surface dead code, unused config, or parse/schema mismatches.

**Required checks (grep/read_file as needed):**

1. **Config → code.** For each significant config section or key (e.g. `turn_tracking.auto_reset_turn`, `read_coalescing.max_reads_per_turn`), find where it is **read** and **used**. If a key is never read, or is read but no code path uses it to change behavior, document as [VERIFIED] bug: unused/ignored config.

2. **Public / important methods → call sites.** For methods that are part of the component’s contract (e.g. `reset_turn`, handlers, “main” entry points), search for **call sites**. If a method is never called, document as [VERIFIED] bug: dead code or missing integration.

3. **External input boundaries.** For each place that reads external input (HTTP headers, request body, env, config files, or structured files like scenarios.cfg), confirm: (a) **Validation:** type and range (e.g. Content-Length non-negative, capped); (b) **Parse vs schema:** number and meaning of columns/fields matches the documented or implied schema (e.g. 6-column scenario line parsed with 4 variables = bug). Document missing validation or schema mismatch as [VERIFIED] bugs.

4. **One depth pass per area.** For at least one component per cycle, do a **depth** pass: pick one behavior (e.g. "read coalescing," "turn reset," "scenario launch"), trace from config/entry point to all code paths that implement it, and verify they are connected (no missing calls, no wrong arguments). Document any disconnect as a bug.

**Tracker:** Add findings from these checks to the same tracker (ARCHITECTURAL ISSUES / FINDINGS). Label them so they are visible as **correctness/behavior** issues (e.g. "dead code", "unused config", "input validation", "parse/schema mismatch"). Fix strategy remains as in the rest of the protocol.

---

## A1 - WORKFLOW

### A1.1 ANALYZE

**PROTOCOL ENTRY (AI SELF-CHECK):**

- [ ] Will I generate markdown reports or documentation files?
  - YES → CANNOT proceed. DO NOT generate markdown reports or documentation files.
  - NO → Evidence: [No markdown reports will be generated]

- [ ] Will I show code snippets in output?
  - YES → CANNOT proceed. DO NOT show code snippets. Use prose descriptions only.
  - NO → Evidence: [No code snippets will be shown]

**SYSTEMATIC ARCHITECTURE ANALYSIS:**

**ANALYSIS WORKFLOW:**

FOR each analysis cycle:

1. **Identify NEW areas to analyze:**
   - Check tracker ANALYZED section to see what's been covered
   - Select NEW components/modules/areas not yet analyzed
   - Update ANALYZED section with areas being investigated this cycle
   - **CRITICAL:** Do NOT re-analyze areas already in ANALYZED section

2. **Read relevant files from NEW areas:**
   - Read data structures, contracts, entry points, core business logic components
   - While reading, check relevant C1/C2/C3/P1 checklist items
   - Notice patterns, anomalies, violations while reading
   - **TRACK DEPENDENCIES:** Document component relationships (component A → component B) and cross-component references
   - **BUILD GRAPH:** Add nodes (components) and edges (dependencies/relationships) to dependency graph

3. **When pattern noticed:**
   - **DISCOVERY:** Search systematically for ALL instances (grep/codebase_search)
     - Find WHERE patterns exist
     - Output: "Searched for X, found N instances in components: component_a, component_b, component_c"
     - Code snippets in search results are NOT evidence - only tell WHERE to look
   - **VERIFICATION:** See V1 - Verification Standard
     - Output: "Read code at component_a:lines; component_b:lines; component_c:lines"
     - MUST call read_file even if snippets seen in codebase_search
   - **CHECK ALL RELATED CATEGORIES:** When a pattern is noticed, check ALL relevant checklist categories:
     - Example: If infrastructure access pattern noticed → Check C1.5 (infrastructure abstraction) AND C1.6 (code duplication) AND C2.2 (infrastructure access consistency) AND C2.4 (consistency boundaries) AND RM1 (resource management)
     - Example: If error handling pattern noticed → Check C2.1 (error handling consistency) AND C1.6 (code duplication if error handling duplicated) AND O1 (observability for error tracking)
     - **CRITICAL:** One pattern may violate multiple categories - document each violation separately

4. **Document findings immediately:**
   - Add to tracker with proper status ([PENDING], [PARTIALLY VERIFIED], [VERIFIED])
   - **INTERNALLY (for AI verification):** Collect detailed evidence: component names, instances, discovery paths, verification confirmation
   - **OUTPUT (for human):** Show only summary line + concise pattern explanation (detailed evidence not displayed to avoid cognitive overload)
   - **COUNT COMPONENTS:** List component names from read_file calls internally, count distinct components (NOT instances/offsets)
   - **STATUS:** See S1 for status meanings. Apply V1 verification standard to determine status.
   - If architectural violation verified per V1: Add to ARCHITECTURAL ISSUES with category label AND fix strategy
   - **PLAN FIX STRATEGY:** When adding to ARCHITECTURAL ISSUES, plan how to fix it. Include: affected components (from dependency graph), approach (move/rename/refactor pattern), and verification steps. Strategy guides fix execution during A1.2.
   - **EVALUATE ARCHITECTURAL ALTERNATIVES:** BEFORE planning fix strategy, evaluate where logic should live:
     - [ ] Checked for existing components that handle similar concerns?
       - NO → CANNOT proceed. Search for existing components/services that handle similar concerns.
       - YES → Evidence: [Searched for similar components, found: component_names OR none exist]
     - [ ] Evaluated architectural alternatives (where should this logic live)?
       - NO → CANNOT proceed. Consider multiple options: source component, existing service component, new component.
       - YES → Evidence: [Evaluated alternatives: option1 (pros/cons), option2 (pros/cons), chosen: option with reasoning]
     - [ ] Considered separation of concerns (does this belong in current location)?
       - NO → CANNOT proceed. Evaluate if fix location violates separation of concerns or architectural boundaries.
       - YES → Evidence: [Separation analysis: current_location appropriate OR should move to target_component because reason]
     - **NOTE:** Architectural evaluation happens during analysis when problem context is fresh. Fix strategy must include chosen location with justification.
   - **UPDATE GRAPH:** Mark graph edges with violation types when dependencies violate architecture (e.g., component_a → component_b [C1.2 violation])
   - **CRITICAL:** Document ALL violations found across ALL categories (C1, C2, C3, P1, S1, O1, CM1, T1, DM1, RM1, CT1, API1, SM1, DOC1), not just the "primary" or "biggest" issue
   - **CRITICAL:** Each violation is separate - document C1.4, C1.5, C1.6, C2.1, C2.2, S1.1, O1.2, etc. as distinct findings even if they seem related
   - **CRITICAL:** Do not skip documenting violations because they seem "secondary" or will be "fixed by larger changes" - all violations must be tracked

5. **Run behavior and correctness audit (see "MANDATORY: BEHAVIOR AND CORRECTNESS AUDIT" above):**
   - At least one full pass across the codebase: config→code use, method→call sites, input validation, parse/schema alignment, and one depth pass per area. Do not rely on checklist alone to find dead code, unused config, or wrong parsing.

6. **Verify comprehensive checklist coverage:**
   - Before ending analysis cycle, systematically verify all relevant C1/C2/C3/P1 categories were checked
   - For each category relevant to analysis scope:
     - [ ] Checked for violations in this category?
     - [ ] If violations found, documented in tracker with proper status?
     - [ ] If no violations found, documented as [VERIFIED] with evidence OR marked as not applicable?
   - **CRITICAL:** ALL violations must be documented regardless of priority or whether they'll be fixed together
   - **CRITICAL:** Do not prioritize one category over others - check all systematically
   - **CRITICAL:** Tracker must show comprehensive view of ALL issues across ALL categories (C1, C2, C3, P1, S1, O1, CM1, T1, DM1, RM1, CT1, API1, SM1, DOC1), not just prioritized subset

**CRITICAL RULE:** See V1 - Verification Standard. grep/codebase_search are discovery tools, read_file is verification tool. Must complete V1 verification before marking [VERIFIED] per S1.

**CRITICAL: Documentation of Discovery Path**
- ALWAYS document HOW discovered internally (for AI verification) - not shown in output
- Natural: "Noticed while reading component:line_range" OR targeted: "Searched for pattern X"
- Reference tool calls internally: "Discovered while reading component_a:268-293 (read_file call #3), then searched (codebase_search call #5)"
- Evidence must come from read_file tool calls per V1 (not grep/codebase_search snippets)
- Discovery path collected internally but not displayed in FINDINGS output (only summary shown to human)

**PARALLEL ANALYSIS (Efficiency):**
- Group related checklist items together when they share similar search/verification needs
- Example: C1.1 (component boundaries) + C1.2 (dependency direction) can be checked together
- Example: C2.1 (error handling) + C2.2 (infrastructure access consistency) can share component reads
- Example: C1.5 (infrastructure abstraction) + C1.6 (code duplication) + C2.2 (infrastructure access consistency) can be checked together
- Batch read_file calls for related items when reading same components
- Update tracker with multiple findings per round
- **CRITICAL:** When checking one category, also check related categories that may have violations
- **CRITICAL:** Document findings for ALL categories checked, not just the most obvious violation

**Issue handling:**

- [ ] Architectural violation found (C1/C2/C3/P1 pattern violation)?
  - Found 1 instance via read_file? → Mark [PARTIALLY VERIFIED] in FINDINGS, continue searching for more instances
  - Verified per V1? → Mark [VERIFIED] per S1 in FINDINGS, add to ARCHITECTURAL ISSUES with verified evidence, continue analyzing
  - Not verified yet (no read_file evidence)? → Continue verification (do not add to ARCHITECTURAL ISSUES until verified)
  - Document ALL violations found, even if they seem related or will be fixed together. Each violation is separate.

- [ ] Multiple violations found from same pattern or related patterns?
  - YES → Document EACH violation separately in tracker with its specific category (C1.4, C1.5, C1.6, C2.1, C2.2, C2.4, etc.)
  - Do not combine into single "architectural shortcoming" - each category violation is distinct
  - Example: Missing data access layer violates C1.5 (data access abstraction) AND code duplication violates C1.6 (code duplication) - these are TWO separate violations, both must be documented
  - **CRITICAL:** One code pattern may violate multiple categories - check and document all of them

- [ ] All relevant categories checked for violations?
   - When analyzing an area, check ALL relevant categories (C1/C2/C3/P1/S1/O1/CM1/T1/DM1/RM1/CT1/API1/SM1/DOC1), not just the most obvious one
   - Example: When analyzing infrastructure access → Check C1.5, C1.6, C2.2, C2.4, P1.2, P1.3, RM1, O1
  - Document findings for each category checked, even if no violations found

**ITERATION ("c" - Analyze New Areas):**

When user chooses "c" to continue analysis:

**MANDATORY:** For each iteration:
- [ ] Identify NEW areas/components to analyze:
  - Check ANALYZED section in tracker
  - Select components/modules/areas NOT yet analyzed
  - Update ANALYZED section with new areas being investigated
- [ ] For NEW areas: Use same verification rigor per V1
- [ ] Document NEW findings in tracker with proper status ([VERIFIED], [PARTIALLY VERIFIED], [PENDING])
- [ ] **RE-EVALUATE PREVIOUS CONCLUSIONS:** When new evidence from NEW areas contradicts or clarifies previous findings:
  - Review existing FINDINGS and ARCHITECTURAL ISSUES in tracker
  - If new evidence shows a previous finding was incorrect or needs revision, update the finding with new evidence
  - Document the revision with reference to the new evidence that prompted the re-evaluation
  - This is NOT re-analyzing the same area - it's correcting conclusions based on new context from different areas
- [ ] DO NOT re-analyze already analyzed areas without new evidence (reuse existing findings from tracker)
- [ ] DO NOT re-check already analyzed checklist items in same areas without new evidence
- [ ] Update ANALYZED section with areas covered this cycle

**CRITICAL:** Iteration widens scope only. Reuse verified findings from tracker. Do NOT re-verify everything - only analyze NEW areas with full rigor. However, ALWAYS re-evaluate previous conclusions when new evidence from NEW areas contradicts or clarifies them. This ensures findings remain accurate as investigation scope expands.

**ITERATION:** A1.1 continues until user chooses "f" to proceed to A1.2.

**EXPECTED BEHAVIOR:**
- Multiple "c" cycles are NORMAL and EXPECTED
- Each "c" iteration should:
  - Analyze NEW areas/components
  - Add new findings
  - Expand ANALYZED section
- Progress is incremental - each cycle covers new ground

### A1.2 FIX

**PROTOCOL ENTRY (AI SELF-CHECK):**

- [ ] User chose "f" from menu?
  - NO → CANNOT proceed. Show A1.1 tracker + menu + WAIT.
  - YES → Evidence: [User message contains "f"]

**BEFORE fixing code:**

- [ ] Will I generate markdown reports or documentation files?
  - YES → CANNOT proceed. DO NOT generate markdown reports or documentation files.
  - NO → Evidence: [No markdown reports will be generated]

- [ ] Will I show code snippets in output?
  - YES → CANNOT proceed. DO NOT show code snippets. Use prose descriptions only.
  - NO → Evidence: [No code snippets will be shown]

- [ ] Will I fix violations from ARCHITECTURAL ISSUES section?
  - NO → CANNOT proceed. Fix violations from tracker.
  - YES → Evidence: [Will fix violations from ARCHITECTURAL ISSUES section]

**DURING fixing:**

**BEFORE fixing each violation:**
- [ ] Reviewed fix strategy from ARCHITECTURAL ISSUES?
  - NO → CANNOT proceed. Review fix strategy planned during analysis phase.
  - YES → Evidence: [Reviewed fix strategy: approach, affected components, verification steps]
  - NOTE: Fix strategy was planned during analysis when problem context was fresh. Use it as starting point, but verify and adjust if needed.

- [ ] Checked tracker findings for existing patterns/components to reuse?
  - NO → CANNOT proceed. Review tracker findings for existing patterns.
  - YES → Evidence: [Checked tracker findings]

- [ ] Identified ALL components affected by moving/renaming identifier or pattern?
  - NO → CANNOT proceed. Use dependency graph AND systematic search to find all components that reference the identifier/pattern being moved/renamed.
  - YES → Evidence: [Affected components: source_component (defines identifier), consumer_component1, consumer_component2, ...]
  - NOTE: Must include source component (where identifier/pattern is defined) AND all consumer components (where it is referenced). 
  - **STEP 1:** Check dependency graph for existing relationships (graph shows cross-component references discovered during analysis).
  - **STEP 2:** Use systematic search (grep/codebase_search) to find ALL usages of identifier/pattern.
  - **STEP 3:** Compare graph and search results - search may find relationships not in graph (if analysis didn't cover all areas, or if source component uses its own identifier).
  - **STEP 4:** Combine results - affected components = graph relationships + search results (union, not intersection).
  - Graph is PRIMARY source but may be incomplete - search VERIFIES completeness and catches missing relationships.

- [ ] Verified source component is included in affected components list?
  - NO → CANNOT proceed. Source component must be updated if identifier/pattern is moved or renamed.
  - YES → Evidence: [Source component: component_name is in affected components list]

- [ ] Read context around pattern to be replaced?
  - NO → CANNOT proceed. Read 5-10 lines before and after pattern location.
  - YES → Evidence: [Read context: component:identifier, lines X-Y]

- [ ] Understood semantic behavior of pattern being replaced?
  - NO → CANNOT proceed. Identify behavior: error-raising, conditional, assignment, etc.
  - YES → Evidence: [Pattern behavior: description of what pattern does]

**BEFORE applying helper/component:**
- [ ] Understood helper/component behavior?
  - NO → CANNOT proceed. Read helper/component implementation to understand behavior.
  - YES → Evidence: [Helper behavior: description of what it does]

- [ ] Verified helper/component behavior matches pattern being replaced?
  - NO → CANNOT proceed. Adjust approach or create different helper.
  - YES → Evidence: [Behavior match: helper does X, pattern did X]

FOR each violation being fixed:
- Verify against C1 (component boundaries, single responsibility, abstraction levels)?
  - **C1.6 (Code duplication):** Am I duplicating existing logic? If yes, reuse existing component.
- Verify against C2 (consistency, error handling, validation, atomic operations)?
  - **C2.1 (Error handling):** Am I using existing error handling pattern? If no, use existing pattern.
  - **C2.3 (Validation):** Am I duplicating validation logic? If yes, reuse existing validation functions.
- Verify against C3 (pattern and contract verification)?
  - **C3.3 (Follow established patterns):** Am I following existing patterns from tracker findings? If no, why not?
- Verify against P1 (expensive operations optimized, efficient data loading)?
- Verify against S1 (security patterns)?
- Verify against O1 (observability patterns)?
- Verify against other relevant categories?
- Violation found? Fix violation before continuing.

**DURING fixing:**

**BEFORE replacing each pattern instance:**
- [ ] Verified replacement matches original pattern semantics?
  - NO → CANNOT proceed. Adjust replacement to match behavior.
  - YES → Evidence: [Replacement preserves: error-raising/conditional/assignment behavior]

- [ ] Read context around this specific instance?
  - NO → CANNOT proceed. Read surrounding code for this instance.
  - YES → Evidence: [Context read: component:identifier, lines X-Y]

**WHEN moving/renaming identifier or pattern:**
- [ ] Updated identifier usage AND cross-component references in same operation?
  - NO → CANNOT proceed. When replacing identifier usage, also update all cross-component references (where identifier is declared/imported) in affected components.
  - YES → Evidence: [Updated identifier usage and cross-component references in: component1, component2, ...]
  - NOTE: Identifier replacement and cross-component reference updates must happen together. Do not replace identifier usage without updating references in source component and all consumer components. 
  - **VERIFICATION:** Use dependency graph AND search results to verify all consumer components are updated. Graph shows relationships discovered during analysis, but search ensures completeness (catches relationships not in graph, including source component self-usage).

**AFTER replacing each pattern instance:**
- [ ] Verified syntax is valid?
  - NO → CANNOT proceed. Fix syntax errors.
  - YES → Evidence: [Syntax check passed: component:identifier]

- [ ] Verified replacement preserves intended behavior?
  - NO → CANNOT proceed. Adjust replacement.
  - YES → Evidence: [Behavior preserved: description]

**AFTER each fix:**

- [ ] Code follows architectural patterns?
  - NO → CANNOT proceed. Fix violations.
  - YES → Evidence: [Patterns followed]

- [ ] No new violations introduced?
  - NO → CANNOT proceed. Fix violations.
  - YES → Evidence: [No violations]

- [ ] No duplication of existing patterns/logic (C1.6, C2.1, C3.3)?
  - NO → CANNOT proceed. Refactor to reuse existing patterns/components.
  - YES → Evidence: [Reused existing pattern X from component Y:Z OR verified no existing pattern exists]

- [ ] All pattern replacements in this fix preserve original semantics?
  - NO → CANNOT proceed. Review and correct semantic mismatches.
  - YES → Evidence: [All replacements preserve: error-raising/conditional/assignment behavior]

- [ ] Modified component has valid syntax?
  - NO → CANNOT proceed. Fix syntax errors.
  - YES → Evidence: [Syntax check passed: component:identifier]

- [ ] Modified component can be referenced?
  - NO → CANNOT proceed. Fix reference errors.
  - YES → Evidence: [Reference check passed: component:identifier]

- [ ] All identifiers used in modified components have corresponding cross-component references?
  - NO → CANNOT proceed. Fix missing cross-component references in: [list components with missing references]
  - YES → Evidence: [Cross-component reference completeness verified: all used identifiers have references in component1, component2, ...]
  - NOTE: Verify that all identifiers used in each modified component have corresponding cross-component references (declarations/imports). Use systematic search to find identifier usages and verify references exist.

- [ ] Updated dependency graph to reflect new component relationships?
  - NO → CANNOT proceed. Update dependency graph edges to show new relationships after moving/renaming identifier.
  - YES → Evidence: [Dependency graph updated: old_relationship removed, new_relationship added]
  - NOTE: After moving/renaming identifier, update graph edges to reflect new source component and updated consumer relationships.

- [ ] Mark fix as [FIXED] in tracker FIXES section with modified components listed?

**BEFORE marking all fixes complete (gate to A1.3):**

- [ ] All modified components have valid syntax?
  - NO → CANNOT proceed. Fix syntax errors in: [list components from FIXES section]
  - YES → Evidence: [Syntax check passed for all modified components: component1:identifier1, component2:identifier2, ...]

- [ ] All modified components can be referenced?
  - NO → CANNOT proceed. Fix reference errors in: [list components from FIXES section]
  - YES → Evidence: [Reference check passed for all modified components: component1:identifier1, component2:identifier2, ...]

### A1.3 VERIFY

**SYSTEMATIC VERIFICATION:**

Iterate through each fix in FIXES section.

FOR each fix:
- Read fixed code per V1
- Check against relevant category checklist items (C1/C2/C3/P1/S1/O1/CM1/T1/DM1/RM1/CT1/API1/SM1/DOC1)
- Decision point:
  - No violations found? → Mark [VERIFIED] with evidence
  - Violations found? → Mark [VIOLATION] with evidence, add back to ARCHITECTURAL ISSUES
  - Not checked yet? → Continue verification

THEN: Show verification results to user.

---

## S1 - Status Indicators

**Status meanings:**

- **[PENDING]** = Not yet verified, needs investigation
- **[PARTIALLY VERIFIED]** = Found 1 component via read_file, needs more evidence (see V1 for verification requirements)
- **[VERIFIED]** = Verified with evidence per V1 (2-3+ components/contexts, code read from each, no violations, OR exception justified)
- **[VIOLATION]** = Violation found in fixed code (A1.3 only)
- **[FIXED]** = Fix applied (A1.2 only)
- **[GOOD]** = Pattern follows architecture correctly

---

## TRACKER

**Single structure tracks findings, fixes, and analyzed areas:**

```
ANALYZED:
✅ component_a, component_b, component_c | Categories: C1, C2
✅ component_d, component_e | Categories: C3, P1

DEPENDENCY GRAPH:
component_a → component_b [C1.2 violation: wrong dependency direction]
component_b → component_c [OK]
component_d → component_a [C1.5 violation: infrastructure abstraction missing]
component_e → component_d [OK]

FINDINGS:
🔍 Description | Category | [PENDING] | Next: search query
🔶 Description | Category | [PARTIALLY VERIFIED] | 1 component: component_name (needs 2-3+)
✅ Description | Category | [VERIFIED] | N components, X instances
   Pattern: concise explanation of the pattern/issue

ARCHITECTURAL ISSUES:
❌ Issue description | Category | Violates: [C1/C2/C3/P1 pattern]
   Impact: what this means for the codebase
   Found at: location description
   Fix strategy: [How to fix - planned during analysis when context is fresh]
   - NOTE: Plan fix strategy during analysis phase when problem context is clear. Include: affected components (from dependency graph), approach (move/rename/refactor), and verification steps. Strategy guides fix execution during A1.2.

FIXES:
🔧 Fix description | Category | [FIXED] | Fixed in: component:lines
   What was changed: brief description

VERIFIED FIXES:
✅ Fix description | Category | [VERIFIED] | Evidence: component:lines
   Verification: what was checked
```

**Tracker rules:**
- Add findings immediately when noticed (start [PENDING] per S1)
- Mark [PARTIALLY VERIFIED] per S1 when: Found 1 component via read_file, but need V1 verification for [VERIFIED]
- **MANDATORY:** Before marking [VERIFIED] per S1, MUST complete VERIFICATION CHECKPOINT (see VERIFICATION REQUIREMENTS section below)
- Mark [VERIFIED] per S1 ONLY after: V1 verification complete + explicitly listed component/context names + no blocking issues + VERIFICATION CHECKPOINT passed
- Architectural violation found with 1 component (via read_file)? → Mark [PARTIALLY VERIFIED] per S1 in FINDINGS, do not add to ARCHITECTURAL ISSUES yet (need V1 verification, unless exception)
- Architectural violation verified per V1? → Mark [VERIFIED] per S1 in FINDINGS, add to ARCHITECTURAL ISSUES section with verified evidence and category label
- Violation not verified yet (no read_file evidence)? → Keep [PENDING] per S1 in FINDINGS only, do not add to ARCHITECTURAL ISSUES until verified per V1
- ANALYZED: List components/areas already analyzed. Update each cycle with NEW areas.
- DEPENDENCY GRAPH: Show component relationships (component_a → component_b). Mark edges with violation types when dependencies violate architecture (e.g., [C1.2 violation]). Mark [OK] when dependency is valid. Build incrementally as components are analyzed. Graph shows all discovered dependencies and their architectural status.
- ARCHITECTURAL ISSUES: List violations verified per V1 (all category violations). Only add violations that are [VERIFIED] per S1. Violations with 1 component stay [PARTIALLY VERIFIED] per S1 in FINDINGS until V1 verification complete. Note impact but may not block fixing if workaround exists. Do not add violations based on grep/discovery alone - must verify per V1. Include category label (C1.4, S1.2, O1.3, etc.).
- FIXES: List fixes applied during A1.2. Mark as [FIXED] when code changed.
- VERIFIED FIXES: List fixes verified during A1.3. Mark as [VERIFIED] when verification complete.
- Show tracker AFTER each cycle completes (includes GRAPH by default)

---

## V1 - Verification Standard

**PRINCIPLE:** Verify systemic patterns, not isolated instances.

**Verification count:**
- **Default:** Evidence from 2-3+ distinct components
- **Contextual:** Small codebase (<5 components) or monolithic structure → Evidence from 2+ distinct contexts/locations
- **Exception:** Single instance sufficient only if pattern unambiguous AND component is central/representative (document justification)

**Tool usage:**
- **Discovery tools (grep/codebase_search):** Identify WHERE to look, cannot be used as evidence
- **Verification tool (read_file):** Must read actual code from identified locations to verify patterns
- **Rule:** grep/codebase_search identify locations → read_file verifies with actual code

**Goal:** Distinguish patterns from outliers, mistakes, or isolated instances.

---

## VERIFICATION REQUIREMENTS

**MANDATORY VERIFICATION CHECKPOINT - MUST COMPLETE BEFORE MARKING [VERIFIED]**

**VERIFICATION PRINCIPLE:**
- See V1 - Verification Standard

**STEP-BY-STEP WORKFLOW:**

**STEP 1: Read code from multiple distinct contexts**
- Use read_file tool with specific offsets per V1
- Apply V1 verification count (default 2-3+ components, contextual 2+ contexts if monolithic/small)
- Example: `component_a:276; component_b:100` = 2 components ✓
- Example (monolithic): `component_a:276 (method X); component_a:450 (method Y)` = 2 distinct contexts ✓
- NOT: `component_a:276; component_a:299` = 1 component ✗

**STEP 2: List context names explicitly**
- Write out component/context names from read_file calls: [e.g., component_a, component_b OR component_a:method_X, component_a:method_Y]
- Count distinct components/contexts: [N]
- **CRITICAL:** Multiple offsets in same component = 1 component (unless distinct contexts in monolithic structure)
- **CRITICAL:** Must explicitly list names before proceeding

**STEP 3: Count distinct components/contexts**
- Count: [N] components/contexts (NOT instance count, NOT line number count)
- Apply V1 verification count requirement
- Component/context count insufficient per V1 → STOP. Mark [PARTIALLY VERIFIED] per S1 (unless V1 exception applies - document justification).
- Component/context count sufficient per V1 → Continue to STEP 4

**STEP 4: Check for violations FIRST**
- [ ] Checked for violations? (cross-layer dependencies violating architecture boundaries, expensive operations in loops)
  - NO → CANNOT mark [VERIFIED]. Check violations first.
  - YES → Continue

**STEP 5: Verify code evidence**
- [ ] Read actual code per V1 using read_file tool (NOT grep results, NOT codebase_search snippets)?
  - NO → CANNOT mark [VERIFIED] per S1. Must call read_file tool per V1.
  - YES → Continue

**STEP 6: Verify evidence completeness**
- [ ] Evidence includes: component:line references, component count (not instance count), pattern explanation, discovery path?
  - NO → CANNOT mark [VERIFIED]. Add missing evidence.
  - YES → Continue

**STEP 7: Check blocking issues**
- [ ] No violations found OR violations verified and documented in ARCHITECTURAL ISSUES?
  - NO → CANNOT mark [VERIFIED]. Verify violations first.
  - YES → Continue

**STEP 8: Decision**
- IF ALL STEPS PASS AND V1 verification complete → Mark [VERIFIED] per S1
- IF V1 exception applies (pattern unambiguous, component central/representative) → Mark [VERIFIED] per S1 with justification
- IF V1 verification incomplete → Mark [PARTIALLY VERIFIED] per S1
- OTHERWISE → Mark [PENDING] per S1

---

**When marking finding [VERIFIED], MUST collect internally (for AI verification) but show summary in output:**

**INTERNAL VERIFICATION (AI must collect, not shown to human):**
1. **Component names** - List all component names from read_file calls: [e.g., component_a, component_b, component_c]
2. **Component count** - Count distinct components (NOT instance count): [N] components
3. **Instances** - Full component:line references per V1 (e.g., `component_a:42,58,63,77; component_b:32,45,99`)
   - **CRITICAL:** Apply V1 verification count (2-3+ different components, not just different offsets in same component)
   - **CRITICAL:** Must explicitly list component names internally (e.g., "component_a, component_b" = 2 components)
   - NOT: `component_a:276; component_a:299` = 1 component → [VERIFIED] per S1
   - CORRECT: `component_a:276; component_b:100` = 2 components → [VERIFIED] per S1
4. **Discovery path** - HOW discovered with tool call references: "read_file calls #N, grep/codebase_search calls #M"
   - Natural: "Noticed while reading component:line_range" OR targeted: "Searched for pattern X"
5. **Verification confirmation** - "N different components confirmed via read_file"

**OUTPUT FORMAT (shown to human - concise summary only):**
- **Summary line**: "Description | Category | [VERIFIED] | N components, X instances"
- **Pattern line** (indented): "Pattern: concise explanation of the pattern/issue"
- DO NOT show detailed component lists, instance lists, or discovery paths in output
- Keep output concise to avoid cognitive overload for human
- Detailed evidence exists internally for AI verification but not displayed

**NOT sufficient for [VERIFIED] per S1:**
- V1 verification incomplete (without exception justification) → Use [PARTIALLY VERIFIED] per S1 instead
- Found 2+ instances from same component (without distinct context separation per V1) → Use [PARTIALLY VERIFIED] per S1 instead
- Noticed pattern but didn't search systematically
- Based on filenames/comments without reading code
- Assumed pattern without verification
- **Grep counts or codebase_search snippets alone** (per V1: these are discovery tools, not verification)
- **"Found N instances" without reading actual code using read_file tool per V1**
- **Reading multiple offsets in same component** (counts as 1 component - need V1 verification)
- **Seeing code snippets in codebase_search results** (must still call read_file per V1)
- **Not explicitly listing component names before marking [VERIFIED]** → Must list component names and count them
- **Skipping VERIFICATION CHECKPOINT** → Must complete all 8 steps above

**Sufficient for [PARTIALLY VERIFIED] per S1:**
- Found 1 component via read_file (even if multiple offsets from same component)
- Code evidence from read_file call per V1 (not grep/codebase_search snippets)
- Pattern explanation provided
- Component name: [e.g., component_a]
- Still needs V1 verification to reach [VERIFIED] per S1

---

## CATEGORY CHECKLISTS

**Use to ensure breadth of analysis:**

### C1 - Component Boundaries and Responsibilities

- ☑️ Component boundaries clear?
  → CHECK: Do components have single, well-defined responsibilities?
  → CHECK: Are boundaries between components explicit?
  → CHECK: Are unrelated concerns mixed in same component?

- ☑️ Dependency direction appropriate?
  → CHECK: Do dependencies flow in appropriate direction for your architecture?
  → CHECK: Are there circular dependencies?
  → CHECK: Do higher-level components depend on lower-level implementation details inappropriately?

- ☑️ Abstraction levels respected?
  → CHECK: Are implementation details hidden behind abstractions?
  → CHECK: Do components depend on interfaces/contracts rather than concrete implementations?
  → CHECK: Can implementation change without affecting consumers?

- ☑️ Concerns appropriately separated?
  → CHECK: Are core business logic concerns appropriately separated from infrastructure/interface concerns?
  → CHECK: Are core business logic components depending on infrastructure/interface types inappropriately?
  → CHECK: Are core business logic rules isolated from infrastructure/interface concerns?
  → NOTE: What "concerns" means depends on your architecture (layered, microservices, event-driven, functional, etc.)

- ☑️ Infrastructure concerns abstracted?
  → CHECK: Are infrastructure concerns (data access, external services, etc.) appropriately abstracted?
  → CHECK: Are there direct infrastructure calls scattered in core business logic?
  → CHECK: Are infrastructure operation boundaries clear?
  → NOTE: What "infrastructure" means depends on your architecture (database, APIs, message queues, file systems, etc.)

- ☑️ Code duplication across components?
  → CHECK: Is same logic repeated in multiple components?
  → CHECK: Are common patterns extracted to reusable components?
  → CHECK: Is core business logic duplicated unnecessarily?

- ☑️ Data structures appropriate?
  → CHECK: Do data structures contain only data, or do they also contain behavior?
  → CHECK: Is core business logic embedded in data structures where appropriate for your architecture?

### C2 - Consistency and Patterns

- ☑️ Error handling uniform?
  → CHECK: Do components handle errors consistently?
  → CHECK: Are error propagation patterns uniform?
  → CHECK: Are failures explicit (not hidden)?
  → NOTE: Error handling approach depends on your architecture (exceptions, result types, event-driven failures, etc.)

- ☑️ Infrastructure access patterns consistent?
  → CHECK: Is access to infrastructure (data, external services, etc.) consistent?
  → CHECK: Are there ad-hoc access patterns scattered throughout?
  → CHECK: Are infrastructure operation boundaries clear?
  → NOTE: What "infrastructure" means depends on your architecture (database, APIs, message queues, file systems, etc.)

- ☑️ Validation rules centralized or scattered?
  → CHECK: Is validation logic duplicated across components?
  → CHECK: Are validation patterns consistent?
  → CHECK: How are constraints validated? (grep for validation patterns, verify per V1)

- ☑️ Consistency boundaries appropriately managed?
  → CHECK: Are consistency/atomicity boundaries properly bounded for your architecture?
  → CHECK: Are consistency boundaries managed consistently?
  → NOTE: Consistency model depends on your architecture (transactions, sagas, eventual consistency, etc.)

- ☑️ Naming and conventions consistent?
  → CHECK: Are naming conventions followed?
  → CHECK: Are similar operations named/handled similarly?
  → CHECK: Are patterns applied uniformly?

### C3 - Pattern and Contract Verification

- ☑️ Contracts verified before use?
  → CHECK: Are contracts/interfaces verified to exist before using them?
  → CHECK: Are component boundaries verified (what's exposed, how accessed)?
  → CHECK: Are actual identifiers checked (naming varies by codebase)?

- ☑️ Patterns extracted from existing code?
  → CHECK: Are all existing components serving same role identified?
  → CHECK: Is the pattern structure extracted (how are they structured)?
  → CHECK: Are 2-3+ examples verified before applying pattern broadly?

- ☑️ New components follow established patterns?
  → CHECK: Do new components follow existing patterns when similar components exist?
  → CHECK: Are reference implementations studied before creating new components?
  → CHECK: Is pattern consistency verified per V1?

- ☑️ Contract modifications handled correctly?
  → CHECK: Is current contract verified before modification?
  → CHECK: Are all call sites identified before contract changes?
  → CHECK: Are contract changes implemented with all call sites together?

### P1 - Performance and Efficiency

- ☑️ Expensive operations optimized?
  → CHECK: Are expensive operations repeated unnecessarily?
  → CHECK: Are operations that access external resources/data repeated in loops?
  → CHECK: Could multiple accesses be batched together?

- ☑️ Data/resource loading efficient?
  → CHECK: Are related data/resource items loaded together when needed?
  → CHECK: Is loading strategy appropriate for your architecture? (eager/lazy/batch as needed)
  → CHECK: Is data/resource loading causing repeated access when it could be optimized?
  → NOTE: Loading strategy depends on your architecture (ORM lazy loading, graph DB traversal, API batching, etc.)

- ☑️ Large collections handled efficiently?
  → CHECK: Are large collections loaded without pagination/streaming?
  → CHECK: Are collections processed in batches when possible?
  → CHECK: Is memory usage appropriate for data volume?

- ☑️ Computations done at appropriate level?
  → CHECK: Are aggregations/computations done at the appropriate level for your architecture?
  → CHECK: Are computations repeated unnecessarily?
  → CHECK: Is work delegated to the appropriate layer for your architecture?
  → NOTE: Appropriate level depends on your architecture (database, service, client, etc.)

### S1 - Security

- ☑️ Input validation and sanitization patterns?
  → CHECK: Are all external inputs validated before use?
  → CHECK: Are validation patterns consistent across components?
  → CHECK: Are inputs sanitized to prevent injection attacks?
  → NOTE: Input sources vary by architecture (HTTP requests, file uploads, message queues, user input, etc.)

- ☑️ Authentication and authorization boundaries?
  → CHECK: Are authentication checks applied at appropriate boundaries?
  → CHECK: Are authorization rules enforced consistently?
  → CHECK: Are access control checks applied before operations?
  → NOTE: Security boundaries depend on architecture (API gateways, service boundaries, component boundaries, etc.)

- ☑️ Secrets management patterns?
  → CHECK: Are secrets stored securely (not hardcoded)?
  → CHECK: Are secrets accessed through secure mechanisms?
  → CHECK: Are secrets separated from configuration?
  → NOTE: Secrets include API keys, passwords, tokens, certificates, etc.

- ☑️ Output encoding patterns?
  → CHECK: Are outputs encoded to prevent injection attacks?
  → CHECK: Are encoding patterns consistent across components?
  → CHECK: Are security headers configured appropriately?
  → NOTE: Encoding prevents XSS, injection, and other output-based attacks.

### O1 - Observability

- ☑️ Logging patterns consistent?
  → CHECK: Are logging patterns uniform across components?
  → CHECK: Are log levels used appropriately?
  → CHECK: Is structured logging used where beneficial?
  → NOTE: Logging approach depends on architecture (structured logs, log aggregation, distributed tracing, etc.)

- ☑️ Metrics and instrumentation coverage?
  → CHECK: Are critical operations instrumented?
  → CHECK: Are metrics collected consistently?
  → CHECK: Are performance metrics tracked appropriately?
  → NOTE: Instrumentation varies by architecture (application metrics, business metrics, infrastructure metrics, etc.)

- ☑️ Error tracking and reporting?
  → CHECK: Are errors tracked and reported consistently?
  → CHECK: Are error contexts captured appropriately?
  → CHECK: Are error reporting patterns uniform?
  → NOTE: Error tracking includes exceptions, failures, and error aggregation.

- ☑️ Health check patterns?
  → CHECK: Are health checks implemented for critical components?
  → CHECK: Are health check patterns consistent?
  → CHECK: Do health checks verify actual functionality?
  → NOTE: Health checks enable monitoring and orchestration systems.

### CM1 - Configuration Management

- ☑️ Configuration access patterns consistent?
  → CHECK: Is configuration accessed through centralized mechanisms?
  → CHECK: Are configuration access patterns uniform?
  → CHECK: Are configuration values validated?
  → NOTE: Configuration includes settings, feature flags, environment-specific values, etc.

- ☑️ Environment-specific handling?
  → CHECK: Are environment-specific configurations handled appropriately?
  → CHECK: Are environment boundaries clear?
  → CHECK: Are environment configurations validated?
  → NOTE: Environments include development, staging, production, testing, etc.

- ☑️ Secrets vs configuration separation?
  → CHECK: Are secrets separated from configuration?
  → CHECK: Are secrets accessed through secure mechanisms?
  → CHECK: Are configuration and secrets managed differently?
  → NOTE: Secrets require secure storage and access, configuration may be version-controlled.

- ☑️ Configuration validation and defaults?
  → CHECK: Are configuration values validated at startup?
  → CHECK: Are default values provided appropriately?
  → CHECK: Are missing configuration errors handled clearly?
  → NOTE: Validation prevents runtime failures from misconfiguration.

### T1 - Testing Patterns

- ☑️ Test organization and structure?
  → CHECK: Are tests organized consistently?
  → CHECK: Do test structures mirror component structures?
  → CHECK: Are test boundaries clear?
  → NOTE: Test organization depends on architecture (unit tests, integration tests, end-to-end tests, etc.)

- ☑️ Test coverage of critical paths?
  → CHECK: Are critical business logic paths tested?
  → CHECK: Are error handling paths tested?
  → CHECK: Are edge cases covered?
  → NOTE: Critical paths include core functionality, failure modes, and boundary conditions.

- ☑️ Testability patterns?
  → CHECK: Are components designed for testability?
  → CHECK: Are dependencies injectable or mockable?
  → CHECK: Are test doubles used appropriately?
  → NOTE: Testability enables isolated testing and reduces test complexity.

- ☑️ Test data management?
  → CHECK: Is test data managed consistently?
  → CHECK: Are test fixtures reusable?
  → CHECK: Is test data isolated between tests?
  → NOTE: Test data management includes fixtures, factories, and test databases.

### DM1 - Dependency Management

- ☑️ Dependency hygiene?
  → CHECK: Are unused dependencies removed?
  → CHECK: Are dependencies up to date?
  → CHECK: Are dependency versions pinned appropriately?
  → NOTE: Dependencies include external libraries, frameworks, and internal components.

- ☑️ Dependency injection patterns?
  → CHECK: Are dependencies injected rather than created internally?
  → CHECK: Are dependency injection patterns consistent?
  → CHECK: Are dependency boundaries clear?
  → NOTE: Dependency injection improves testability and flexibility.

- ☑️ Circular dependency detection?
  → CHECK: Are circular dependencies avoided?
  → CHECK: Are dependency cycles detected and resolved?
  → CHECK: Are dependency directions appropriate?
  → NOTE: Circular dependencies indicate architectural boundary violations.

- ☑️ External dependency boundaries?
  → CHECK: Are external dependencies abstracted appropriately?
  → CHECK: Are external dependency failures handled?
  → CHECK: Are external dependencies versioned and managed?
  → NOTE: External dependencies include third-party services, APIs, and libraries.

### RM1 - Resource Management

- ☑️ Resource cleanup patterns?
  → CHECK: Are resources cleaned up appropriately?
  → CHECK: Are cleanup patterns consistent?
  → CHECK: Are resources released even on errors?
  → NOTE: Resources include connections, file handles, memory, locks, etc.

- ☑️ Connection pooling strategies?
  → CHECK: Are connections pooled where beneficial?
  → CHECK: Are pooling strategies consistent?
  → CHECK: Are connection limits configured appropriately?
  → NOTE: Connection pooling applies to databases, HTTP clients, message queues, etc.

- ☑️ Resource lifecycle management?
  → CHECK: Are resource lifecycles managed explicitly?
  → CHECK: Are resource creation and destruction paired?
  → CHECK: Are resource lifetimes bounded appropriately?
  → NOTE: Resource lifecycle includes initialization, usage, and cleanup phases.

- ☑️ Timeout and retry patterns?
  → CHECK: Are timeouts configured for external operations?
  → CHECK: Are retry patterns consistent?
  → CHECK: Are timeout and retry strategies appropriate?
  → NOTE: Timeouts and retries prevent resource exhaustion and improve resilience.

### CT1 - Concurrency and Threading

- ☑️ Thread safety patterns?
  → CHECK: Are shared resources accessed safely?
  → CHECK: Are thread safety mechanisms consistent?
  → CHECK: Are race conditions prevented?
  → NOTE: Thread safety applies to shared state, concurrent access, and parallel execution.

- ☑️ Locking strategies?
  → CHECK: Are locks used appropriately?
  → CHECK: Are locking strategies consistent?
  → CHECK: Are deadlocks prevented?
  → NOTE: Locking includes mutexes, semaphores, read-write locks, etc.

- ☑️ Async/await patterns?
  → CHECK: Are async patterns used consistently?
  → CHECK: Are async operations handled appropriately?
  → CHECK: Are async error handling patterns uniform?
  → NOTE: Async patterns vary by language and architecture (promises, futures, coroutines, etc.).

- ☑️ Concurrent access patterns?
  → CHECK: Are concurrent access patterns safe?
  → CHECK: Are concurrent operations coordinated appropriately?
  → CHECK: Are concurrent failures handled?
  → NOTE: Concurrent access includes parallel processing, concurrent requests, and shared state access.

### API1 - API Design

- ☑️ API versioning strategies?
  → CHECK: Are APIs versioned appropriately?
  → CHECK: Are versioning strategies consistent?
  → CHECK: Are version transitions handled?
  → NOTE: API versioning includes URL versioning, header versioning, and semantic versioning.

- ☑️ Backward compatibility handling?
  → CHECK: Are API changes backward compatible?
  → CHECK: Are breaking changes handled appropriately?
  → CHECK: Are deprecation strategies clear?
  → NOTE: Backward compatibility enables gradual migration and reduces client disruption.

- ☑️ Request and response validation?
  → CHECK: Are requests validated before processing?
  → CHECK: Are responses validated before sending?
  → CHECK: Are validation patterns consistent?
  → NOTE: Validation includes type checking, schema validation, and constraint checking.

- ☑️ Error response consistency?
  → CHECK: Are error responses formatted consistently?
  → CHECK: Are error codes used appropriately?
  → CHECK: Are error messages helpful?
  → NOTE: Error responses should be predictable and actionable for API consumers.

- ☑️ Rate limiting patterns?
  → CHECK: Are rate limits applied appropriately?
  → CHECK: Are rate limiting strategies consistent?
  → CHECK: Are rate limit errors handled clearly?
  → NOTE: Rate limiting prevents abuse and ensures fair resource usage.

### SM1 - State Management

- ☑️ Global state usage patterns?
  → CHECK: Is global state used minimally?
  → CHECK: Are global state access patterns consistent?
  → CHECK: Is global state thread-safe?
  → NOTE: Global state includes singletons, module-level variables, and shared mutable state.

- ☑️ Shared mutable state patterns?
  → CHECK: Is shared mutable state minimized?
  → CHECK: Are shared state access patterns safe?
  → CHECK: Are state mutations coordinated?
  → NOTE: Shared mutable state increases complexity and risk of bugs.

- ☑️ Immutability patterns?
  → CHECK: Are immutable data structures used where beneficial?
  → CHECK: Are immutability patterns consistent?
  → CHECK: Are state updates handled through immutability?
  → NOTE: Immutability reduces bugs and improves reasoning about code.

- ☑️ State synchronization?
  → CHECK: Is state synchronized appropriately?
  → CHECK: Are synchronization patterns consistent?
  → CHECK: Are state conflicts handled?
  → NOTE: State synchronization includes cache invalidation, state replication, and consistency guarantees.

### DOC1 - Documentation

- ☑️ Code documentation coverage?
  → CHECK: Are complex components documented?
  → CHECK: Are non-obvious behaviors explained?
  → CHECK: Are documentation patterns consistent?
  → NOTE: Documentation includes comments, docstrings, and inline explanations.

- ☑️ API documentation completeness?
  → CHECK: Are APIs documented with contracts?
  → CHECK: Are API examples provided?
  → CHECK: Are API changes documented?
  → NOTE: API documentation includes request/response formats, error codes, and usage examples.

- ☑️ Architecture documentation?
  → CHECK: Are architectural decisions documented?
  → CHECK: Are component relationships documented?
  → CHECK: Are design patterns explained?
  → NOTE: Architecture documentation includes ADRs, diagrams, and design explanations.

- ☑️ README and setup documentation?
  → CHECK: Are setup instructions clear?
  → CHECK: Are dependencies documented?
  → CHECK: Are usage examples provided?
  → NOTE: README documentation enables new contributors and users.

---

## OUTPUT

**TRACKER DISPLAY RULES:**
- **During A1.1 ANALYZE:** Show unified tracker (analyzed + findings + issues) after EACH analysis cycle (with menu)
- **During A1.2 FIX:** Show tracker with fixes being applied
- **During A1.3 VERIFY:** Show tracker with verification results

**AFTER each analysis cycle, show:**

1. **Current Scope Indicator** - Show: "ANALYSIS CYCLE N: [scope description]" (e.g., "ANALYSIS CYCLE 1: Initial architecture review", "ANALYSIS CYCLE 2: Data access patterns")
2. **Tracker** - Unified tracker showing ANALYZED + DEPENDENCY GRAPH + FINDINGS + ARCHITECTURAL ISSUES + FIXES + VERIFIED FIXES
   - Format tracker with clear section headers and visual separation
   - Use icons/emojis to highlight key findings (✅ verified, ⚠️ issues, 🔍 to investigate)
   - **DEPENDENCY GRAPH:** Show component relationships (component_a → component_b) with violation markers. Graph is shown BY DEFAULT after each analysis cycle. Mark edges with [violation type] when dependencies violate architecture, [OK] when valid. Graph builds incrementally as components are analyzed.
   - **FINDINGS entries:** Summary line with component/instance counts, then concise pattern explanation on indented line (detailed evidence collected internally by AI but not shown to avoid cognitive overload)
   - **ARCHITECTURAL ISSUES:** Format with Impact, Found at, and Fix strategy on separate indented lines for readability
   - **ANALYZED:** List components/areas already analyzed. Show what categories were checked.
3. **Evidence gathered** - Brief summary of what was verified this cycle (1-2 sentences, not duplication of tracker)
   - Use 🔍 icon for discovery, ✅ for verification
4. **Next proposal** - Plan/scope for next analysis cycle
   - What NEW areas to analyze next
   - Present as PLAN, not OPTIONS
   - ARCHITECTURAL ISSUES exist? → Note impact, suggest fixing (use ⚠️ icon)
5. **Menu** - Format clearly with visual separation
   - **Show menu:** After EACH analysis cycle (always available during A1.1), during A1.2, during A1.3
   - **Context-driven:** Show ONLY options for current phase (A1.1 shows c/k/f/x, A1.2 shows c/v/x, A1.3 shows c/d/x)
   - **Title:** "**Menu:**" (no phase prefix)

**OUTPUT RULES:**
- Tracker = single source of truth - AI collects detailed evidence internally for verification, but shows concise summary to human
- Show unified tracker after EACH analysis cycle (with menu)
- **DEPENDENCY GRAPH:** Always shown by default after each analysis cycle. Graph shows component relationships discovered during analysis. Edges marked with violation types when dependencies violate architecture. Graph builds incrementally - new components and relationships added each cycle.
- FINDINGS entries: Summary line (component/instance counts) + concise pattern explanation (detailed evidence collected internally by AI but not displayed)
- ARCHITECTURAL ISSUES: Use formatted layout with Impact and Found at on separate indented lines
- "Evidence gathered" = brief summary (1-2 sentences) of new verifications this cycle
- "Next proposal" references NEW areas to analyze, does not duplicate findings
- DO NOT show checklist iteration or duplicate tracker content
- **Flow:** Analysis cycles continue until user chooses "f" to fix - menu always available
- **CRITICAL:** AI must still complete all verification steps internally per V1 (document discovery paths, etc.) even though detailed evidence is not shown in output
- **CRITICAL:** Graph must be built and shown by default - no menu option needed, always displayed
- **CRITICAL:** DO NOT generate markdown reports or documentation files - tracker output only
- **CRITICAL:** DO NOT show code snippets in output - use prose descriptions only (e.g., "component_a calls component_b at line 42", not code blocks)

**Menu enforcement (BEFORE showing menu):**
- [ ] Am I showing ONLY options for current phase?
  - NO → CANNOT proceed. Show only options for current phase (A1.1: c/k/f/x, A1.2: c/v/x, A1.3: c/d/x).
  - YES → Evidence: [Menu shows only current phase options]
- [ ] Menu title is "**Menu:**" (no phase prefix)?
  - NO → CANNOT proceed. Use "**Menu:**" as title, not "During... Menu:".
  - YES → Evidence: [Title format correct]
- [ ] Will I generate markdown reports or documentation files?
  - YES → CANNOT proceed. DO NOT generate markdown reports or documentation files.
  - NO → Evidence: [No markdown reports will be generated]
- [ ] Will I show code snippets in output?
  - YES → CANNOT proceed. DO NOT show code snippets. Use prose descriptions only.
  - NO → Evidence: [No code snippets will be shown]

---

## MENU

**Menu Display (show after each cycle):**

Format menu with clear visual separation and icons. Show ONLY options for current phase:

**During A1.1 ANALYZE:**
```
---

**Menu:**
- 🔍 **c** - continue: Analyze NEW areas of codebase
- ✅ **k** - check: Targeted category checklist verification in NEW areas (e.g., "k C3", "k S1", "k all")
- 🔧 **f** - fix: Start fixing architectural issues
- ❌ **x** - exit: Complete analysis

... or anything else?

---
```

**During A1.2 FIX:**
```
---

**Menu:**
- ➡️ **c** - continue: Continue fixing issues
- ✅ **v** - verify: Check fixes against architectural patterns
- ❌ **x** - exit: Stop fixing

... or anything else?

---
```

**During A1.3 VERIFY:**
```
---

**Menu:**
- ➡️ **c** - continue: More verification
- ✅ **k** - check: Targeted category checklist verification (e.g., "k C3", "k S1", "k all")
- ✨ **d** - done: Verification complete
- ❌ **x** - exit: Stop verification

... or anything else?

---
```

**Menu Formatting Rules:**
- Show menu after EACH analysis cycle (always available during A1.1), during A1.2, during A1.3
- Show ONLY options for current phase (context-driven)
- Use horizontal rule (`---`) before and after menu for visual separation
- Use consistent icons for each menu option
- Bold the command letter for quick scanning
- Title: "**Menu:**" (no phase prefix in title)

**Menu Options:**

**c - continue:**
- Available: After EACH analysis cycle (during A1.1), during A1.2, during A1.3
- A1.1: Analyze NEW areas/components (widens scope: new areas, NOT re-checking same areas)
- A1.2: Continue fixing code
- A1.3: Continue verification checks

**BEFORE responding to "c" during A1.1:**
- [ ] Will I analyze NEW areas/components (not yet in ANALYZED section)?
  - NO → CANNOT proceed. Analyze NEW areas only.
  - YES → Evidence: [Specific NEW areas listed]
- [ ] Will I re-evaluate previous conclusions if new evidence contradicts them?
  - NO → CANNOT proceed. Must re-evaluate previous findings when new evidence from NEW areas contradicts them.
  - YES → Evidence: [Will review existing FINDINGS and ARCHITECTURAL ISSUES, update if new evidence contradicts]
- [ ] Will I re-analyze already analyzed areas without new evidence?
  - YES → CANNOT proceed. Only analyze NEW areas, reuse existing findings unless new evidence warrants re-evaluation.
  - NO → Evidence: [Will only analyze NEW areas, reuse existing findings unless new evidence contradicts them]
- [ ] Will I update ANALYZED section with new areas?
  - NO → CANNOT proceed. Must update ANALYZED section.
  - YES → Evidence: [Will update ANALYZED section]

**k - check:**
- Available: During A1.1 ANALYZE and A1.3 VERIFY
- Purpose: Explicitly trigger targeted checklist category verification in NEW areas
- Usage: User specifies category(ies) to check (e.g., "k C3", "k S1", "k C2,C3", "k all")
- Behavior:
  1. Identify NEW areas not yet analyzed for specified category(ies)
  2. Perform verification per V1 standard (grep/codebase_search → read_file)
  3. Document findings in tracker with proper status ([VERIFIED], [PARTIALLY VERIFIED], [VIOLATION])
  4. Show summary of what was checked and what was found
  5. Update ANALYZED section with areas checked
  6. Return to menu for next action
- Does NOT re-check already analyzed areas unless explicitly requested
- Results feed into same tracker as other analysis cycles

**BEFORE responding to "k" command:**
- [ ] Did user specify which category(ies) to check?
  - NO → Prompt user: "Which category to check? (C1, C2, C3, P1, S1, O1, CM1, T1, DM1, RM1, CT1, API1, SM1, DOC1, or 'all')"
  - YES → Continue
- [ ] Will I check NEW areas not yet analyzed?
  - NO → CANNOT proceed. Must check NEW areas only.
  - YES → Evidence: [Will check NEW areas]
- [ ] Will I perform verification per V1 standard?
  - NO → CANNOT proceed. Must use V1 verification (grep/codebase_search → read_file)
  - YES → Evidence: [Will use V1 verification standard]
- [ ] Will I document findings in tracker?
  - NO → CANNOT proceed. All findings must be documented in tracker.
  - YES → Evidence: [Findings will be added to tracker]

**f - fix:**
- Available: During A1.1
- Invokes: A1.2 FIX phase

**BEFORE responding to "f" command:**
- [ ] Will I fix violations from ARCHITECTURAL ISSUES section?
  - NO → CANNOT proceed. Fix violations from ARCHITECTURAL ISSUES section.
  - YES → Evidence: [Will fix violations from ARCHITECTURAL ISSUES]
  
**THEN:**
- Show summary of violations to fix from ARCHITECTURAL ISSUES section
- Proceed to A1.2 FIX phase

**v - verify:**
- Available: During A1.2
- Invokes: A1.3 VERIFY phase

**d - done:**
- Available: During A1.3 when verification complete
- Invokes: Exit protocol

**x - exit:**
- Available: During all phases
- Invokes: Exit protocol

---

## FREE-FORM HANDLING

**PRINCIPLE:** Free-form feedback stays within protocol, not outside it.

**BEFORE responding to free-form feedback:**

- [ ] Will I stay in protocol?
  - NO → CANNOT proceed. Stay in protocol with menu.
  - YES → Evidence: [Response will show protocol header + menu]

- [ ] Will I re-show menu after response?
  - NO → CANNOT proceed. Menu must be re-shown after free-form response.
  - YES → Evidence: [Menu will appear as last element]

**BLOCKING RULE:** Free-form discussion does NOT bypass protocol. Menu must be re-shown to maintain protocol state.

---

*Last Updated: January 2025 (v1.6 - Added architectural alternatives evaluation to fix strategy planning in A1.1 ANALYZE phase)*
