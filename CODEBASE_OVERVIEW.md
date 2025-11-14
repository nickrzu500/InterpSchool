# AI Safety Learning Platform - Complete Codebase Overview

## Quick Navigation

This document serves as an index for understanding the AI Safety Learning Platform codebase.

### Documentation Files Created
1. **LESSON_STRUCTURE_ANALYSIS.md** - Comprehensive analysis of project structure, lessons, and what's missing
2. **LESSON_TEMPLATE_GUIDE.md** - How to create new lessons with detailed examples
3. **CODEBASE_OVERVIEW.md** - This file, serving as your navigation hub

---

## The Three Questions You Need Answered

### 1. "What lessons exist and where are they?"
See: **LESSON_STRUCTURE_ANALYSIS.md**, Section 3 (Lesson Content Statistics)
- 18 complete lessons with 335 interactive steps
- Stored in: `/home/user/InterpSchool/lessons-content.js`
- Lines provided for each lesson location

### 2. "What's under construction?"
See: **LESSON_STRUCTURE_ANALYSIS.md**, Section 4 (Under Construction)
- 1 full module: Developmental Interpretability
- 5 individual lessons: training-dynamics, efficient-inference, model-compression, circuit-analysis, jailbreaking-intro
- All have HTML placeholders but no content in lessons-content.js

### 3. "How does the lesson system work?"
See: **LESSON_STRUCTURE_ANALYSIS.md**, Sections 5-6 (Organization & Connection to HTML)
- Navigation hierarchy: Home → Module → Difficulty → Lesson → Steps
- Data flow: lessons-content.js → HTML → JavaScript → DOM rendering
- Multiple exercise types (copy, fill-in, multiple-choice, construct, reflection, ordering)

---

## Project Statistics at a Glance

| Metric | Value |
|--------|-------|
| **Total Files** | 3 main files |
| **Total Lessons** | 18 complete + 6 under construction |
| **Completed Steps** | 335 interactive steps |
| **Average Steps/Lesson** | 17.6 |
| **Code Completion** | ~91% complete (18/24 planned lessons) |

---

## File Structure

```
/home/user/InterpSchool/
│
├── ai-safety-platform.html        (68.6 KB)
│   ├── All UI/UX (HTML + CSS)
│   ├── Navigation logic (JavaScript)
│   ├── Step rendering (updateStep())
│   ├── Code validation (checkCode())
│   └── Simple router for URL navigation
│
├── lessons-content.js             (433 KB)
│   └── LESSONS object containing all 18 lesson definitions
│       ├── Basic Transformer (6 lessons)
│       ├── Intermediate Implementation (7 lessons)
│       ├── Basic Interpretability (5 lessons)
│       └── Bonus (1 lesson)
│
├── platform-readme.md
│   └── Project documentation
│
├── LESSON_STRUCTURE_ANALYSIS.md   (NEW)
│   └── Complete analysis document
│
├── LESSON_TEMPLATE_GUIDE.md       (NEW)
│   └── How to create new lessons
│
├── CODEBASE_OVERVIEW.md           (NEW - this file)
│   └── Navigation guide
│
└── .git/
    └── Version control with 17 commits
```

---

## How to Use This Documentation

### For Understanding the Project
1. Start with this file (CODEBASE_OVERVIEW.md)
2. Read LESSON_STRUCTURE_ANALYSIS.md for detailed information
3. Reference LESSON_TEMPLATE_GUIDE.md when adding content

### For Adding New Lessons
1. Read LESSON_TEMPLATE_GUIDE.md (complete examples)
2. Follow the structure shown in Section 1
3. Add lesson ID to goBackToLessons() function in HTML
4. Add navigation card to appropriate HTML section
5. Test with URL: `?lesson=your-lesson-id`

### For Modifying Existing Lessons
1. Find lesson in lessons-content.js using line numbers in LESSON_STRUCTURE_ANALYSIS.md
2. Edit the steps array
3. Test changes in browser
4. Commit to git with descriptive message

### For Understanding HTML/JS Connection
See LESSON_STRUCTURE_ANALYSIS.md, Section 6:
- Shows exact HTML element IDs that get populated
- Shows JavaScript functions that handle the flow
- Shows data flow from LESSONS object to DOM

---

## Lesson Organization Map

### Complete (Ready to Use)
```
Transformer from Scratch
├── BASIC (6 lessons, 131 steps)
│   ├── Tokenization & Text Processing (26 steps)
│   ├── Embeddings & Positional Encoding (21 steps)
│   ├── Attention Mechanism Basics (31 steps)
│   ├── MLP Layers (22 steps)
│   ├── Putting It All Together (15 steps)
│   └── Text Generation (16 steps)
│
└── INTERMEDIATE (7 lessons, 135 steps)
    ├── LayerNorm Implementation (23 steps)
    ├── Embedding & Positional Layers (26 steps)
    ├── Attention Implementation (28 steps)
    ├── MLP Implementation (19 steps)
    ├── Transformer Blocks (13 steps)
    ├── Complete Transformer Model (14 steps)
    └── Sampling Methods for AI Safety (12 steps)

Basic Interpretability (5 lessons, 60 steps)
├── Visualizing Attention Patterns (14 steps)
├── The Logit Lens (13 steps)
├── Activation Analysis (12 steps)
├── Simple Probing (11 steps)
└── Finding Safety-Relevant Features (10 steps)

BONUS: Gradient Flow Visualization (9 steps)
```

### Under Construction
```
Transformer from Scratch
└── ADVANCED (0/3 lessons, 0 steps)
    ├── Training Dynamics & Optimization [NO CONTENT]
    ├── Efficient Inference & Scaling [NO CONTENT]
    └── Model Compression & Deployment [NO CONTENT]

Advanced Interpretability (0/1 lessons, 0 steps)
└── Circuit Analysis & Mechanistic Interpretability [NO CONTENT]

Advanced: Adversarial Robustness (0/1 lessons, 0 steps)
└── Understanding Jailbreaking & Prompt Injection [NO CONTENT]

Developmental Interpretability (0/? lessons, 0 steps)
└── [NO LESSON IDS DEFINED]
```

---

## Key Technical Details

### Lesson Data Structure
Each lesson in LESSONS object has:
- `title` - Human readable title
- `steps[]` - Array of step objects

Each step has:
- `instruction` - What to do
- `code` - Code they should produce
- `explanation` - Why it matters
- `why` (optional) - AI safety context
- `type` (optional) - Exercise type (default: "copy")
- `expectedOutput` (optional) - Console output to show
- Type-specific properties (template, choices, etc.)

### Exercise Types
1. **copy** - Type exact code shown
2. **fill-in** - Complete code with blanks (___)
3. **multiple-choice** - Select option + type complete line
4. **construct** - Write code from description
5. **reflection** - Conceptual reflection
6. **ordering** - Order items correctly

### Navigation Flow
```
showWelcome()
    ↓
showAgenda(type)          [Module selection]
    ↓
showDifficulty(agenda, difficulty)  [Difficulty selection]
    ↓
showLesson(lessonId)      [Start lesson]
    ↓
updateStep()              [Render current step]
    ↓
checkCode()               [Validate user input]
    ↓
nextStep()                [Go to next step]
    ↓
showBadge()               [Completion screen]
```

---

## Development Guidelines

### When to Reference Each Document

| Task | Document | Section |
|------|----------|---------|
| Understand overall structure | LESSON_STRUCTURE_ANALYSIS.md | 1 |
| Find specific lesson content | LESSON_STRUCTURE_ANALYSIS.md | 10 |
| Know what's missing | LESSON_STRUCTURE_ANALYSIS.md | 4 |
| Create new lesson | LESSON_TEMPLATE_GUIDE.md | "How to Add" |
| Understand HTML/JS connection | LESSON_STRUCTURE_ANALYSIS.md | 6-7 |
| Learn exercise types | LESSON_TEMPLATE_GUIDE.md | "Step Properties Reference" |
| Debug navigation issues | LESSON_STRUCTURE_ANALYSIS.md | 5 |
| Modify existing content | lessons-content.js + line numbers from LESSON_STRUCTURE_ANALYSIS.md | 10 |

---

## Quick Reference: Lesson IDs

**Lesson ID Format**: kebab-case (lowercase with hyphens)

Examples:
- `tokenization-basics`
- `attention-mechanism`
- `sampling-methods-safety`

Use these IDs in:
1. `lessons-content.js` as object keys
2. `showLesson()` function calls in HTML
3. `goBackToLessons()` array lists
4. URL parameters: `?lesson=tokenization-basics`

---

## Next Steps

### If you want to...

**Understand how lessons work:**
→ Read LESSON_STRUCTURE_ANALYSIS.md (20-30 min read)

**Add a new lesson:**
→ Read LESSON_TEMPLATE_GUIDE.md (10-15 min)
→ Use template structure
→ Add to lessons-content.js
→ Update navigation in HTML

**Fix a specific lesson:**
→ Find line number in LESSON_STRUCTURE_ANALYSIS.md Section 10
→ Edit lessons-content.js at that location
→ Test in browser

**Add a new module:**
→ Create HTML section in ai-safety-platform.html
→ Create lesson definitions in lessons-content.js
→ Add navigation links
→ Link from home page

**Enable real code execution:**
→ Consider Pyodide (Python in WebAssembly)
→ Or add backend API for execution
→ Currently using simulated/mock execution

**Add progress tracking:**
→ Use localStorage for browser persistence
→ Or implement backend with user accounts

---

## File Statistics

| File | Size | Type | Purpose |
|------|------|------|---------|
| ai-safety-platform.html | 68.6 KB | HTML/CSS/JS | UI + Logic |
| lessons-content.js | 433 KB | JavaScript | All lesson content |
| platform-readme.md | 1.9 KB | Markdown | Project docs |
| LESSON_STRUCTURE_ANALYSIS.md | ~50 KB | Markdown | Analysis (NEW) |
| LESSON_TEMPLATE_GUIDE.md | ~15 KB | Markdown | Template guide (NEW) |

**Total Codebase Size**: ~550 KB (very lightweight!)

---

## Contact & Attribution

This platform is:
- Built with human + AI collaboration (Anthropic's Claude)
- Inspired by ARENA (AI Alignment Research Engineers Accelerator)
- Free and open educational resource
- MIT License

For more info, see the "About" page in the platform or platform-readme.md

---

## Last Updated
November 12, 2025

## Version
Based on git commit: 157e81a (HTML-JS connection fixes)

---

**Start here for navigation: LESSON_STRUCTURE_ANALYSIS.md**
