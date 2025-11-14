# Lesson Structure Template - Complete Example

## How to Add a New Lesson

Here's what a complete lesson looks like in `lessons-content.js`:

```javascript
'lesson-id-example': {
    title: "Your Lesson Title Here",
    steps: [
        // STEP 1: Simple copy exercise
        {
            instruction: "What the user should do. Be clear and specific.",
            why: "Why this matters for AI safety. Include context about risks, implications, or relevance to the field.",
            code: "your_code_here = 42",
            explanation: "Why this code works. What it demonstrates.",
            type: "copy"  // Default type
        },
        
        // STEP 2: Fill-in-the-blank exercise
        {
            instruction: "Complete the code by filling in the blanks:",
            code: "x = ___\ny = ___\nprint(x + y)",  // This is what they need to produce
            explanation: "Explanation of what the code does.",
            type: "fill-in",
            template: "x = ___\ny = ___\nprint(x + y)",  // Same as code usually
            hints: [
                "The first blank should be a number",
                "The second blank should also be a number"
            ]
        },
        
        // STEP 3: Multiple choice exercise
        {
            instruction: "Which function is used to print output?",
            code: "output = ___",  // Their task is to fill this
            explanation: "The print() function is fundamental to Python output.",
            type: "multiple-choice",
            template: "output = ___",
            choices: [
                "print()",
                "output()",
                "display()",
                "show()"
            ],
            correct: 0,  // Index of correct answer (0-based)
            hint: "Think about what we used to display text earlier"
        },
        
        // STEP 4: Construct exercise
        {
            instruction: "Write a function that checks if a number is positive",
            description: "Create a function called 'is_positive' that takes a number and returns True if positive, False otherwise.",
            explanation: "Functions are reusable blocks of code essential for larger programs.",
            type: "construct",
            template: "def is_positive(x):\n    return ___",
            hints: [
                "Compare the number to 0",
                "Use the > operator"
            ]
        },
        
        // STEP 5: With expected output
        {
            instruction: "Print the result:",
            code: "result = 2 + 2\nprint(f'Result: {result}')",
            explanation: "F-strings allow embedding variables directly in strings.",
            expectedOutput: "Result: 4",  // This will be displayed
            type: "copy"
        },
        
        // STEP 6: Reflection (conceptual)
        {
            instruction: "Reflect on what you've learned",
            code: "# Reflection - no code to type",
            explanation: "You've learned how Python handles arithmetic, variables, and output. These foundations are crucial for AI safety applications where precise computation is essential.",
            type: "reflection",
            prompts: [
                "How would incorrect arithmetic affect AI model training?",
                "Why is output validation important in safety-critical systems?"
            ]
        }
    ]
}
```

## Step Properties Reference

### Required Properties (all steps)
- **instruction** (string): What the user should do
- **code** (string): The final code result they should produce
- **explanation** (string): Why this code/concept matters

### Optional Properties (any step)
- **why** (string): Deep context on AI safety implications (appears in blue box)
- **type** (string): Exercise type - see below. Default: "copy"
- **expectedOutput** (string): Console output to display for validation

### Type-Specific Properties

#### type: "copy"
User types the exact code shown. No additional properties needed.

#### type: "fill-in"
```javascript
{
    template: "code with ___ blanks",     // Required
    hints: ["hint1", "hint2", "hint3"]    // Optional, one per blank
}
```

#### type: "multiple-choice"
```javascript
{
    template: "code with ___ to fill",    // Required
    choices: ["option a", "option b"],    // Required array
    correct: 0,                           // Required index (0-based)
    hint: "hint text"                     // Optional
}
```

#### type: "construct"
```javascript
{
    description: "Write a function that...", // Required
    template: "def ___():\n    return ___",  // Required partial code
    hints: ["hint 1", "hint 2"]            // Optional
}
```

#### type: "reflection"
```javascript
{
    prompts: [                            // Optional, for discussion
        "Question 1?",
        "Question 2?"
    ]
}
```

#### type: "ordering"
```javascript
{
    items: ["item 1", "item 2"],          // Required
    correct_order: [0, 1],                // Required indices in order
    feedback: "explanation"               // Optional
}
```

## Real Example: Finding a Lesson in the Codebase

### Tokenization Lesson Example (Excerpt)

Location: `/home/user/InterpSchool/lessons-content.js` lines 8-180

```javascript
'tokenization-basics': {
    title: "Tokenization & Text Processing",
    steps: [
        {
            instruction: "Let's start by understanding what tokenization is. First, import the transformers library:",
            why: "Tokenization is the foundation of how AI models understand text. Without it, models would have to process raw characters or entire words, which would be inefficient...",
            code: "from transformers import GPT2TokenizerFast",
            explanation: "The transformers library gives us access to tokenizers - the tools that convert text into numbers that AI models can understand."
        },
        {
            instruction: "Create a tokenizer instance for GPT-2:",
            why: "Each model family has its own tokenizer trained on specific data. Using the wrong tokenizer is like speaking the wrong language to the model...",
            code: "tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')",
            explanation: "This loads the same tokenizer that GPT-2 was trained with..."
        },
        // ... 24 more steps
    ]
}
```

## How the Lesson is Used in HTML

1. **Navigation**: User clicks card, triggers `showLesson('tokenization-basics')`

2. **Display**: HTML `#lesson-page` becomes visible

3. **Step Rendering**: `updateStep()` function:
   ```javascript
   const currentLesson = LESSONS['tokenization-basics'];
   const step = currentLesson.steps[0];  // First step
   
   document.getElementById('lesson-title').textContent = 
       currentLesson.title;  // "Tokenization & Text Processing"
   
   document.getElementById('step-instruction').textContent = 
       step.instruction;  // "Let's start by understanding..."
   
   document.getElementById('code-example').textContent = 
       step.code;  // "from transformers import..."
   ```

4. **Code Checking**: When user clicks "Check Code":
   ```javascript
   // Get what user typed
   const userCode = document.getElementById('code-editor').value;
   
   // Build expected code (accumulate all steps up to current)
   let expectedCode = '';
   for (let i = 0; i <= currentStep; i++) {
       expectedCode += currentLesson.steps[i].code + '\n';
   }
   
   // Compare and validate
   if (normalizeCode(userCode) === normalizeCode(expectedCode)) {
       // Show success!
   } else {
       // Show error with hints
   }
   ```

## Adding a Lesson to Navigation

1. **In lessons-content.js**: Add lesson definition

2. **In goBackToLessons() function** (line ~1263):
   ```javascript
   const newLessons = ['your-new-lesson-id'];
   
   if (newLessons.includes(currentLessonId)) {
       // Navigate back to appropriate page
       showDifficulty('module-name', 'difficulty');
   }
   ```

3. **In HTML navigation** (e.g., intermediate-lessons-page):
   ```html
   <div class="card" onclick="showLesson('your-new-lesson-id')">
       <h2 class="card-title">Your Lesson Title</h2>
       <p class="card-description">Brief description of what you'll learn.</p>
       <p class="card-meta">XX minutes</p>
   </div>
   ```

## Quick Reference: Lesson IDs to HTML Locations

| Lesson ID | HTML Location | Line Numbers |
|-----------|---------------|--------------|
| tokenization-basics | basic-lessons-page | 716-722 |
| embeddings-positional | basic-lessons-page | 724-730 |
| attention-mechanism | basic-lessons-page | 732-738 |
| mlp-layers | basic-lessons-page | 740-746 |
| complete-transformer-basic | basic-lessons-page | 748-754 |
| text-generation | basic-lessons-page | 756-762 |
| layernorm-implementation | intermediate-lessons-page | 776-782 |
| embedding-layers | intermediate-lessons-page | 784-790 |
| attention-implementation | intermediate-lessons-page | 792-798 |
| mlp-implementation | intermediate-lessons-page | 800-806 |
| transformer-blocks | intermediate-lessons-page | 808-814 |
| complete-transformer | intermediate-lessons-page | 816-822 |
| sampling-methods-safety | intermediate-lessons-page | 824-830 |
| attention-patterns | basic-interpretability-page | 844-850 |
| logit-lens | basic-interpretability-page | 852-858 |
| activation-analysis | basic-interpretability-page | 860-866 |
| probing-experiments | basic-interpretability-page | 868-874 |
| finding-features | basic-interpretability-page | 876-882 |

