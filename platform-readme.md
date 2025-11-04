# AI Safety Learning Platform

An interactive platform for learning technical AI safety through hands-on coding exercises.

## Project Structure

```
ai-safety-platform/
├── index.html       # Main platform interface
├── lessons.js       # All lesson content
└── README.md        # This file
```

## Setup

1. Save both files (`index.html` and `lessons.js`) in the same directory
2. Open `index.html` in a web browser
3. Start learning!

## Features

- **Interactive Coding Exercises**: Multiple exercise types including copy, fill-in-the-blank, multiple choice, and construct
- **Progressive Learning**: Start with conceptual understanding, then move to implementation
- **AI Safety Focus**: Every lesson connects technical concepts to AI safety implications
- **No Setup Required**: Works directly in the browser

## Adding New Lessons

To add new lessons, edit `lessons.js` and add your lesson to the `LESSONS` object:

```javascript
LESSONS['your-lesson-id'] = {
    title: "Your Lesson Title",
    steps: [
        {
            instruction: "Step instruction",
            code: "example code",
            explanation: "Why this matters",
            type: "copy", // or "fill-in", "multiple-choice", "construct"
            // Additional properties based on type
        }
    ]
};
```

## Exercise Types

1. **Copy**: Users type the exact code shown
2. **Fill-in**: Complete code with blanks (`___`)
3. **Multiple Choice**: Select from options then type the complete line
4. **Construct**: Write code based on a description

## Future Enhancements

- Real Python execution using Pyodide
- User accounts and progress tracking
- Interactive visualizations
- Community discussions
- More interpretability experiments

## Contributing

This is an open project aimed at lowering barriers to AI safety research. Contributions welcome!

## License

MIT License - Feel free to use and modify for educational purposes.