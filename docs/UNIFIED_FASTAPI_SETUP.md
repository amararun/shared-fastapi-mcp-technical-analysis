# Unified FastAPI Application with Frontend and API Endpoints

This guide explains how to structure a FastAPI application that serves both a frontend interface and API endpoints using a single uvicorn command, similar to the implementation in your markdown-to-pdf converter.

## Project Structure

```
your_project/
├── main.py              # Main FastAPI application
├── static/             # Static files directory
│   ├── css/
│   ├── js/
│   └── images/
├── templates/          # Jinja2 templates
│   └── index.html
└── requirements.txt    # Project dependencies
```

## Dependencies

```txt
fastapi==0.109.2
uvicorn==0.27.1
jinja2==3.1.3          # For template rendering
python-multipart       # For handling form data
```

## Implementation Steps

### 1. Initialize FastAPI with Template Support

```python
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Your App Name")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")
```

### 2. Define Frontend Routes

```python
@app.get("/")
async def read_root(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})
```

### 3. Define API Endpoints

```python
@app.post("/api/your-endpoint")
async def api_endpoint():
    """Your API endpoint logic"""
    return {"message": "API response"}
```

### 4. Create HTML Template with Professional Header and Footer

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Markdown to PDF Converter</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script>
        /* Custom Tailwind Configuration */
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        indigo: {
                            950: '#1e1b4b',
                            900: '#312e81',
                            800: '#3730a3',
                            700: '#4338ca',
                            200: '#c7d2fe',
                            100: '#e0e7ff',
                            50: '#eef2ff',
                        },
                    },
                },
            },
        }
    </script>
    <style>
        /* Custom Gradient Header */
        .header-gradient {
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        }
        .file-drop-zone {
            border: 2px dashed #c7d2fe;
            transition: all 0.3s ease;
        }
        .file-drop-zone:hover {
            border-color: #4338ca;
            background-color: #eef2ff;
        }
        .markdown-input {
            min-height: 200px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        }
    </style>
</head>
<body class="min-h-screen bg-slate-50">
    <!-- Professional Header with Gradient Background -->
    <header class="header-gradient text-white shadow-lg border-b border-white/5">
        <div class="max-w-7xl mx-auto flex items-center gap-4 py-3 px-4">
            <h1 class="text-xl font-semibold whitespace-nowrap tracking-tight">
                Markdown to PDF
            </h1>
            <div class="h-5 w-px bg-indigo-300/20 mx-3"></div>
            <span class="text-base text-indigo-100 font-medium whitespace-nowrap tracking-tight">
                Convert • Preview • Download
            </span>
            <div class="h-5 w-px bg-indigo-300/20 mx-3"></div>
            <div class="flex items-center gap-3">
                <span class="text-[15px] font-medium whitespace-nowrap bg-indigo-800/80 px-4 py-1.5 rounded-md border border-indigo-700 shadow-sm">
                    <span class="text-indigo-200 mr-2">Powered by:</span>
                    <span class="text-white">ReportLab</span>
                </span>
                <img src="{{ url_for('static', path='reportlab-logo.png') }}" alt="ReportLab" class="h-7 w-auto rounded-lg">
            </div>
        </div>
    </header>

    <!-- Main Content Area -->
    <div class="max-w-7xl mx-auto p-4">
        <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <!-- Your content here -->
        </div>
    </div>

    <!-- Professional Footer -->
    <footer class="bg-white/50 border-t border-indigo-100 py-4 mt-8 text-sm text-indigo-950/70">
        <div class="max-w-7xl mx-auto px-4">
            <div class="flex flex-col md:flex-row justify-between items-center gap-4">
                <div class="text-sm text-indigo-950/70 text-center md:text-left">
                    Amar Harolikar <span class="mx-1.5 text-indigo-300">•</span> 
                    Specialist - Decision Sciences & Applied Generative AI
                </div>
                <div class="flex items-center gap-4 text-sm">
                    <a href="https://www.linkedin.com/in/amarharolikar" target="_blank" rel="noopener noreferrer"
                        class="text-indigo-600 hover:text-indigo-700 hover:underline">
                        LinkedIn
                    </a>
                    <a href="https://rex.tigzig.com" target="_blank" rel="noopener noreferrer"
                        class="text-indigo-600 hover:text-indigo-700 hover:underline">
                        rex.tigzig.com
                    </a>
                    <a href="https://tigzig.com" target="_blank" rel="noopener noreferrer"
                        class="text-indigo-600 hover:text-indigo-700 hover:underline">
                        tigzig.com
                    </a>
                </div>
            </div>
        </div>
    </footer>
</body>
</html>
```

### 5. Running the Application

Single command to run both frontend and API:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## URL Structure

- Frontend: `http://localhost:8000/`
- API Endpoints: `http://localhost:8000/api/*`
- Static Files: `http://localhost:8000/static/*`

## Key Features

1. **Unified Server**: 
   - One server process handles both frontend and API
   - No need for separate frontend server

2. **Template Integration**:
   - Uses Jinja2 for server-side rendering
   - Can pass data from backend to frontend

3. **Static File Handling**:
   - Serves static files (CSS, JS, images) efficiently
   - Proper caching and performance optimization

4. **API Integration**:
   - Frontend can call API endpoints using fetch/axios
   - Same-origin requests, no CORS issues

## Design System Details

### Actual Header Implementation (Required Elements)

Your application must use the following header exactly as shown:

```html
<header class="header-gradient text-white shadow-lg border-b border-white/5">
    <div class="max-w-7xl mx-auto flex items-center gap-4 py-3 px-4">
        <h1 class="text-xl font-semibold whitespace-nowrap tracking-tight">
            Markdown to PDF
        </h1>
        <div class="h-5 w-px bg-indigo-300/20 mx-3"></div>
        <span class="text-base text-indigo-100 font-medium whitespace-nowrap tracking-tight">
            Convert • Preview • Download
        </span>
        <div class="h-5 w-px bg-indigo-300/20 mx-3"></div>
        <div class="flex items-center gap-3">
            <span class="text-[15px] font-medium whitespace-nowrap bg-indigo-800/80 px-4 py-1.5 rounded-md border border-indigo-700 shadow-sm">
                <span class="text-indigo-200 mr-2">Powered by:</span>
                <span class="text-white">ReportLab</span>
            </span>
            <img src="{{ url_for('static', path='reportlab-logo.png') }}" alt="ReportLab" class="h-7 w-auto rounded-lg">
        </div>
    </div>
</header>
```

### Actual Footer Implementation (Required Elements)

Your application must use the following footer exactly as shown:

```html
<footer class="bg-white/50 border-t border-indigo-100 py-4 mt-8 text-sm text-indigo-950/70">
    <div class="max-w-7xl mx-auto px-4">
        <div class="flex flex-col md:flex-row justify-between items-center gap-4">
            <div class="text-sm text-indigo-950/70 text-center md:text-left">
                Amar Harolikar <span class="mx-1.5 text-indigo-300">•</span> 
                Specialist - Decision Sciences & Applied Generative AI
            </div>
            <div class="flex items-center gap-4 text-sm">
                <a href="https://www.linkedin.com/in/amarharolikar" target="_blank" rel="noopener noreferrer"
                    class="text-indigo-600 hover:text-indigo-700 hover:underline">
                    LinkedIn
                </a>
                <a href="https://rex.tigzig.com" target="_blank" rel="noopener noreferrer"
                    class="text-indigo-600 hover:text-indigo-700 hover:underline">
                    rex.tigzig.com
                </a>
                <a href="https://tigzig.com" target="_blank" rel="noopener noreferrer"
                    class="text-indigo-600 hover:text-indigo-700 hover:underline">
                    tigzig.com
                </a>
            </div>
        </div>
    </div>
</footer>
```

Important notes:
- The name "Amar Harolikar" must be preserved
- The title "Specialist - Decision Sciences & Applied Generative AI" must be preserved
- All three links (LinkedIn, rex.tigzig.com, tigzig.com) must be preserved exactly as shown
- The styling and layout must match exactly as shown

### Color Scheme

The application uses a professional indigo color palette:

```
Indigo Colors:
- Darkest: #1e1b4b (indigo-950) - Used for header gradient start
- Dark: #312e81 (indigo-900) - Used for header gradient end
- Medium: #3730a3 (indigo-800) - Used for UI elements background
- Light: #4338ca (indigo-700) - Used for UI borders
- Lighter: #c7d2fe (indigo-200) - Used for UI accents
- Lightest: #e0e7ff (indigo-100) - Used for background accents
- Faintest: #eef2ff (indigo-50) - Used for hover states
```

### Header Specifications

The professional header includes:

1. **Gradient Background**: 
   - Linear gradient from indigo-950 to indigo-900
   - Creates depth and visual interest

2. **Layout**:
   - Height: Automatic based on content with `py-3` (padding-y: 0.75rem)
   - Maximum width: 7xl (80rem/1280px) with auto margins
   - Flexible layout with items centered vertically
   - Gap between items: 1rem

3. **Typography**:
   - App name: text-xl (1.25rem), semibold, tight tracking
   - Tagline: text-base (1rem), medium weight, tight tracking, lighter color

4. **Dividers**:
   - Vertical dividers: 5px tall, 1px wide with semi-transparent color
   - Horizontal spacing: 0.75rem (mx-3)

5. **Accent Elements**:
   - "Powered by" badge: Darker background, rounded corners, subtle border
   - Logo image: Fixed height (7px), auto width, rounded corners

### Footer Specifications

The footer includes:

1. **Background**: 
   - Semi-transparent white with top border

2. **Layout**:
   - Padding: py-4 (1rem vertical), px-4 (1rem horizontal)
   - Responsive: Column on mobile, row on medium screens and up
   - Flexible spacing with items centered

3. **Typography**:
   - Small text (0.875rem)
   - Muted indigo color (indigo-950 at 70% opacity)
   - Links: indigo-600 with hover state (indigo-700 + underline)

## Button Styling

Standard button styling:

```
Primary buttons: 
- Background: indigo-600
- Hover: indigo-700
- Text: white
- Padding: px-6 py-2.5
- Border radius: rounded-lg
- Font weight: medium
- Shadow: shadow-sm
- Transitions: transition-colors

Secondary/success buttons:
- Background: green-600
- Hover: green-700
- Disabled state: gray-400 with reduced opacity
```

## Best Practices

1. **Directory Structure**:
   - Keep templates and static files separate
   - Organize API endpoints logically

2. **Error Handling**:
   - Use proper HTTP status codes
   - Implement error pages for frontend

3. **Security**:
   - Implement CORS if needed
   - Validate all inputs

4. **Performance**:
   - Use async/await for I/O operations
   - Implement caching where appropriate

## Example: Frontend Calling API

```javascript
// In your frontend JavaScript
async function callApi() {
    const response = await fetch('/api/your-endpoint', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });
    const result = await response.json();
    // Handle result
}
```

## Advantages of This Approach

1. **Simplicity**: Single codebase, single deployment
2. **Development Speed**: Quick to set up and modify
3. **Resource Efficiency**: One process handles everything
4. **Easy Maintenance**: No need to coordinate multiple services
5. **Built-in Features**: FastAPI provides automatic API documentation
6. **Consistent Design**: Shared templates ensure UI consistency

This structure allows you to serve both your frontend and API from a single FastAPI application, making development and deployment simpler while maintaining good separation of concerns and a professional design language. 