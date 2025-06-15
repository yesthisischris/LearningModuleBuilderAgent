"""Reusable LangGraph nodes.

This module contains simple functions implementing each step
of the learning module workflow so they can be composed into
LangGraph graphs.
"""
from __future__ import annotations

import json
import re
from typing import Dict

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from langchain_openai import ChatOpenAI


def clarify(state: Dict, llm: ChatOpenAI) -> Dict:
    """Use the provided topic or ask the user for a concise topic clarification."""
    if "topic" not in state or not state["topic"]:
        question = "Please provide a short topic for the learning module"
        response = llm.invoke(question)
        state["topic"] = response.content
    return state


def plan(state: Dict, llm: ChatOpenAI) -> Dict:
    """Create a brief lesson outline for the clarified topic."""
    prompt = f"""Create a high-level outline for a lesson on {state['topic']}. 
    Keep it concise and focused on the main learning objectives and key topics to cover.
    Format it as a clear, structured outline."""
    response = llm.invoke(prompt)
    state["outline"] = response.content
    return state


def ask_approval(state: Dict, llm: ChatOpenAI) -> Dict:
    """Ask user for approval of the plan before continuing."""
    print("\n" + "="*60)
    print("LESSON PLAN")
    print("="*60)
    print(f"Topic: {state['topic']}")
    print("\nOutline:")
    print(state['outline'])
    print("\n" + "="*60)
    
    while True:
        choice = input("\nWhat would you like to do?\n(a) Approve and continue\n(f) Give feedback for changes\n(q) Quit\nChoice (a/f/q): ").strip().lower()
        
        if choice in ['a', 'approve']:
            state["approved"] = True
            state["needs_feedback"] = False
            break
        elif choice in ['f', 'feedback']:
            feedback = input("\nWhat changes would you like to see in the lesson structure?\n> ")
            state["feedback"] = feedback
            state["approved"] = False
            state["needs_feedback"] = True
            break
        elif choice in ['q', 'quit']:
            state["approved"] = False
            state["needs_feedback"] = False
            break
        else:
            print("Please enter 'a' for approve, 'f' for feedback, or 'q' to quit.")
    return state


def revise_plan(state: Dict, llm: ChatOpenAI) -> Dict:
    """Revise the plan based on user feedback."""
    if not state.get("needs_feedback", False) or not state.get("feedback"):
        return state
    
    print("\n🔄 Revising lesson plan based on your feedback...")
    
    prompt = f"""Revise this lesson outline based on the user's feedback:

Original Topic: {state['topic']}
Original Outline: {state['outline']}

User Feedback: {state['feedback']}

Please create a revised outline that addresses the user's feedback while maintaining educational value and structure. Keep it concise and focused on the main learning objectives."""

    response = llm.invoke(prompt)
    state["outline"] = response.content
    
    # Clear feedback flags so we can ask for approval again
    state["feedback"] = ""
    state["needs_feedback"] = False
    
    return state


def generate(state: Dict, llm: ChatOpenAI) -> Dict:
    """Generate notebook cells for the given outline."""
    if not state.get("approved", False):
        print("Plan was not approved. Exiting.")
        state["cells"] = ""
        return state
    
    print("\n📝 Generating notebook content...")    # Build context from research
    research_context = ""
    
    # Add PyPI package information if available
    if state.get("package_info"):
        research_context += "\n\nPACKAGE INFORMATION FROM PyPI:\n"
        for pkg_name, info in state["package_info"].items():
            research_context += f"- {pkg_name} v{info['version']}: {info['summary']}\n"
            if info.get('docs_url'):
                research_context += f"  Documentation: {info['docs_url']}\n"
    
    if state.get("research_results"):
        research_context += "\n\nCURRENT DOCUMENTATION SOURCES:\n"
        for result in state["research_results"][:5]:  # Use top 5 results
            research_context += f"- {result['title']} ({result.get('package', 'unknown')}): {result['snippet'][:150]}...\n"
    
    if state.get("doc_content"):
        research_context += f"\n\nCURRENT CODE EXAMPLES FROM DOCUMENTATION:\n{state['doc_content'][:2000]}..."
    package_names = state.get("package_names", [])
    packages_str = ", ".join(package_names) if package_names else "Python"
    prompt = """Generate Jupyter notebook cells for this lesson outline:

Topic: {}
Outline: {}
Packages: {}

{}

CRITICAL IMPORTANCE: Use ONLY the NEWEST, CURRENT syntax based on the research above. 
- DO NOT use any deprecated or legacy function names
- VERIFY every function name against the research provided above
- If the research shows multiple syntax options, use the MOST RECENT one
- When in doubt, search for the "latest" or "current" function names in the research

Create a hands-on learning module following this pattern:
- CONCEPT (markdown cell explaining a concept)
- PRACTICE (code cell with example + exercise)
- CONCEPT (markdown cell explaining next concept) 
- PRACTICE (code cell with example + exercise)
- Continue alternating...

Guidelines:
- Skip package installation instructions - assume libraries are already installed
- Focus on practical, hands-on learning with immediate code examples
- Each concept should be concise and immediately followed by practice
- Code cells should include both examples AND exercises for the learner to try
- Use clear, educational comments in code
- Make exercises progressively build on previous concepts
- Use the EXACT function names and syntax patterns shown in the research examples
- If the research shows version-specific changes, use the syntax for the LATEST version shown
- Include comments mentioning if certain functions replaced older deprecated ones

Format the output as a valid JSON array of notebook cells. Each cell must have:
- "cell_type": "markdown" or "code"
- "source": array of strings (each line as a separate string)
- "metadata": object with "language" property ("markdown" for markdown cells, "python" for code cells)

For code cells, also include:
- "execution_count": null
- "outputs": []

Example format:
[
  {{
    "cell_type": "markdown",
    "metadata": {{"language": "markdown"}},
    "source": ["# Title", "Description text"]
  }},
  {{
    "cell_type": "code", 
    "metadata": {{"language": "python"}},
    "source": ["import package", "# Example code"],
    "execution_count": null,
    "outputs": []
  }}
]

Return ONLY the JSON array of cells, no additional text or formatting.""".format(
        state['topic'], 
        state['outline'], 
        packages_str, 
        research_context
    )

    response = llm.invoke(prompt)
    state["cells"] = response.content
    return state


def save_notebook(state: Dict, llm: ChatOpenAI) -> Dict:
    """Save the generated content as a Jupyter notebook file."""
    if not state.get("approved", False) or not state.get("cells"):
        return state
    
    # Create a safe filename from the topic
    topic = state['topic'].replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    filename = ''.join(c for c in topic if c in safe_chars) + '.ipynb'
    
    try:
        # Try to parse the generated cells as JSON
        cells_content = state['cells']
        if cells_content.startswith('```json'):
            cells_content = cells_content.split('```json')[1].split('```')[0].strip()
        elif cells_content.startswith('```'):
            cells_content = cells_content.split('```')[1].split('```')[0].strip()
        try:
            cells = json.loads(cells_content)
            # Ensure each cell has proper metadata with language property
            for cell in cells:
                if "metadata" not in cell:
                    cell["metadata"] = {}
                if "language" not in cell["metadata"]:
                    if cell["cell_type"] == "code":
                        cell["metadata"]["language"] = "python"
                    else:
                        cell["metadata"]["language"] = "markdown"
                
                # Ensure code cells have execution_count and outputs
                if cell["cell_type"] == "code":
                    if "execution_count" not in cell:
                        cell["execution_count"] = None
                    if "outputs" not in cell:
                        cell["outputs"] = []
                        
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing failed: {e}")
            # If parsing fails, create a simple notebook with the content as markdown
            cells = [
                {
                    "cell_type": "markdown",
                    "metadata": {"language": "markdown"},
                    "source": [f"# {state['topic']}\n\n"]
                },
                {
                    "cell_type": "markdown", 
                    "metadata": {"language": "markdown"},
                    "source": [state['outline']]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {"language": "markdown"},
                    "source": ["## Generated Content\n\n", state['cells']]
                }
            ]
    
        # Create the notebook structure
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python", 
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.10.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Save the notebook
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        state["notebook_file"] = filename
        print(f"\n✅ Notebook saved as: {filename}")
        
    except Exception as e:
        print(f"\n❌ Error saving notebook: {e}")
        # Print more details for debugging
        print(f"Error details: {str(e)}")
        state["notebook_file"] = None
    
    return state


def research_package(state: Dict, llm: ChatOpenAI) -> Dict:
    """Research the latest package documentation and examples."""
    if not state.get("approved", False):
        return state
    
    print("\n🔍 Researching latest package documentation and examples...")
    
    # Use LLM to intelligently extract package names from the topic
    topic = state['topic']
    extraction_prompt = f"""From this learning topic: "{topic}"
    
    Extract the main Python package/library names that would be used. Return ONLY a comma-separated list of package names.
    
    Examples:
    - "Intro to NumPy" -> numpy
    - "Data Analysis with Pandas and Matplotlib" -> pandas, matplotlib  
    - "Web Scraping with BeautifulSoup and Requests" -> beautifulsoup4, requests
    - "Machine Learning with scikit-learn" -> scikit-learn
    - "Building APIs with FastAPI" -> fastapi
    - "Geospatial Analysis with H3" -> h3
    - "intro to h3" -> h3
    - "Working with Jupyter Widgets" -> ipywidgets
    - "Time Series Analysis with Prophet" -> prophet
    
    If no specific packages are mentioned, return the most likely packages for the topic.
    If it's a general Python topic without specific packages, return "python".
    
    Package names only, no explanations:"""
    
    try:
        response = llm.invoke(extraction_prompt)
        extracted_packages = response.content.strip().split(',')
        package_names = [pkg.strip() for pkg in extracted_packages if pkg.strip()]
        print(f"📦 Detected packages: {', '.join(package_names)}")
    except Exception as e:
        print(f"⚠️  Package extraction failed: {e}")
        # Fallback to simple keyword extraction
        topic_lower = topic.lower()
        common_packages = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'tensorflow', 'pytorch', 'flask', 'django', 'fastapi', 'requests', 'beautifulsoup']
        package_names = [pkg for pkg in common_packages if pkg in topic_lower]
        if not package_names:
            package_names = [topic.split()[0].lower() if topic.split() else "python"]
    
    research_results = []
    all_doc_content = ""
    package_info = {}
    
    try:
        # First, get package information from PyPI
        for package_name in package_names[:3]:  # Limit to 3 packages
            print(f"   📦 Getting PyPI info for {package_name}...")
            try:
                pypi_url = f"https://pypi.org/pypi/{package_name}/json"
                response = requests.get(pypi_url, timeout=10)
                if response.status_code == 200:
                    pypi_data = response.json()
                    info = pypi_data.get("info", {})
                    package_info[package_name] = {
                        "version": info.get("version", "unknown"),
                        "summary": info.get("summary", ""),
                        "home_page": info.get("home_page", ""),
                        "docs_url": info.get("docs_url", ""),
                        "project_urls": info.get("project_urls", {}),
                        "description": info.get("description", "")[:500]  # Truncate description
                    }
                    print(f"      ✅ Found {package_name} v{package_info[package_name]['version']}")
                else:
                    print(f"      ⚠️  PyPI lookup failed for {package_name} (status: {response.status_code})")
            except Exception as e:
                print(f"      ⚠️  PyPI lookup failed for {package_name}: {e}")
        
        # Research each package with web search
        for package_name in package_names[:3]:  # Limit to 3 packages to avoid too many requests
            print(f"   🔎 Researching {package_name}...")
              # Create diverse search queries, enhanced with version info if available
            version_info = package_info.get(package_name, {}).get("version", "")
            search_queries = [
                f"{package_name} python API reference {version_info} function names",
                f"{package_name} python latest documentation {version_info} current syntax",
                f"{package_name} python migration guide deprecated functions {version_info}",
                f"site:github.com {package_name} python examples {version_info}",
                f"site:readthedocs.io {package_name} API reference"
            ]
            
            ddgs = DDGS()
            
            for query in search_queries[:3]:  # 3 queries per package for better coverage
                try:
                    results = ddgs.text(query, max_results=3)
                    for result in results:
                        # Accept more diverse sources but prioritize official ones
                        priority_domains = ['readthedocs.io', 'docs.python.org', 'github.com', 'pypi.org']
                        secondary_domains = ['stackoverflow.com', 'medium.com', 'towardsdatascience.com', 'realpython.com']
                        
                        if any(domain in result['href'] for domain in priority_domains + secondary_domains):
                            research_results.append({
                                'title': result['title'],
                                'url': result['href'],
                                'snippet': result['body'],
                                'package': package_name
                            })
                except Exception as e:
                    print(f"      ⚠️  Search failed for '{query}': {e}")
                    continue
              # Try to fetch detailed content from the best sources
            package_doc_content = ""
            
            # First try official documentation from PyPI info
            pkg_info = package_info.get(package_name, {})
            docs_urls = []
            if pkg_info.get('docs_url'):
                docs_urls.append(pkg_info['docs_url'])
            if pkg_info.get('project_urls'):
                for url_type, url in pkg_info['project_urls'].items():
                    if any(keyword in url_type.lower() for keyword in ['doc', 'api', 'reference']):
                        docs_urls.append(url)
            
            # Try official docs first
            for docs_url in docs_urls[:2]:
                try:
                    print(f"      📚 Fetching official docs from {docs_url}")
                    response = requests.get(docs_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0 (compatible; LearningBot/1.0)'})
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Extract current API examples
                        code_blocks = soup.find_all(['pre', 'code', 'div'], class_=lambda x: x and ('highlight' in x or 'code' in x or 'example' in x))
                        for block in code_blocks[:10]:
                            code_text = block.get_text().strip()
                            if code_text and len(code_text) > 10:
                                package_doc_content += f"\n\nOfficial docs example for {package_name}:\n{code_text[:600]}...\n"
                        
                        if package_doc_content:  # Found good content, use it
                            break
                except Exception as e:
                    print(f"      ⚠️  Failed to fetch official docs from {docs_url}: {e}")
                    continue            # Then try other research results if we didn't get enough from official docs
            if len(package_doc_content) < 500:
                for result in research_results:
                    if result.get('package') == package_name and any(domain in result['url'] for domain in ['readthedocs.io', 'docs.python.org', 'github.com']):
                        try:
                            response = requests.get(result['url'], timeout=10, headers={'User-Agent': 'Mozilla/5.0 (compatible; LearningBot/1.0)'})
                            if response.status_code == 200:
                                soup = BeautifulSoup(response.content, 'html.parser')
                                
                                # Extract code examples and function signatures
                                code_blocks = soup.find_all(['pre', 'code', 'div'], class_=lambda x: x and ('highlight' in x or 'code' in x or 'example' in x or 'function' in x))
                                
                                # Also look for common code containers and function definitions
                                if not code_blocks:
                                    code_blocks = soup.find_all(['pre', 'code'])
                                
                                # Look for API reference sections specifically
                                api_sections = soup.find_all(['div', 'section'], class_=lambda x: x and ('api' in x or 'reference' in x or 'method' in x))
                                for section in api_sections[:3]:
                                    code_in_section = section.find_all(['code', 'pre'])
                                    code_blocks.extend(code_in_section)
                                
                                for block in code_blocks[:12]:  # Get more examples per package
                                    code_text = block.get_text().strip()
                                    # Prioritize code that shows function calls or imports
                                    if (code_text and len(code_text) > 10 and 
                                        (package_name.replace('-', '') in code_text.lower() or 
                                         package_name.replace('-', '_') in code_text.lower() or
                                         '(' in code_text)):  # Likely a function call
                                        package_doc_content += f"\n\nCode example for {package_name} from {result['url']}:\n{code_text[:600]}...\n"
                                
                                # Look for function definitions and imports more broadly
                                import_statements = soup.find_all(text=re.compile(rf'import.*{package_name.replace("-", "[-_]?")}'))
                                for imp in import_statements[:5]:
                                    package_doc_content += f"\nImport example: {imp.strip()}\n"
                                
                                # Look for deprecated/migration information
                                deprecated_text = soup.find_all(text=re.compile(r'deprecated|migration|replaced|legacy|old.*new', re.IGNORECASE))
                                for dep_text in deprecated_text[:3]:
                                    if package_name.lower() in dep_text.lower():
                                        package_doc_content += f"\nDeprecation note: {dep_text.strip()[:200]}...\n"
                                
                                break  # Got content for this package, move to next
                        except Exception as e:
                            print(f"      ⚠️  Failed to fetch content from {result['url']}: {e}")
                            continue
            
            all_doc_content += package_doc_content
        state["research_results"] = research_results
        state["doc_content"] = all_doc_content
        state["package_names"] = package_names
        state["package_info"] = package_info
        
        print(f"✅ Research completed for {len(package_names)} packages")
        print(f"   Found {len(research_results)} relevant sources")
        if all_doc_content:
            print(f"   Extracted {len(all_doc_content.split('Code example'))-1} code examples")
        
    except Exception as e:
        print(f"⚠️  Research failed: {e}")
        state["research_results"] = []
        state["doc_content"] = ""
        state["package_names"] = package_names
    
    return state
